#include "dsv4-comp-pool.cuh"

// DeepSeek V4(.1) KV compressor pooling (GGML_OP_DSV4_COMP_POOL, see ggml.h). The graph form concatenated the state
// rows with the current rows (and a zero row), gathered the candidate rows, copied the two halves of the overlapped
// rows apart, concatenated them, transposed values and scores, then ran soft_max, mul, sum_rows, a transpose back, the
// RMS norm and the rope: ~20 kernels per compressor and token, three compressors per CSA/HCA layer pair. Here one
// workgroup per compressed block: G groups of E/2 threads, thread t of a group owns dims (2t, 2t + 1) and walks every
// G-th candidate row (an online softmax per dim; the candidate row offsets are resolved once into LDS, the loads are
// branch-free so the unrolled walk keeps several rows in flight), the groups merge through LDS, then the workgroup
// reduces the sum of squares for the norm and the threads in the rope range rotate their pair.

struct dsv4cp_rope {
    int   dims, offs;
    float theta_scale, freq_scale, ext_factor, attn_factor, corr0, corr1;
};

static constexpr int DSV4CP_MAX_CAND = 256;

template <int NP, int G>
static __global__ void __launch_bounds__(NP*G) dsv4cp_kernel(
        const float * __restrict__ bkv, const float * __restrict__ ckv, const float * __restrict__ bsc,
        const float * __restrict__ csc, const int32_t * __restrict__ idx, const float * __restrict__ norm,
        const int32_t * __restrict__ pos, float * __restrict__ dst,
        const int E, const int ratio, const int overlap, const int n_blocks, const int n_base, const int n_cur,
        const int64_t s_bkv, const int64_t s_ckv, const int64_t s_bsc, const int64_t s_csc, const float eps,
        const dsv4cp_rope rp, const bool skip_dummy) {
    const int b  = blockIdx.x;
    const int t  = threadIdx.x % NP; // pair within the group
    const int g  = threadIdx.x / NP; // candidate group
    const int e  = 2*t;
    const bool active = e < E;

    const int n_cand = overlap ? 2*ratio : ratio;

    // a padding block (llama-kv-cache-dsv4.cpp keeps the block count of a step fixed: a decode step that completes no
    // block still pools one, every candidate the same state row, into the masked last cache slot; HCA pools 128 rows):
    // zeros instead. Real blocks read the distinct rows of consecutive positions in their current window (the first
    // block's previous window repeats the zero row, so it is not the test)
    if (skip_dummy && ratio > 1) {
        const int j0 = (overlap ? n_blocks*ratio : 0) + b*ratio;
        if (idx[j0] == idx[j0 + 1]) {
            if (g == 0 && active) {
                *(float2 *) (dst + (int64_t) b*E + e) = make_float2(0.0f, 0.0f);
            }
            return;
        }
    }

    // candidate c: (kv offset, score offset) in floats from the base or cur pointer (bit 62: cur), or -1 (zero row)
    __shared__ int64_t c_kv[DSV4CP_MAX_CAND];
    __shared__ int64_t c_sc[DSV4CP_MAX_CAND];
    for (int c = threadIdx.x; c < n_cand; c += NP*G) {
        const bool second = overlap && c >= ratio;
        const int  j      = second ? n_blocks*ratio + b*ratio + (c - ratio) : b*ratio + c;
        const int  off    = second ? E : 0;
        const int  r      = idx[j];
        int64_t okv = -1, osc = -1;
        if (r < n_base) {
            okv = (int64_t) r*s_bkv + off;
            osc = (int64_t) r*s_bsc + off;
        } else if (r < n_base + n_cur) {
            okv = ((int64_t) 1 << 62) | ((int64_t) (r - n_base)*s_ckv + off);
            osc = ((int64_t) 1 << 62) | ((int64_t) (r - n_base)*s_csc + off);
        }
        c_kv[c] = okv;
        c_sc[c] = osc;
    }
    __syncthreads();

    float m0 = -INFINITY, m1 = -INFINITY, s0 = 0.0f, s1 = 0.0f, a0 = 0.0f, a1 = 0.0f;
    if (active) {
#pragma unroll 8
        for (int c = g; c < n_cand; c += G) {
            const int64_t okv = c_kv[c];
            const int64_t osc = c_sc[c];
            const bool    ok  = okv >= 0;
            const bool    cur = (okv >> 62) & 1;
            const int64_t mk  = ((int64_t) 1 << 62) - 1;
            const float * kvp = (cur ? ckv : bkv) + (ok ? (okv & mk) : 0) + e;
            const float * scp = (cur ? csc : bsc) + (ok ? (osc & mk) : 0) + e;
            const float2 v  = *(const float2 *) kvp;
            float2       sv = *(const float2 *) scp;
            if (!ok) {
                sv = make_float2(-INFINITY, -INFINITY); // the zero row: kv 0 under a -inf score
            }
            if (sv.x > m0) {
                const float f = expf(m0 - sv.x);
                s0 *= f; a0 *= f; m0 = sv.x;
            }
            if (sv.y > m1) {
                const float f = expf(m1 - sv.y);
                s1 *= f; a1 *= f; m1 = sv.y;
            }
            const float p0 = sv.x > -INFINITY ? expf(sv.x - m0) : 0.0f;
            const float p1 = sv.y > -INFINITY ? expf(sv.y - m1) : 0.0f;
            s0 += p0; a0 += p0*v.x;
            s1 += p1; a1 += p1*v.y;
        }
    }

    if (G > 1) {
        // merge the groups' (max, sum, acc) per dim into group 0
        __shared__ float4 part[G > 1 ? (G - 1)*NP : 1][2];
        if (g > 0) {
            part[(g - 1)*NP + t][0] = make_float4(m0, s0, a0, 0.0f);
            part[(g - 1)*NP + t][1] = make_float4(m1, s1, a1, 0.0f);
        }
        __syncthreads();
        if (g == 0) {
#pragma unroll
            for (int k = 0; k < G - 1; ++k) {
                const float4 p0 = part[k*NP + t][0];
                const float4 p1 = part[k*NP + t][1];
                const float M0 = fmaxf(m0, p0.x);
                const float M1 = fmaxf(m1, p1.x);
                const float f0 = m0 > -INFINITY ? expf(m0 - M0) : 0.0f, q0 = p0.x > -INFINITY ? expf(p0.x - M0) : 0.0f;
                const float f1 = m1 > -INFINITY ? expf(m1 - M1) : 0.0f, q1 = p1.x > -INFINITY ? expf(p1.x - M1) : 0.0f;
                s0 = s0*f0 + p0.y*q0; a0 = a0*f0 + p0.z*q0; m0 = M0;
                s1 = s1*f1 + p1.y*q1; a1 = a1*f1 + p1.z*q1; m1 = M1;
            }
        }
    }

    const bool owner = g == 0 && active; // group 0 finishes the pairs
    float c0 = owner ? a0/s0 : 0.0f;
    float c1 = owner ? a1/s1 : 0.0f;

    // sum of squares over the block's E dims (the other groups' waves add zeros)
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    float ss = warp_reduce_sum<ws>(c0*c0 + c1*c1);
    if (NP*G > ws) {
        __shared__ float wsum[NP*G/ws > 0 ? NP*G/ws : 1];
        if (threadIdx.x % ws == 0) {
            wsum[threadIdx.x/ws] = ss;
        }
        __syncthreads();
        ss = 0.0f;
#pragma unroll
        for (int w = 0; w < NP*G/ws; ++w) {
            ss += wsum[w];
        }
    }
    if (!owner) {
        return;
    }
    const float scale = rsqrtf(ss/E + eps);
    c0 *= scale*norm[e];
    c1 *= scale*norm[e + 1];

    if (pos && e >= rp.offs && e < rp.offs + rp.dims) {
        // NORM rope of the pair (rope.cu's rope_norm + rope_yarn math)
        const int   iw           = e - rp.offs;
        const float theta_extrap = pos[b]*powf(rp.theta_scale, iw/2.0f);
        float theta  = rp.freq_scale*theta_extrap;
        float mscale = rp.attn_factor;
        if (rp.ext_factor != 0.0f) {
            const float y = (iw/2 - rp.corr0) / max(0.001f, rp.corr1 - rp.corr0);
            const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y)))*rp.ext_factor;
            theta = theta*(1 - ramp_mix) + theta_extrap*ramp_mix;
            mscale *= 1.0f + 0.1f*logf(1.0f/rp.freq_scale);
        }
        const float cs = cosf(theta)*mscale;
        const float sn = sinf(theta)*mscale;
        const float x0 = c0, x1 = c1;
        c0 = x0*cs - x1*sn;
        c1 = x0*sn + x1*cs;
    }

    *(float2 *) (dst + (int64_t) b*E + e) = make_float2(c0, c1);
}

bool ggml_cuda_dsv4_comp_pool_supported(const ggml_tensor * dst) {
    const ggml_tensor * bkv = dst->src[0];
    const ggml_tensor * ckv = dst->src[1];
    const ggml_tensor * bsc = dst->src[2];
    const ggml_tensor * csc = dst->src[3];
    const ggml_tensor * idx = dst->src[4];
    const ggml_tensor * nrm = dst->src[5];
    const ggml_tensor * pos = dst->src[6];
    const int64_t E = dst->ne[0];
    const int ratio   = ggml_get_op_params_i32(dst, 0);
    const int overlap = ggml_get_op_params_i32(dst, 1);
    const auto rows_ok = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->nb[1] % 8 == 0 && ((uintptr_t) t->data) % 8 == 0;
    };
    return E % 2 == 0 && E <= 1024 && (overlap ? 2*ratio : ratio) <= DSV4CP_MAX_CAND &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        rows_ok(bkv) && rows_ok(ckv) && rows_ok(bsc) && rows_ok(csc) &&
        idx->type == GGML_TYPE_I32 && ggml_is_contiguous(idx) && nrm->type == GGML_TYPE_F32 && ggml_is_contiguous(nrm) &&
        (pos == nullptr || (pos->type == GGML_TYPE_I32 && ggml_is_contiguous(pos))) &&
        bkv->ne[1] + ckv->ne[1] < INT32_MAX && ggml_nelements(idx) < INT32_MAX;
}

void ggml_cuda_dsv4_comp_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * bkv = dst->src[0];
    const ggml_tensor * ckv = dst->src[1];
    const ggml_tensor * bsc = dst->src[2];
    const ggml_tensor * csc = dst->src[3];
    const ggml_tensor * idx = dst->src[4];
    const ggml_tensor * nrm = dst->src[5];
    const ggml_tensor * pos = dst->src[6];

    const int   E        = (int) dst->ne[0];
    const int   n_blocks = (int) dst->ne[2];
    const int   ratio    = ggml_get_op_params_i32(dst, 0);
    const int   overlap  = ggml_get_op_params_i32(dst, 1);
    const float eps      = ggml_get_op_params_f32(dst, 2);

    dsv4cp_rope rp = {0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (pos) {
        rp.dims = ggml_get_op_params_i32(dst, 3);
        rp.offs = ggml_get_op_params_i32(dst, 4);
        const int   n_ctx_orig = ggml_get_op_params_i32(dst, 5);
        const float freq_base  = ggml_get_op_params_f32(dst, 6);
        rp.freq_scale  = ggml_get_op_params_f32(dst, 7);
        rp.ext_factor  = ggml_get_op_params_f32(dst, 8);
        rp.attn_factor = ggml_get_op_params_f32(dst, 9);
        float corr[2];
        ggml_rope_yarn_corr_dims(rp.dims, n_ctx_orig, freq_base, ggml_get_op_params_f32(dst, 10),
                ggml_get_op_params_f32(dst, 11), corr);
        rp.corr0 = corr[0];
        rp.corr1 = corr[1];
        rp.theta_scale = powf(freq_base, -2.0f/rp.dims);
    }
    if (n_blocks == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int n_pairs = E/2;
    const int n_cand  = overlap ? 2*ratio : ratio;
    // GGML_CUDA_DSV4CP_G: candidate groups per workgroup for the long (HCA, ratio 128) pools (default 4)
    static const int g_env = [] { const char * e = getenv("GGML_CUDA_DSV4CP_G"); return e ? atoi(e) : 4; }();
    // GGML_CUDA_DSV4CP_SKIP_DUMMY=0: pool the padding blocks too (DeepSeek V4 Flash decode: ~1.2 ms/token of pooling)
    static const bool skip_dummy = [] { const char * e = getenv("GGML_CUDA_DSV4CP_SKIP_DUMMY"); return !e || atoi(e) != 0; }();
    const int G = n_cand >= 32 ? g_env : 1;
#define DSV4CP_LAUNCH(NP, G) dsv4cp_kernel<NP, G><<<n_blocks, NP*G, 0, stream>>>( \
        (const float *) bkv->data, (const float *) ckv->data, (const float *) bsc->data, (const float *) csc->data, \
        (const int32_t *) idx->data, (const float *) nrm->data, pos ? (const int32_t *) pos->data : nullptr, \
        (float *) dst->data, E, ratio, overlap, n_blocks, (int) bkv->ne[1], (int) ckv->ne[1], \
        (int64_t) (bkv->nb[1]/sizeof(float)), (int64_t) (ckv->nb[1]/sizeof(float)), \
        (int64_t) (bsc->nb[1]/sizeof(float)), (int64_t) (csc->nb[1]/sizeof(float)), eps, rp, skip_dummy)
#define DSV4CP_LAUNCH_G(NP) \
    if (G >= 4 && NP*4 <= 1024) { DSV4CP_LAUNCH(NP, 4); } else if (G >= 2 && NP*2 <= 1024) { DSV4CP_LAUNCH(NP, 2); } else { DSV4CP_LAUNCH(NP, 1); }
    if (n_pairs <= 64) {
        DSV4CP_LAUNCH_G(64);
    } else if (n_pairs <= 128) {
        DSV4CP_LAUNCH_G(128);
    } else if (n_pairs <= 256) {
        DSV4CP_LAUNCH_G(256);
    } else {
        DSV4CP_LAUNCH_G(512);
    }
#undef DSV4CP_LAUNCH_G
#undef DSV4CP_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
