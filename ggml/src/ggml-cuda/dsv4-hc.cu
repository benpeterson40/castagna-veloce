#include "common.cuh"
#include "convert.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// RAWP: post holds the raw inject logits and the weight is ps2 * sigmoid(ps1 * post) (the SCALE -> SIGMOID -> SCALE
// chain folded in)
template <bool has_comb, bool RAWP = false>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float   ps1 = 1.0f,
        float   ps2 = 1.0f) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float pw = post[idst*sp0 + it*sp1];
    if constexpr (RAWP) {
        pw = ps2 / (1.0f + expf(-ps1*pw));
    }
    float sum = x[i0*sx0 + it*sx1] * pw;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// prefill-sized DSV4_HC_POST with comb (GGML_CUDA_HC_POST4=0 off): the generic kernel above runs a thread per output
// value with 64-bit index divisions and reads every residual value once per output stream (~265 GB/s, 0.68 ms at 1024
// tokens). Here a thread owns 4 consecutive embedding values of one token for all 4 output streams (float4 loads, the
// token's post/comb weights are block-uniform), with the same per-value operation order.
static __global__ void __launch_bounds__(256) dsv4_hc_post4_f32(
        const float * __restrict__ x, const float * __restrict__ residual, const float * __restrict__ post,
        const float * __restrict__ comb, float * __restrict__ dst, const int n4,
        const int64_t sx1, const int64_t sr1, const int64_t sr2, const int64_t sp0, const int64_t sp1,
        const int64_t sc0, const int64_t sc1, const int64_t sc2, const int64_t sd1, const int64_t sd2) {
    const int64_t it = blockIdx.y;
    const int     i4 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i4 >= n4) {
        return;
    }
    float pw[DSV4_HC];
    float cm[DSV4_HC][DSV4_HC];
#pragma unroll
    for (int d = 0; d < DSV4_HC; ++d) {
        pw[d] = post[d*sp0 + it*sp1];
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            cm[d][s] = comb[d*sc0 + s*sc1 + it*sc2];
        }
    }
    const float4 xv = ((const float4 *) (x + it*sx1))[i4];
    float4 r[DSV4_HC];
#pragma unroll
    for (int s = 0; s < DSV4_HC; ++s) {
        r[s] = ((const float4 *) (residual + s*sr1 + it*sr2))[i4];
    }
#pragma unroll
    for (int d = 0; d < DSV4_HC; ++d) {
        float4 o;
        o.x = xv.x*pw[d]; o.y = xv.y*pw[d]; o.z = xv.z*pw[d]; o.w = xv.w*pw[d];
#pragma unroll
        for (int s = 0; s < DSV4_HC; ++s) {
            o.x += r[s].x*cm[d][s]; o.y += r[s].y*cm[d][s]; o.z += r[s].z*cm[d][s]; o.w += r[s].w*cm[d][s];
        }
        ((float4 *) (dst + d*sd1 + it*sd2))[i4] = o;
    }
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    static const bool post4 = [] { const char * e = getenv("GGML_CUDA_HC_POST4"); return !e || atoi(e) != 0; }();
    const auto al16 = [](const void * p) { return ((uintptr_t) p & 15) == 0; };
    if (post4 && comb && hc == DSV4_HC && n_embd % 4 == 0 && n_tokens >= 16 && n_tokens <= 65535 &&
            nbx0 == sizeof(float) && nbr0 == sizeof(float) && nbd0 == sizeof(float) &&
            nbx1 % 16 == 0 && nbr1 % 16 == 0 && nbr2 % 16 == 0 && nbd1 % 16 == 0 && nbd2 % 16 == 0 &&
            al16(x->data) && al16(residual->data) && al16(dst->data)) {
        const int n4 = (int) (n_embd/4);
        dsv4_hc_post4_f32<<<dim3((n4 + 255)/256, (unsigned) n_tokens, 1), 256, 0, ctx.stream()>>>(
                (const float *) x->data, (const float *) residual->data, (const float *) post->data,
                (const float *) comb->data, (float *) dst->data, n4,
                nbx1 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
                nbp0 / sizeof(float), nbp1 / sizeof(float),
                nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
                nbd1 / sizeof(float), nbd2 / sizeof(float));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float), 1.0f, 1.0f);
}

// DSV4_HC_POST whose post weights come from SCALE(SIGMOID(SCALE(raw))): the chain is computed per element here
void ggml_cuda_op_dsv4_hc_post_rawpost(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * raw,
                                       const float ps1, const float ps2) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    GGML_ASSERT(dst->src[3] == nullptr);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 && raw->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, raw,      nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    dsv4_hc_post_f32<false, true><<<(nr + block_size - 1) / block_size, block_size, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) residual->data, (const float *) raw->data, nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            0, 0, 0,
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float), ps1, ps2);
    CUDA_CHECK(cudaGetLastError());
}

// DSV4_HC_MIX: RMS norm of the flattened stream, the (2 + hc)*hc mix dot products, the pre/post sigmoid gates and the
// Sinkhorn comb in one launch (at decode the unfused chain was 12 kernels, ~80 us per sublayer on gfx906). Each
// workgroup takes HCMIX_CHUNK flat elements of one token for all mix rows and writes its partial dot products and sum of
// squares; the last workgroup of the token (arrival counter, reset by that workgroup) sums the partials in workgroup
// order, so the result does not depend on the arrival order.
static constexpr int HCMIX_NMIX    = (2 + DSV4_HC)*DSV4_HC;
static constexpr int HCMIX_CHUNK   = 1024;
static constexpr int HCMIX_THREADS = 256;
static constexpr int HCMIX_PSTRIDE = 32;
static constexpr int HCMIX_MAX_TOK = 65535;

template <ggml_type type>
static __global__ void __launch_bounds__(HCMIX_THREADS) dsv4_hc_mix_f32(
        const float * __restrict__ x, const char * __restrict__ fn, const float * __restrict__ scale,
        const float * __restrict__ base, float * __restrict__ dst, float * __restrict__ part, int * __restrict__ counters,
        const int n_embd, const int K, const int64_t sx1, const int64_t sx2, const int64_t s_fn, const int64_t sd1,
        const float eps, const float hc_eps, const int n_iter) {
    constexpr int NW = HCMIX_THREADS/WARP_SIZE;

    const int t    = blockIdx.y;
    const int wg   = blockIdx.x;
    const int nwg  = gridDim.x;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int wv   = tid / WARP_SIZE;
    const int k0   = wg*HCMIX_CHUNK;

    __shared__ float xs[HCMIX_CHUNK];
    __shared__ float rs[HCMIX_NMIX];
    __shared__ float red[NW][HCMIX_NMIX + 1];
    __shared__ int   s_last;

    const float * xt = x + t*sx2;
    float ss = 0.0f;
    auto load_x = [&]() {
#pragma unroll
        for (int e = tid; e < HCMIX_CHUNK; e += HCMIX_THREADS) {
            const int k = k0 + e;
            float v = 0.0f;
            if (k < K) {
                const int h = k / n_embd;
                v = xt[(k - h*n_embd) + h*sx1];
            }
            xs[e] = v;
            ss += v*v;
        }
        __syncthreads();
    };

    if constexpr (type == GGML_TYPE_Q2_K) {
        // thread = (row group rg = tid/64, super-block b of the chunk, qs dword q): rows rg, rg + 4, ...; the dword holds 4
        // values of 4 sub-blocks (one per 2-bit shift j), elements 128*n + 32*j + l0 + 0..3 with scale byte 8*n + 2*j + hh
        static_assert(HCMIX_CHUNK == 4*QK_K && HCMIX_THREADS == 256 && HCMIX_NMIX % 4 == 0, "q2_K thread mapping");
        constexpr int RPG = HCMIX_NMIX/4;
        constexpr int PW  = ggml_cuda_get_physical_warp_size();
        static_assert(64 % PW == 0, "a row group spans whole warps");
        const int rg = tid >> 6;
        const int b  = (tid >> 4) & 3;
        const int q  = tid & 15;
        const int n  = q >> 3;
        const int l0 = 4*(q & 7);
        const int hh = l0 >> 4;
        const int kb = k0/QK_K + b;
        const bool kb_ok = kb < K/QK_K;

        // the weight words first, so their loads overlap the x chunk's
        uint32_t wv_[RPG], wsc0[RPG], wsc1[RPG], wdm[RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const char * blk = fn + (rg + 4*rr)*s_fn + (int64_t) (kb_ok ? kb : 0)*sizeof(block_q2_K);
            wv_[rr]  = *(const uint32_t *) (blk + 16 + 4*q);
            wsc0[rr] = *(const uint32_t *) (blk + 8*n);
            wsc1[rr] = *(const uint32_t *) (blk + 8*n + 4);
            wdm[rr]  = *(const uint32_t *) (blk + 80);
        }
        load_x();

        float xv[16];
        float sx[4];
        const float * xb = xs + QK_K*b + 128*n + l0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (xb + 32*j);
            xv[4*j + 0] = v.x; xv[4*j + 1] = v.y; xv[4*j + 2] = v.z; xv[4*j + 3] = v.w;
            sx[j] = (v.x + v.y) + (v.z + v.w);
        }

        float acc[RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            acc[rr] = 0.0f;
        }
        if (kb_ok) {
#pragma unroll
            for (int rr = 0; rr < RPG; ++rr) {
                const uint32_t v   = wv_[rr];
                const uint32_t sc0 = wsc0[rr];
                const uint32_t sc1 = wsc1[rr];
                const uint32_t dmw = wdm[rr];
                const float d    = __half2float(__ushort_as_half((unsigned short) (dmw & 0xFFFF)));
                const float dmin = __half2float(__ushort_as_half((unsigned short) (dmw >> 16)));
                float sq = 0.0f;
                float sm = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t scw = j < 2 ? sc0 : sc1;
                    const uint32_t sb  = (scw >> (8*(2*(j & 1) + hh))) & 0xFF;
                    const uint32_t qq  = (v >> (2*j)) & 0x03030303u;
                    const float dot = (float) (qq & 0xFF)*xv[4*j + 0] + (float) ((qq >> 8) & 0xFF)*xv[4*j + 1] +
                                      (float) ((qq >> 16) & 0xFF)*xv[4*j + 2] + (float) (qq >> 24)*xv[4*j + 3];
                    sq += (float) (sb & 0xF)*dot;
                    sm += (float) (sb >> 4)*sx[j];
                }
                acc[rr] = d*sq - dmin*sm;
            }
        }
        // per row: sum over the 64 threads of its row group (64/PW physical warps)
        __shared__ float wsum[HCMIX_THREADS/PW][RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            acc[rr] = warp_reduce_sum<PW>(acc[rr]);
        }
        if (tid % PW == 0) {
#pragma unroll
            for (int rr = 0; rr < RPG; ++rr) {
                wsum[tid/PW][rr] = acc[rr];
            }
        }
        __syncthreads();
        if (tid < HCMIX_NMIX) {
            const int rgr = tid % 4;
            const int rr  = tid / 4;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < 64/PW; ++w) {
                a += wsum[rgr*(64/PW) + w][rr];
            }
            rs[tid] = a;
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // q8_0 (GLM-5-Next): thread = (row group rg = tid/64, 256-value span b of the chunk, half block q): rows rg, rg + 4,
        // ...; half q%2 of block (k0 + 256*b)/32 + q/2, its 16 quants from 5 aligned dwords (they start 0 or 2 bytes into a
        // dword: alignbyte) and the f16 scale from the dword holding it, all loaded before the x chunk
        static_assert(HCMIX_CHUNK == 1024 && HCMIX_THREADS == 256 && HCMIX_NMIX % 4 == 0, "q8_0 thread mapping");
        constexpr int RPG = HCMIX_NMIX/4;
        constexpr int PW  = ggml_cuda_get_physical_warp_size();
        static_assert(64 % PW == 0, "a row group spans whole warps");
        const int rg  = tid >> 6;
        const int b   = (tid >> 4) & 3;
        const int q   = tid & 15;
        const int e   = 256*b + 16*q;          // element offset in the chunk
        const bool ok = k0 + e < K;            // K % 32 == 0: whole half blocks
        const int blk = (ok ? k0 + e : 0)/QK8_0;
        const int so  = blk*(int) sizeof(block_q8_0);
        const int off = so + 2 + 16*(q & 1);
        const int sh  = off & 3;
        const int ssh = so & 3;

        uint32_t w8[RPG][5], wsc[RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const char * row = fn + (rg + 4*rr)*s_fn;
            const uint32_t * pw8 = (const uint32_t *) (row + (off - sh));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w8[rr][j] = pw8[j];
            }
            w8[rr][4] = pw8[sh ? 4 : 3];
            wsc[rr]   = *(const uint32_t *) (row + (so - ssh));
        }
        load_x();

        float xv[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (xs + e + 4*j);
            xv[4*j + 0] = v.x; xv[4*j + 1] = v.y; xv[4*j + 2] = v.z; xv[4*j + 3] = v.w;
        }
        float acc[RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const float d = __half2float(__ushort_as_half((unsigned short) (ssh ? wsc[rr] >> 16 : wsc[rr] & 0xFFFF)));
            float sq = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t qd = __builtin_amdgcn_alignbyte(w8[rr][j + 1], w8[rr][j], sh);
                sq += (float) ((int) (qd << 24) >> 24)*xv[4*j + 0] + (float) ((int) (qd << 16) >> 24)*xv[4*j + 1] +
                      (float) ((int) (qd <<  8) >> 24)*xv[4*j + 2] + (float) ((int) qd >> 24)*xv[4*j + 3];
            }
            acc[rr] = ok ? d*sq : 0.0f;
        }
        // per row: sum over the 64 threads of its row group (64/PW physical warps)
        __shared__ float wsum8[HCMIX_THREADS/PW][RPG];
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            acc[rr] = warp_reduce_sum<PW>(acc[rr]);
        }
        if (tid % PW == 0) {
#pragma unroll
            for (int rr = 0; rr < RPG; ++rr) {
                wsum8[tid/PW][rr] = acc[rr];
            }
        }
        __syncthreads();
        if (tid < HCMIX_NMIX) {
            const int rgr = tid % 4;
            const int rr  = tid / 4;
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < 64/PW; ++w) {
                a += wsum8[rgr*(64/PW) + w][rr];
            }
            rs[tid] = a;
        }
    } else {
        // f32 / f16 weights: 4 consecutive elements per thread for every row
        load_x();
        const int e0 = 4*tid;
        const int k  = k0 + e0;
        const float4 xv = *(const float4 *) (xs + e0);
        float acc[HCMIX_NMIX];
#pragma unroll
        for (int r = 0; r < HCMIX_NMIX; ++r) {
            float a = 0.0f;
            if (k < K) {
                float4 w;
                if constexpr (type == GGML_TYPE_F32) {
                    w = *(const float4 *) (fn + r*s_fn + (int64_t) k*sizeof(float));
                } else {
                    const uint2 u = *(const uint2 *) (fn + r*s_fn + (int64_t) k*sizeof(half));
                    const float2 lo = __half22float2(*(const half2 *) &u.x);
                    const float2 hi = __half22float2(*(const half2 *) &u.y);
                    w = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                a = (w.x*xv.x + w.y*xv.y) + (w.z*xv.z + w.w*xv.w);
            }
            acc[r] = warp_reduce_sum(a);
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < HCMIX_NMIX; ++r) {
                red[wv][r] = acc[r];
            }
        }
        __syncthreads();
        if (tid < HCMIX_NMIX) {
            float a = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                a += red[w][tid];
            }
            rs[tid] = a;
        }
    }

    ss = warp_reduce_sum(ss);
    if (lane == 0) {
        red[wv][HCMIX_NMIX] = ss;
    }
    __syncthreads();

    float * pt = part + ((int64_t) t*nwg + wg)*HCMIX_PSTRIDE;
    if (tid < HCMIX_NMIX) {
        pt[tid] = rs[tid];
    } else if (tid == HCMIX_NMIX) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            a += red[w][HCMIX_NMIX];
        }
        pt[HCMIX_NMIX] = a;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counters + t, 1) == nwg - 1;
    }
    __syncthreads();
    if (!s_last || wv != 0) {
        return;
    }
    __threadfence();

    // last workgroup, wave 0: lane j < HCMIX_NMIX sums mix j, lane HCMIX_NMIX the squares, in workgroup order (the fence
    // above invalidated this CU's L1, the loads see the other workgroups' partials)
    float tot = 0.0f;
    if (lane <= HCMIX_NMIX) {
        const float * p = part + (int64_t) t*nwg*HCMIX_PSTRIDE + lane;
        for (int g = 0; g < nwg; ++g) {
            tot += p[g*HCMIX_PSTRIDE];
        }
    }
    const float sst = __shfl_sync(0xffffffff, tot, HCMIX_NMIX, WARP_SIZE);
    const float m   = rsqrtf(sst/K + eps)*tot;

    float * out = dst + t*sd1;
    if (lane < DSV4_HC) {
        out[lane] = 1.0f/(1.0f + expf(-(m*scale[0] + base[lane]))) + hc_eps;
    } else if (lane < 2*DSV4_HC) {
        out[lane] = 2.0f/(1.0f + expf(-(m*scale[1] + base[lane])));
    }

    // comb: lane idx = dst + DSV4_HC*src (< 16) holds logit [dst, src]; the 4 lanes of a quad share src, lanes with the
    // same dst are 4 apart within the row of 16
    constexpr int NC = DSV4_HC*DSV4_HC;
    const float mc = __shfl_sync(0xffffffff, m, (lane + 2*DSV4_HC) % WARP_SIZE, WARP_SIZE);
    float c = lane < NC ? mc*scale[2] + base[2*DSV4_HC + lane] : 0.0f;
#ifdef GGML_USE_HIP
    // DPP: quad_perm [1,0,3,2] / [2,3,0,1] (xor 1 / xor 2), row_ror 4 / 8
    auto x1  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false)); };
    auto x2  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false)); };
    auto r4  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x124, 0xF, 0xF, false)); };
    auto r8  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xF, 0xF, false)); };
    auto rcp = [](float v) { return __builtin_amdgcn_rcpf(v); };
#else
    auto x1  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 1, WARP_SIZE); };
    auto x2  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 2, WARP_SIZE); };
    auto r4  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 4, WARP_SIZE); };
    auto r8  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 8, WARP_SIZE); };
    auto rcp = [](float v) { return 1.0f/v; };
#endif
    float mx = fmaxf(c, x1(c));
    mx = fmaxf(mx, x2(mx));
    c = expf(c - mx);
    float s = c + x1(c);
    s += x2(s);
    c = c*rcp(s) + hc_eps;
    auto norm_cols = [&]() {
        float cs = c + r4(c);
        cs += r8(cs);
        c *= rcp(cs + hc_eps);
    };
    auto norm_rows = [&]() {
        float rsum = c + x1(c);
        rsum += x2(rsum);
        c *= rcp(rsum + hc_eps);
    };
    norm_cols();
    for (int i = 1; i < n_iter; ++i) {
        norm_rows();
        norm_cols();
    }
    if (lane < NC) {
        out[2*DSV4_HC + lane] = c;
    }
    if (lane == 0) {
        counters[t] = 0;
    }
}

// Prefill-sized DSV4_HC_MIX (GGML_CUDA_HC_MIX_PREFILL=0 off; from GGML_CUDA_HC_MIX_PREFILL_MIN_NT tokens, default 65):
// the decode kernel's 20 workgroups per token (q2_K decoded per workgroup, partials + arrival counter) lose to the unfused
// norm + q8_1 quantize + MMQ chain at 256 tokens, and that chain costs ~1 ms per sublayer at 1024 tokens (it writes the
// normalized stream, then its q8_1 copy). Here:
//   (1) hc_fn converted to f32 rows (24 x K, 2 MB),
//   (2) a split-K GEMM (dsv4_hc_mix_gemm_f32): workgroup = 64 tokens x a chunk of the flat elements, partial dot
//       products and sums of squares per (chunk, token),
//   (3) one wave per token sums the chunks in order, then the norm, gates and Sinkhorn exactly as the decode kernel.
static constexpr int HCMP_SUB = 256;          // flat elements per LDS weight sub-tile (24 KB)
static constexpr int HCMP_TPL = 2;            // tokens per lane
static constexpr int HCMP_TPW = 8*HCMP_TPL;   // tokens per wave: 8 token lanes x 8 k lanes
static constexpr int HCMP_TPB = 4*HCMP_TPW;   // tokens per workgroup (4 waves)

// grid (K/KR, ceil(nt/HCMP_TPB)), 256 threads: the chunks of a token tile are dispatched together (with the token tile
// in x, the workgroups in flight read the same columns of rows 64 KB apart: 843 -> 650 us at n_embd 4096, 2048 tokens).
// Lane = (token lane tl = lane/8, k lane kl = lane%8): the 8 k lanes of a token read consecutive float4s (128 B
// segments), every lane accumulates all 24 rows for HCMP_TPL tokens, the weight sub-tile is read from LDS with 8 distinct
// consecutive float4s per instruction (no bank conflicts); x loads two steps ahead were slower. The 8 k lanes are
// summed once per workgroup (fixed xor order), chunks in order by the finalize kernel.
static __global__ void __launch_bounds__(256) dsv4_hc_mix_gemm_f32(
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ part,
        const int n_embd, const int K, const int KR, const int nt, const int64_t sx1, const int64_t sx2) {
    __shared__ float4 ws[HCMIX_NMIX][HCMP_SUB/4];

    const int tid   = threadIdx.x;
    const int lane  = tid & 63;
    const int wv    = tid >> 6;
    const int kl    = lane & 7;
    const int tl    = lane >> 3;
    const int chunk = blockIdx.x;
    const int k0    = chunk*KR;
    const int h     = k0 / n_embd; // a chunk lies in one stream (n_embd % KR == 0)
    const int tb    = blockIdx.y*HCMP_TPB + wv*HCMP_TPW + tl*HCMP_TPL;

    const float4 * xr[HCMP_TPL];
#pragma unroll
    for (int q = 0; q < HCMP_TPL; ++q) {
        const int t = min(tb + q, nt - 1); // rows past nt load a valid row, their results are dropped
        xr[q] = (const float4 *) (x + (int64_t) t*sx2 + (int64_t) h*sx1 + (k0 - h*n_embd));
    }

    float acc[HCMP_TPL][HCMIX_NMIX];
    float ss[HCMP_TPL];
#pragma unroll
    for (int q = 0; q < HCMP_TPL; ++q) {
        ss[q] = 0.0f;
#pragma unroll
        for (int r = 0; r < HCMIX_NMIX; ++r) {
            acc[q][r] = 0.0f;
        }
    }

    // x for the first step of each sub-tile is loaded before the barrier, every later step's x one step ahead; the
    // weight rows are read from LDS 4 at a time ahead of their FMAs (one read in flight per row serialized the loop)
    constexpr int NSTEP = HCMP_SUB/32;
    for (int s = 0; s < KR; s += HCMP_SUB) {
        float4 xv[HCMP_TPL];
#pragma unroll
        for (int q = 0; q < HCMP_TPL; ++q) {
            xv[q] = xr[q][s/4 + kl];
        }
        __syncthreads();
#pragma unroll
        for (int i = tid; i < HCMIX_NMIX*(HCMP_SUB/4); i += 256) {
            const int r = i / (HCMP_SUB/4);
            const int c = i % (HCMP_SUB/4);
            ws[r][c] = *(const float4 *) (w + (int64_t) r*K + k0 + s + 4*c);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NSTEP; ++i) {
            const int c = kl + 8*i;
            float4 xn[HCMP_TPL];
            if (i + 1 < NSTEP) {
#pragma unroll
                for (int q = 0; q < HCMP_TPL; ++q) {
                    xn[q] = xr[q][s/4 + c + 8];
                }
            }
#pragma unroll
            for (int rb = 0; rb < HCMIX_NMIX; rb += 4) {
                float4 wr[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    wr[u] = ws[rb + u][c];
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
#pragma unroll
                    for (int q = 0; q < HCMP_TPL; ++q) {
                        float a = acc[q][rb + u];
                        a = fmaf(wr[u].x, xv[q].x, a);
                        a = fmaf(wr[u].y, xv[q].y, a);
                        a = fmaf(wr[u].z, xv[q].z, a);
                        a = fmaf(wr[u].w, xv[q].w, a);
                        acc[q][rb + u] = a;
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < HCMP_TPL; ++q) {
                float a = ss[q];
                a = fmaf(xv[q].x, xv[q].x, a);
                a = fmaf(xv[q].y, xv[q].y, a);
                a = fmaf(xv[q].z, xv[q].z, a);
                a = fmaf(xv[q].w, xv[q].w, a);
                ss[q] = a;
            }
            if (i + 1 < NSTEP) {
#pragma unroll
                for (int q = 0; q < HCMP_TPL; ++q) {
                    xv[q] = xn[q];
                }
            }
        }
    }

    // sum the 8 k lanes of each token (lanes 8*tl .. 8*tl + 7)
#pragma unroll
    for (int q = 0; q < HCMP_TPL; ++q) {
#pragma unroll
        for (int r = 0; r < HCMIX_NMIX; ++r) {
            acc[q][r] = warp_reduce_sum<8>(acc[q][r]);
        }
        ss[q] = warp_reduce_sum<8>(ss[q]);
    }
    if (kl == 0) {
#pragma unroll
        for (int q = 0; q < HCMP_TPL; ++q) {
            const int t = tb + q;
            if (t < nt) {
                float * pt = part + ((int64_t) chunk*nt + t)*HCMIX_PSTRIDE;
#pragma unroll
                for (int r = 0; r < HCMIX_NMIX; ++r) {
                    pt[r] = acc[q][r];
                }
                pt[HCMIX_NMIX] = ss[q];
            }
        }
    }
}

// one warp per token: lane j < 24 sums mix j over the chunks (in order), lane 24 the squares; then as dsv4_hc_mix_f32
static __global__ void __launch_bounds__(256) dsv4_hc_mix_fin_f32(
        const float * __restrict__ part, const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ dst, const int nt, const int nchunk, const int K, const int64_t sd1,
        const float eps, const float hc_eps, const int n_iter) {
    static_assert(WARP_SIZE >= 32 && HCMIX_NMIX < 32, "one (logical) warp per token, lanes 0..24 used");
    const int lane = threadIdx.x % WARP_SIZE;
    const int t    = blockIdx.x*(256/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (t >= nt) {
        return;
    }
    float tot = 0.0f;
    if (lane <= HCMIX_NMIX) {
        const float * p = part + (int64_t) t*HCMIX_PSTRIDE + lane;
        for (int c = 0; c < nchunk; ++c) {
            tot += p[(int64_t) c*nt*HCMIX_PSTRIDE];
        }
    }
    const float sst = __shfl_sync(0xffffffff, tot, HCMIX_NMIX, WARP_SIZE);
    const float m   = rsqrtf(sst/K + eps)*tot;

    float * out = dst + t*sd1;
    if (lane < DSV4_HC) {
        out[lane] = 1.0f/(1.0f + expf(-(m*scale[0] + base[lane]))) + hc_eps;
    } else if (lane < 2*DSV4_HC) {
        out[lane] = 2.0f/(1.0f + expf(-(m*scale[1] + base[lane])));
    }

    constexpr int NC = DSV4_HC*DSV4_HC;
    const float mc = __shfl_sync(0xffffffff, m, (lane + 2*DSV4_HC) % WARP_SIZE, WARP_SIZE);
    float c = lane < NC ? mc*scale[2] + base[2*DSV4_HC + lane] : 0.0f;
    auto x1  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false)); };
    auto x2  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false)); };
    auto r4  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x124, 0xF, 0xF, false)); };
    auto r8  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xF, 0xF, false)); };
    auto rcp = [](float v) { return __builtin_amdgcn_rcpf(v); };
    float mx = fmaxf(c, x1(c));
    mx = fmaxf(mx, x2(mx));
    c = expf(c - mx);
    float sm = c + x1(c);
    sm += x2(sm);
    c = c*rcp(sm) + hc_eps;
    auto norm_cols = [&]() {
        float cs = c + r4(c);
        cs += r8(cs);
        c *= rcp(cs + hc_eps);
    };
    auto norm_rows = [&]() {
        float rsum = c + x1(c);
        rsum += x2(rsum);
        c *= rcp(rsum + hc_eps);
    };
    norm_cols();
    for (int i = 1; i < n_iter; ++i) {
        norm_rows();
        norm_cols();
    }
    if (lane < NC) {
        out[2*DSV4_HC + lane] = c;
    }
}

static bool ggml_cuda_dsv4_hc_mix_prefill_ok(const ggml_tensor * dst) {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_HC_MIX_PREFILL"); return !e || atoi(e) != 0; }();
    static const int  min_nt = [] { const char * e = getenv("GGML_CUDA_HC_MIX_PREFILL_MIN_NT"); return e ? atoi(e) : 65; }();
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * fn = dst->src[1];
    const int64_t n_embd = x->ne[0];
    return env && x->ne[2] >= min_nt && n_embd % HCMP_SUB == 0 && x->nb[1] % 16 == 0 && x->nb[2] % 16 == 0 &&
        ((uintptr_t) x->data) % 16 == 0 && ggml_is_contiguous(fn) && ggml_get_to_fp32_cuda(fn->type) != nullptr;
}

static void ggml_cuda_op_dsv4_hc_mix_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * fn    = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * base  = dst->src[3];

    const int     n_embd = (int) x->ne[0];
    const int     K      = n_embd*DSV4_HC;
    const int     nt     = (int) x->ne[2];
    // chunk size: 512 when that gives >= ~240 workgroups (60 CUs x 4), else 256; a chunk must divide n_embd (it lies in
    // one stream). Longer chunks left too few workgroups for the 2 waves per SIMD this kernel runs (MI50: n_embd 4096,
    // 2048 tokens: chunk 2048 650 us, 1024 480, 512 438, 256 495; n_embd 5120, 1024 tokens: 1024 352, 512 306, 256 326)
    const int ntile = (nt + HCMP_TPB - 1)/HCMP_TPB;
    int KR = HCMP_SUB;
    for (const int cand : {512}) {
        if (n_embd % cand == 0 && (int64_t) ntile*(K/cand) >= 240) {
            KR = cand;
            break;
        }
    }
    const int     nchunk = K/KR;
    cudaStream_t  stream = ctx.stream();

    const float * w = nullptr;
    ggml_cuda_pool_alloc<float> wbuf(ctx.pool());
    if (fn->type == GGML_TYPE_F32) {
        w = (const float *) fn->data;
    } else {
        wbuf.alloc((size_t) HCMIX_NMIX*K);
        ggml_get_to_fp32_cuda(fn->type)(fn->data, wbuf.get(), (int64_t) HCMIX_NMIX*K, stream);
        w = wbuf.get();
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nchunk*nt*HCMIX_PSTRIDE);

    dsv4_hc_mix_gemm_f32<<<dim3(nchunk, ntile), 256, 0, stream>>>((const float *) x->data, w,
        part.get(), n_embd, K, KR, nt, x->nb[1]/sizeof(float), x->nb[2]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());

    const float   eps    = ggml_get_op_params_f32(dst, 0);
    const float   hc_eps = ggml_get_op_params_f32(dst, 1);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 2);
    constexpr int tok_per_block = 256/WARP_SIZE;
    dsv4_hc_mix_fin_f32<<<(nt + tok_per_block - 1)/tok_per_block, 256, 0, stream>>>(part.get(), (const float *) scale->data, (const float *) base->data,
        (float *) dst->data, nt, nchunk, K, dst->nb[1]/sizeof(float), eps, hc_eps, n_iter);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x  = op->src[0];
    const ggml_tensor * fn = op->src[1];
    const int64_t K = x->ne[0]*x->ne[1];
    bool type_ok = false;
    switch (fn->type) {
        case GGML_TYPE_Q2_K:
            type_ok = K % QK_K == 0 && fn->nb[1] % 4 == 0;
            break;
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            type_ok = K % 4 == 0 && fn->nb[1] % 16 == 0;
            break;
        case GGML_TYPE_Q8_0:
            type_ok = K % QK8_0 == 0 && fn->nb[1] % 4 == 0;
            break;
        default:
            break;
    }
    return type_ok && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && x->ne[1] == DSV4_HC &&
        x->nb[0] == sizeof(float) && fn->ne[0] == K && fn->ne[1] == HCMIX_NMIX && fn->nb[0] == ggml_type_size(fn->type) &&
        fn->ne[2] == 1 && fn->ne[3] == 1 && op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) && x->ne[2] <= HCMIX_MAX_TOK && K < INT_MAX &&
        ((uintptr_t) fn->data) % 16 == 0;
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * fn    = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * base  = dst->src[3];
    GGML_ASSERT(ggml_cuda_dsv4_hc_mix_supported(dst));
    if (ggml_cuda_dsv4_hc_mix_prefill_ok(dst)) {
        ggml_cuda_op_dsv4_hc_mix_prefill(ctx, dst);
        return;
    }

    const int     n_embd = (int) x->ne[0];
    const int     K      = n_embd*DSV4_HC;
    const int64_t nt     = x->ne[2];
    const int     nwg    = (K + HCMIX_CHUNK - 1)/HCMIX_CHUNK;
    cudaStream_t  stream = ctx.stream();

    if (ctx.hc_mix_counters == nullptr) {
        // zero once; every launch leaves them zero (the last workgroup of a token resets its counter)
        CUDA_CHECK(cudaMalloc((void **) &ctx.hc_mix_counters, HCMIX_MAX_TOK*sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.hc_mix_counters, 0, HCMIX_MAX_TOK*sizeof(int), stream));
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), nt*nwg*HCMIX_PSTRIDE);

    const float   eps    = ggml_get_op_params_f32(dst, 0);
    const float   hc_eps = ggml_get_op_params_f32(dst, 1);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 2);

    const dim3 grid(nwg, nt);
#define HCMIX_LAUNCH(T) dsv4_hc_mix_f32<T><<<grid, HCMIX_THREADS, 0, stream>>>((const float *) x->data, \
        (const char *) fn->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data, part.get(), \
        ctx.hc_mix_counters, n_embd, K, x->nb[1]/sizeof(float), x->nb[2]/sizeof(float), fn->nb[1], dst->nb[1]/sizeof(float), \
        eps, hc_eps, n_iter)
    switch (fn->type) {
        case GGML_TYPE_Q2_K: HCMIX_LAUNCH(GGML_TYPE_Q2_K); break;
        case GGML_TYPE_F32:  HCMIX_LAUNCH(GGML_TYPE_F32);  break;
        case GGML_TYPE_F16:  HCMIX_LAUNCH(GGML_TYPE_F16);  break;
        case GGML_TYPE_Q8_0: HCMIX_LAUNCH(GGML_TYPE_Q8_0); break;
        default: GGML_ABORT("unsupported hc_fn type");
    }
#undef HCMIX_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// DSV4 HC sublayer transition in one launch (decode): X' = HC_POST(f, X, post, comb) (written out), the next mix
// HC_MIX(X') (as dsv4_hc_mix_f32) and the next sublayer's input RMS_NORM(HC_PRE(X', pre_prev)) * w. Workgroup w owns
// embedding positions 256*w .. + 255 of all hc streams: it computes those X' values, their collapse y (to a scratch row)
// and, with X' in LDS, the mix dot products over super-blocks w, w + nb, w + 2*nb, w + 3*nb of every hc_fn row (nb =
// n_embd/256 super-blocks per stream); the last workgroup of the token (arrival counter) sums the partials in workgroup
// order, finishes the mix and normalizes y.
static constexpr int HCSTEP_PSTRIDE = 32;
// the tokens' arrival counters 256 B apart: adjacent counters put a 6-token verify's 96 workgroup arrivals on one cache
// line, where the L2 serializes the atomics
static constexpr int HCSTEP_CSTRIDE = 64;

// HCSTEP_TRACE 1: per-workgroup s_memrealtime stamps of the step's phases (GGML_CUDA_HC_STEP_TRACE=N prints N calls after
// GGML_CUDA_HC_STEP_TRACE_SKIP; needs GGML_CUDA_DISABLE_GRAPHS=1: the host synchronizes after each traced call), and
// GGML_CUDA_HC_STEP_NOP=1 (an empty kernel instead of the step: its in-graph cost). s_memrealtime counts at 25 MHz on
// gfx906 (hipDeviceAttributeWallClockRate). GLM-5.3 decode, 2026-10-04: ~15 us per step (phase 1 ~6 us: loads 2.3, dots
// and partials 3.3; then the last workgroup alone: merge 2.8 (it waits for its X' reloads), collapse 3.0, mix finish 3.2)
// and 1.47 ms of the 15.0 ms token in the graph (89 steps). The last workgroup's partial loads go out first and wave 0
// finishes the mix before the collapse (the other waves' share of it overlaps): GLM tg 67.4 -> 68.0, DS4 52.2 -> 52.8.
// Issuing the X' reloads only after the merge's barrier moved time from the merge to the mix finish (no gain)
#define HCSTEP_TRACE 0
#if HCSTEP_TRACE
static constexpr double HCS_US = 0.04; // us per s_memrealtime tick
static __device__ unsigned long long hcs_trace[1024][8];
#define HCS_T(ph) do { if (threadIdx.x == 0) { hcs_trace[blockIdx.x + gridDim.x*blockIdx.y][ph] = __builtin_amdgcn_s_memrealtime(); } } while (0)
#else
#define HCS_T(ph)
#endif

// sum over the 64 lanes of a wave by DPP (row sums by quad_perm / row mirrors, then row_bcast:15 / :31): the total lands
// in lane 63. LDS-free, so independent sums interleave (the shfl ladder is a chain of LDS round trips per step).
static __device__ __forceinline__ float hcstep_wave_sum63(float v) {
#ifdef GGML_USE_HIP
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1,  0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E,  0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x140, 0xF, 0xF, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x142, 0xA, 0xF, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x143, 0xC, 0xF, false));
    return v;
#else
    return warp_reduce_sum<WARP_SIZE>(v);
#endif
}

// GB (grid barrier): the token's workgroups (all resident: they are dispatched together) wait for each other after the
// arrival count (generation counter gen[t], bumped by the last arriver) and each normalizes and writes its own 256
// positions of the collapse (y stays in registers); the last arriver alone finishes the mix. Without GB the last
// workgroup reads y back (ybuf) and writes the whole collapse.
// Q8: q8_0 hc_fn (GLM-5-Next) instead of q2_K: lane (stream b, q) reads half q%2 of q8_0 block 8*wg + q/2 of its stream
// (its 16 quants from 5 aligned dwords: they start 0 or 2 bytes into a dword, alignbyte; the f16 scale from the dword
// holding it) for each of its wave's 6 rows, and dots them with X' positions 16q .. 16q + 15 of stream b
// CUR (GLM-5-Next, no GB): the collapse's pre gate is THIS mix's (HC_PRE takes a view of the mix being computed), so
// phase 1 does no collapse; the last workgroup reads X' back (every stream), takes the pre gate from the merged sums
// (the expression of the mix output) and writes the normalized collapse
// F16W: f16 hc_fn (DeepSeek V4 Flash): lane (stream b, q) reads the 16 halves of its positions 16q .. 16q + 15 of stream b
// (two 16-byte loads) for each of its wave's 6 rows; f32 products
template <int MAXP, bool GB, bool Q8 = false, bool CUR = false, bool F16W = false> // MAXP: n_embd/256 rounded up to a supported count (= the number of workgroups)
static __global__ void __launch_bounds__(256) dsv4_hc_step_f32(
        const float * __restrict__ f, const float * __restrict__ X, const float * __restrict__ post,
        const float * __restrict__ comb, const float * __restrict__ pre_prev, const char * __restrict__ fn,
        const float * __restrict__ scale, const float * __restrict__ base, const float * __restrict__ norm_w,
        float * __restrict__ Xn, float * __restrict__ mix, float * __restrict__ yn,
        float * __restrict__ ybuf, float * __restrict__ part, int * __restrict__ counters, int * __restrict__ gen,
        const int n_embd, const int64_t sf1, const int64_t sX1, const int64_t sX2, const int64_t sp0, const int64_t sp1,
        const int64_t sc0, const int64_t sc1, const int64_t sc2, const int64_t spp0, const int64_t spp1, const int64_t s_fn,
        const int64_t sXn1, const int64_t sXn2, const int64_t smix1, const int64_t syn1,
        const float eps_mix, const float hc_eps, const int n_iter, const float eps_norm) {
    constexpr int PW = ggml_cuda_get_physical_warp_size();
    const int t   = blockIdx.y;
    const int wg  = blockIdx.x;
    const int nwg = gridDim.x;
    const int tid = threadIdx.x;
    const int i   = 256*wg + tid;          // embedding position
    const int nb  = n_embd/QK_K;           // super-blocks per stream
    const int K   = DSV4_HC*n_embd;
    HCS_T(0);

    __shared__ float xs[DSV4_HC][256];
    __shared__ float rs[HCMIX_NMIX];
    __shared__ float red[256/PW][2];
    __shared__ int   s_last;
    __shared__ int   s_gen0;
    if (GB && tid == 0) {
        s_gen0 = __hip_atomic_load(gen + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); // before this workgroup arrives
    }
    const float wn1 = GB ? norm_w[i] : 0.0f;

    // the mix finish's constants (needed only by the last workgroup, which every workgroup may be): issued first so they
    // are not a round trip after the merge
    const float bA  = base[min(tid, HCMIX_NMIX - 1)];                // pre/post: base[lane]
    const float bB  = base[min(2*DSV4_HC + tid, HCMIX_NMIX - 1)];   // comb: base[8 + lane]
    const float ms0 = scale[0], ms1 = scale[1], ms2 = scale[2];

    // X' = post*f + comb*X, y = pre_prev . X'
    float pw[DSV4_HC], pp[DSV4_HC], cm[DSV4_HC][DSV4_HC], xv[DSV4_HC];
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        pw[h] = post[h*sp0 + t*sp1];
        pp[h] = CUR ? 0.0f : pre_prev[h*spp0 + t*spp1];
#pragma unroll
        for (int src = 0; src < DSV4_HC; ++src) {
            cm[h][src] = comb[h*sc0 + src*sc1 + t*sc2];
        }
        xv[h] = X[i + h*sX1 + t*sX2];
    }
    const float fv = f[i + t*sf1];

    // then the hc_fn weight words (thread = row group rg, stream/block slot b, qs dword q; rows rg + 4*rr): vector loads
    // return in order, so the X' math above waits only for X and f
    const int rg = tid >> 6;
    const int b  = (tid >> 4) & 3;
    const int q  = tid & 15;
    const int n  = q >> 3;
    const int l0 = 4*(q & 7);
    const int hh = l0 >> 4;
    constexpr int RPG = HCMIX_NMIX/4;
    constexpr int R2  = Q8 ? 1 : RPG;
    constexpr int R8  = Q8 ? RPG : 1;
    uint32_t wq[R2], ws0[R2], ws1[R2], wdm[R2];
    uint32_t w8[R8][5], wsc[R8];
    uint4 wf[F16W ? RPG : 1][2];
    int sh8 = 0, ssh8 = 0;
    if constexpr (F16W) {
        const int64_t eo = ((int64_t) b*n_embd + 256*wg + 16*q)*(int64_t) sizeof(half);
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const uint4 * pw16 = (const uint4 *) (fn + (rg + 4*rr)*s_fn + eo);
            wf[rr][0] = pw16[0];
            wf[rr][1] = pw16[1];
        }
    } else if constexpr (Q8) {
        const int blk = b*(n_embd/QK8_0) + 8*wg + (q >> 1);
        const int so  = blk*(int) sizeof(block_q8_0);     // the block (scale) offset in the row
        const int off = so + 2 + 16*(q & 1);              // its 16 quants
        sh8  = off & 3;
        ssh8 = so & 3;
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const char * row = fn + (rg + 4*rr)*s_fn;
            const uint32_t * pw8 = (const uint32_t *) (row + (off - sh8));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                w8[rr][k] = pw8[k];
            }
            w8[rr][4] = pw8[sh8 ? 4 : 3];
            wsc[rr]   = *(const uint32_t *) (row + (so - ssh8));
        }
    } else {
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const char * blk = fn + (rg + 4*rr)*s_fn + (int64_t) (b*nb + wg)*sizeof(block_q2_K);
            wq[rr]  = *(const uint32_t *) (blk + 16 + 4*q);
            ws0[rr] = *(const uint32_t *) (blk + 8*n);
            ws1[rr] = *(const uint32_t *) (blk + 8*n + 4);
            wdm[rr] = *(const uint32_t *) (blk + 80);
        }
    }

#ifdef GGML_USE_HIP
    // every load above (X, f, the gates, the hc_fn words) goes out before the first use: left alone, the scheduler
    // sinks the gate loads to their stream (one serialized scalar round trip per stream) and the weight loads behind them
    __builtin_amdgcn_sched_barrier(0);
#endif
    float y = 0.0f, ssx = 0.0f;
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        float v = fv*pw[h];
#pragma unroll
        for (int src = 0; src < DSV4_HC; ++src) {
            v += xv[src]*cm[h][src];
        }
        Xn[i + h*sXn1 + t*sXn2] = v;
        xs[h][tid] = v;
        if constexpr (!CUR) {
            y += v*pp[h];
        }
        ssx += v*v;
    }
    if (!GB && !CUR) {
        ybuf[(int64_t) t*n_embd + i] = y;
    }
    float ssy = y*y;
    __syncthreads();
    HCS_T(1);

    // mix dot products over this workgroup's 4 super-blocks (one per stream) of every row
    float acc[RPG];
    if constexpr (F16W) {
        float x16[16];
        const float * xb16 = xs[b] + 16*q;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (xb16 + 4*j);
            x16[4*j + 0] = v.x; x16[4*j + 1] = v.y; x16[4*j + 2] = v.z; x16[4*j + 3] = v.w;
        }
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const uint32_t wv[8] = { wf[rr][0].x, wf[rr][0].y, wf[rr][0].z, wf[rr][0].w,
                                     wf[rr][1].x, wf[rr][1].y, wf[rr][1].z, wf[rr][1].w };
            float sq = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float2 wk = __half22float2(*(const half2 *) &wv[k]);
                sq += wk.x*x16[2*k + 0] + wk.y*x16[2*k + 1];
            }
            acc[rr] = sq;
        }
    } else if constexpr (Q8) {
        float x8[16];
        const float * xb8 = xs[b] + 16*q;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (xb8 + 4*j);
            x8[4*j + 0] = v.x; x8[4*j + 1] = v.y; x8[4*j + 2] = v.z; x8[4*j + 3] = v.w;
        }
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const float d = __half2float(__ushort_as_half((unsigned short) (ssh8 ? wsc[rr] >> 16 : wsc[rr] & 0xFFFF)));
            float sq = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qd = __builtin_amdgcn_alignbyte(w8[rr][k + 1], w8[rr][k], sh8);
                sq += (float) ((int) (qd << 24) >> 24)*x8[4*k + 0] + (float) ((int) (qd << 16) >> 24)*x8[4*k + 1] +
                      (float) ((int) (qd <<  8) >> 24)*x8[4*k + 2] + (float) ((int) qd >> 24)*x8[4*k + 3];
            }
            acc[rr] = d*sq;
        }
    } else {
        float xw[16], sx[4];
        const float * xb = xs[b] + 128*n + l0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float4 v = *(const float4 *) (xb + 32*j);
            xw[4*j + 0] = v.x; xw[4*j + 1] = v.y; xw[4*j + 2] = v.z; xw[4*j + 3] = v.w;
            sx[j] = (v.x + v.y) + (v.z + v.w);
        }
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            const uint32_t v = wq[rr];
            const float d    = __half2float(__ushort_as_half((unsigned short) (wdm[rr] & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) (wdm[rr] >> 16)));
            float sq = 0.0f, sm = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t scw = j < 2 ? ws0[rr] : ws1[rr];
                const uint32_t sb  = (scw >> (8*(2*(j & 1) + hh))) & 0xFF;
                const uint32_t qq  = (v >> (2*j)) & 0x03030303u;
                const float dot = (float) (qq & 0xFF)*xw[4*j + 0] + (float) ((qq >> 8) & 0xFF)*xw[4*j + 1] +
                                  (float) ((qq >> 16) & 0xFF)*xw[4*j + 2] + (float) (qq >> 24)*xw[4*j + 3];
                sq += (float) (sb & 0xF)*dot;
                sm += (float) (sb >> 4)*sx[j];
            }
            acc[rr] = d*sq - dmin*sm;
        }
    }
    // wave sums (DPP, in lane PW - 1; PW == 64 on the HIP path)
#pragma unroll
    for (int rr = 0; rr < RPG; ++rr) {
        acc[rr] = hcstep_wave_sum63(acc[rr]);
    }
    ssx = hcstep_wave_sum63(ssx);
    ssy = CUR ? 0.0f : hcstep_wave_sum63(ssy);
    __shared__ float wsum[256/PW][RPG];
    if (tid % PW == PW - 1) {
#pragma unroll
        for (int rr = 0; rr < RPG; ++rr) {
            wsum[tid/PW][rr] = acc[rr];
        }
        red[tid/PW][0] = ssx;
        red[tid/PW][1] = ssy;
    }
    __syncthreads();
    float * pt = part + ((int64_t) t*nwg + wg)*HCSTEP_PSTRIDE;
    if (tid < HCMIX_NMIX) {
        const int rgr = tid % 4, rr = tid / 4;
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < 64/PW; ++w) {
            a += wsum[rgr*(64/PW) + w][rr];
        }
        pt[tid] = a;
    } else if (tid < HCMIX_NMIX + 2) {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < 256/PW; ++w) {
            a += red[w][tid - HCMIX_NMIX];
        }
        pt[tid] = a;
    }
    __threadfence();
    __syncthreads();
    HCS_T(2);
    if (GB) {
        if (tid == 0) {
            s_last = atomicAdd(counters + HCSTEP_CSTRIDE*t, 1) == nwg - 1;
            if (s_last) {
                counters[HCSTEP_CSTRIDE*t] = 0;
                __threadfence();
                atomicAdd(gen + t, 1);  // release the others
            } else {
                while (__hip_atomic_load(gen + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == s_gen0) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
        }
        __syncthreads();
        __threadfence();
        // every workgroup: the collapse norm from the partial sums of squares (workgroup order: the same value everywhere)
        const float * pq = part + (int64_t) t*nwg*HCSTEP_PSTRIDE + HCMIX_NMIX + 1;
        float sq[MAXP];
#pragma unroll
        for (int g = 0; g < MAXP; ++g) {
            sq[g] = pq[min(g, nwg - 1)*HCSTEP_PSTRIDE];
        }
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < MAXP; ++g) {
            a += g < nwg ? sq[g] : 0.0f;
        }
        const float ry = rsqrtf(a/n_embd + eps_norm);
        yn[t*syn1 + i] = y*ry*wn1;
        if (!s_last) {
            return;
        }
    } else {
        if (tid == 0) {
            s_last = atomicAdd(counters + HCSTEP_CSTRIDE*t, 1) == nwg - 1;
        }
        __syncthreads();
        HCS_T(3);
        if (!s_last) {
            return;
        }
        __threadfence();
    }

    // last workgroup: the sums in workgroup order (lanes of wave 0), the mix, then (without GB) the normalized collapse.
    // The partial sums are loaded first (vector loads return in order: the merge does not wait for the 64 KB of X' the
    // CUR collapse reloads), then every load of the normalization pass (one L2 round trip each)
    const float * yt = ybuf + (int64_t) t*n_embd;
    // all loads straight-line from clamped indices (a uniform `g < nwg ? load : 0` compiles to a branch and a wait per
    // load), unused values dropped afterwards
    constexpr int NQ = (MAXP + 3)/4;         // float4 per thread
    constexpr int NH = CUR ? DSV4_HC : 1;    // CUR: X' of every stream instead of the collapse
    const int n4  = n_embd/4;
    float pv[MAXP];
    if (tid <= HCMIX_NMIX + 1) {
        const float * p = part + (int64_t) t*nwg*HCSTEP_PSTRIDE + tid;
#pragma unroll
        for (int g = 0; g < MAXP; ++g) {
            pv[g] = p[min(g, nwg - 1)*HCSTEP_PSTRIDE];
        }
    }
    float4 yv[NQ], wn[NQ], xq[NH][NQ];
    float bpre[DSV4_HC];
    if (!GB) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const int k4 = min(q*256 + tid, n4 - 1);
            if constexpr (CUR) {
#pragma unroll
                for (int h = 0; h < DSV4_HC; ++h) {
                    xq[h][q] = ((const float4 *) (Xn + h*sXn1 + t*sXn2))[k4];
                }
            } else {
                yv[q] = ((const float4 *) yt)[k4];
            }
            wn[q] = ((const float4 *) norm_w)[k4];
        }
    }
    if constexpr (CUR) {
#pragma unroll
        for (int h = 0; h < DSV4_HC; ++h) {
            bpre[h] = base[h];
        }
    }
    __shared__ float fin[HCMIX_NMIX + 2];
    if (tid <= HCMIX_NMIX + 1) {
        float a = 0.0f;
#pragma unroll
        for (int g = 0; g < MAXP; ++g) {
            a += g < nwg ? pv[g] : 0.0f;
        }
        fin[tid] = a;
    }
    __syncthreads();
    HCS_T(4);
    // the mix output (wave 0; with CUR before the collapse: the Sinkhorn chain is the longest stretch of the last
    // workgroup, and the other waves compute their part of the collapse meanwhile)
    const auto mix_finish = [&]() {
        const int lane = tid;
        const float r  = rsqrtf(fin[HCMIX_NMIX]/K + eps_mix);
        const float m  = lane < HCMIX_NMIX ? r*fin[lane] : 0.0f;
        float * out = mix + t*smix1;
        if (lane < DSV4_HC) {
            out[lane] = 1.0f/(1.0f + expf(-(m*ms0 + bA))) + hc_eps;
        } else if (lane < 2*DSV4_HC) {
            out[lane] = 2.0f/(1.0f + expf(-(m*ms1 + bA)));
        }
        constexpr int NC = DSV4_HC*DSV4_HC;
        float c = lane < NC ? r*fin[2*DSV4_HC + lane]*ms2 + bB : 0.0f;
#ifdef GGML_USE_HIP
        auto x1  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false)); };
        auto x2  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false)); };
        auto r4  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x124, 0xF, 0xF, false)); };
        auto r8  = [](float v) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xF, 0xF, false)); };
        auto rcp = [](float v) { return __builtin_amdgcn_rcpf(v); };
#else
        auto x1  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 1, WARP_SIZE); };
        auto x2  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 2, WARP_SIZE); };
        auto r4  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 4, WARP_SIZE); };
        auto r8  = [](float v) { return __shfl_xor_sync(0xffffffff, v, 8, WARP_SIZE); };
        auto rcp = [](float v) { return 1.0f/v; };
#endif
        float mx = fmaxf(c, x1(c));
        mx = fmaxf(mx, x2(mx));
        c = expf(c - mx);
        float sm = c + x1(c);
        sm += x2(sm);
        c = c*rcp(sm) + hc_eps;
        auto norm_cols = [&]() { float cs = c + r4(c); cs += r8(cs); c *= rcp(cs + hc_eps); };
        auto norm_rows = [&]() { float rsum = c + x1(c); rsum += x2(rsum); c *= rcp(rsum + hc_eps); };
        norm_cols();
        for (int it = 1; it < n_iter; ++it) {
            norm_rows();
            norm_cols();
        }
        if (lane < NC) {
            out[2*DSV4_HC + lane] = c;
        }
        if (!GB && lane == 0) {
            counters[HCSTEP_CSTRIDE*t] = 0;
        }
        HCS_T(6);
    };
    if (CUR && tid < WARP_SIZE) {
        mix_finish();
    }
    if constexpr (CUR) {
        // every thread: the pre gate (the expression of the mix output below, so the same values), the collapse and its
        // sum of squares (wave sums in wave order), then the normalized collapse
        const float r = rsqrtf(fin[HCMIX_NMIX]/K + eps_mix);
        float pg[DSV4_HC];
#pragma unroll
        for (int h = 0; h < DSV4_HC; ++h) {
            const float m = r*fin[h];
            pg[h] = 1.0f/(1.0f + expf(-(m*ms0 + bpre[h]))) + hc_eps;
        }
        float ss = 0.0f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int h = 0; h < DSV4_HC; ++h) {
                a.x += xq[h][q].x*pg[h];
                a.y += xq[h][q].y*pg[h];
                a.z += xq[h][q].z*pg[h];
                a.w += xq[h][q].w*pg[h];
            }
            yv[q] = a;
            ss += q*256 + tid < n4 ? (a.x*a.x + a.y*a.y) + (a.z*a.z + a.w*a.w) : 0.0f;
        }
        ss = hcstep_wave_sum63(ss);
        __shared__ float red_y[256/PW];
        if (tid % PW == PW - 1) {
            red_y[tid/PW] = ss;
        }
        __syncthreads();
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < 256/PW; ++w) {
            a += red_y[w];
        }
        const float ry = rsqrtf(a/n_embd + eps_norm);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const int k4 = q*256 + tid;
            if (k4 < n4) {
                const float4 v = yv[q], w = wn[q];
                ((float4 *) (yn + t*syn1))[k4] = make_float4(v.x*ry*w.x, v.y*ry*w.y, v.z*ry*w.z, v.w*ry*w.w);
            }
        }
        HCS_T(5);
    }
    if (!CUR && tid < WARP_SIZE) {
        mix_finish();
    }
    if (GB || CUR) {
        return;
    }
    const float ry = rsqrtf(fin[HCMIX_NMIX + 1]/n_embd + eps_norm);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const int k4 = q*256 + tid;
        if (k4 < n4) {
            const float4 a = yv[q], w = wn[q];
            ((float4 *) (yn + t*syn1))[k4] = make_float4(a.x*ry*w.x, a.y*ry*w.y, a.z*ry*w.z, a.w*ry*w.w);
        }
    }
}

// GGML_CUDA_HC_STEP_F16=0: f16 hc_fn (DeepSeek V4 Flash) keeps the separate HC_POST / HC_MIX / HC_PRE / RMS_NORM kernels
static bool hcstep_f16_env() {
    static const bool env = [] { const char * e = getenv("GGML_CUDA_HC_STEP_F16"); return !e || atoi(e) != 0; }();
    return env;
}

bool ggml_cuda_dsv4_hc_step_supported(const ggml_tensor * post, const ggml_tensor * mixn, const ggml_tensor * pre,
                                      const ggml_tensor * rn, const ggml_tensor * mul, bool * yn_temp) {
    if (yn_temp) {
        *yn_temp = false;
    }
    // the pre gate: from an earlier mix (DeepSeek V4.1 mixes one sublayer ahead, phase 1 collapses with it), or
    // (GLM-5-Next) the pre slice of THIS mix (a direct view at offset 0), finished only by the last workgroup, which then
    // reads X' back for the collapse (CUR: n_embd <= 4096, one float4 load per stream per 256 positions in flight)
    const ggml_tensor * pg = pre->src[1];
    bool cur = false;
    for (const ggml_tensor * t = pg; t != nullptr; t = t->view_src) {
        cur = cur || t == mixn;
    }
    if (cur && !(pg->view_src == mixn && pg->view_offs == 0 && pg->nb[0] == sizeof(float) && pg->nb[1] == mixn->nb[1] &&
                 pg->ne[1] == post->src[0]->ne[1] && post->src[0]->ne[0] <= 4096 && ((uintptr_t) post->data) % 16 == 0)) {
        return false;
    }
    const ggml_tensor * f  = post->src[0];
    const ggml_tensor * X  = post->src[1];
    const ggml_tensor * fn = mixn->src[1];
    const ggml_tensor * w  = mul->src[1];
    const int64_t n_embd = f->ne[0];
    const int64_t nt     = f->ne[1];
    // outputs over inputs: the allocator reuses buffers that die inside the fused group (f, the gates' mix, X). With one
    // token every read comes before the writes that could reach it (phase 1 reads its positions before writing them,
    // the last workgroup writes after every workgroup arrived); with more tokens a block of one token can overwrite
    // another token's input before that is read (wikitext ub32: PPL 2619). X' may lie exactly in place of X (or of f
    // with one token); everything else must not overlap
    {
        auto ovl = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
                return false;
            }
            const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        const ggml_tensor * ins[] = { f, X, post->src[2], post->src[3], cur ? nullptr : pre->src[1], fn, mixn->src[2],
                                      mixn->src[3], w };
        static const char * in_names[] = { "f", "X", "post", "comb", "pre_prev", "fn", "scale", "base", "w" };
        static const bool dbg = getenv("GGML_CUDA_HC_STEP_DBG") != nullptr;
        auto rej = [&](const char * o, int k) {
            if (dbg) {
                fprintf(stderr, "hc_step reject nt %d: %s over %s (%+lld bytes)\n", (int) nt, o, in_names[k],
                        (long long) ((const char *) (o[0] == 'X' ? post : o[0] == 'm' ? mixn : mul)->data - (const char *) ins[k]->data));
            }
            return false;
        };
        // mix and yn are written per token by that token's last workgroup, after all of the token's readers arrived:
        // exactly row on row over an input (the allocator's usual choice: the new mix over the previous one, yn over f)
        // touches only rows already read
        auto row_on_row = [](const ggml_tensor * o, const ggml_tensor * in) {
            const ggml_tensor * base = in->view_src ? in->view_src : in;
            return o->data == base->data && o->nb[1] == base->nb[1] && o->ne[0]*ggml_type_size(o->type) <= base->nb[1];
        };
        for (int k = 0; k < 9; ++k) {
            const ggml_tensor * in = ins[k];
            const bool in_place = (in == X && post->data == X->data && ggml_are_same_stride(post, X)) ||
                                  (in == f && nt == 1 && post->data == f->data);
            if (!in_place && ovl(post, in)) {
                return rej("Xn", k);
            }
            if (nt > 1 && ovl(mixn, in) && !(k != 1 && row_on_row(mixn, in))) {
                // mix over an input (f): through a temporary as yn (GLM-5.3 2-token verify: ~5 of 90 steps per pass)
                if (!yn_temp) {
                    return rej("mix", k);
                }
                *yn_temp = true;
            }
            if (nt > 1 && ovl(mul, in) && !(k != 1 && row_on_row(mul, in))) {
                // yn over an input (the allocator puts yn exactly over X, which dies in the group: 6-token DSpark verify):
                // the kernel writes yn to a temporary, copied over afterwards (see ggml_cuda_dsv4_hc_step)
                if (!yn_temp) {
                    return rej("yn", k);
                }
                *yn_temp = true;
            }
        }
        if (ovl(mixn, post) || ovl(mul, post) || ovl(mul, mixn)) {
            return false;
        }
    }
    auto f32 = [](const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32; };
    return post->src[3] != nullptr && f32(f) && f32(X) && f32(post->src[2]) && f32(post->src[3]) && f32(post) &&
        f32(mixn) && f32(mixn->src[2]) && f32(mixn->src[3]) && f32(pre->src[1]) && f32(pre) && f32(rn) && f32(mul) && f32(w) &&
        (fn->type == GGML_TYPE_Q2_K || fn->type == GGML_TYPE_Q8_0 ||
         (fn->type == GGML_TYPE_F16 && cur && hcstep_f16_env() && fn->nb[1] % 16 == 0 && ((uintptr_t) fn->data) % 16 == 0)) &&
        fn->ne[0] == DSV4_HC*n_embd && fn->ne[1] == HCMIX_NMIX &&
        fn->nb[1] % 4 == 0 && fn->nb[0] == ggml_type_size(fn->type) &&
        ((uintptr_t) fn->data) % 4 == 0 && X->ne[1] == DSV4_HC && n_embd % QK_K == 0 && n_embd % 256 == 0 &&
        nt >= 1 && nt <= 64 && n_embd <= 8192 && f->nb[0] == sizeof(float) && X->nb[0] == sizeof(float) && post->nb[0] == sizeof(float) &&
        ggml_is_contiguous(post) && ggml_is_contiguous(mixn) && ggml_is_contiguous(mul) && ggml_is_contiguous(w) &&
        ggml_nelements(w) == n_embd && ((uintptr_t) w->data) % 16 == 0 && ((uintptr_t) mul->data) % 16 == 0 &&
        mul->src[0] == rn && rn->src[0] == pre && pre->src[0] == post && mixn->src[0] == post &&
        ggml_get_op_params_i32(pre, 1) == 0 && ggml_get_op_params_f32(pre, 0) == 1.0f &&
        ggml_is_contiguous(mixn->src[2]) && ggml_is_contiguous(mixn->src[3]) && pre->src[1]->ne[0] == DSV4_HC;
}

#if HCSTEP_TRACE
static __global__ void hcstep_nop_kernel() {}
#endif

void ggml_cuda_dsv4_hc_step(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * mixn, const ggml_tensor * pre,
                            const ggml_tensor * rn, ggml_tensor * mul, bool yn_temp) {
#if HCSTEP_TRACE
    // GGML_CUDA_HC_STEP_NOP=1 (measurement only, results garbage): an empty kernel instead of the step
    static const bool nop = [] { const char * e = getenv("GGML_CUDA_HC_STEP_NOP"); return e && atoi(e) != 0; }();
    if (nop) {
        hcstep_nop_kernel<<<1, 64, 0, ctx.stream()>>>();
        return;
    }
#endif
    const ggml_tensor * f    = post->src[0];
    const ggml_tensor * X    = post->src[1];
    const ggml_tensor * pw   = post->src[2];
    const ggml_tensor * cb   = post->src[3];
    const ggml_tensor * fn   = mixn->src[1];
    const ggml_tensor * ppre = pre->src[1];
    const ggml_tensor * w    = mul->src[1];
    const int     n_embd = (int) f->ne[0];
    const int64_t nt     = f->ne[1];
    const int     nwg    = n_embd/256;
    cudaStream_t  stream = ctx.stream();

    if (ctx.hc_mix_counters == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &ctx.hc_mix_counters, HCMIX_MAX_TOK*sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.hc_mix_counters, 0, HCMIX_MAX_TOK*sizeof(int), stream));
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool(), nt*nwg*HCSTEP_PSTRIDE);
    ggml_cuda_pool_alloc<float> ybuf(ctx.pool(), nt*n_embd);
    // yn_temp: yn or the mix overlaps an input of the group (see supported): both are written to temporaries (contiguous
    // rows) and copied to their tensors at the end
    ggml_cuda_pool_alloc<float> yt(ctx.pool());
    ggml_cuda_pool_alloc<float> mt(ctx.pool());
    float * yn_out = (float *) mul->data;
    int64_t yn_s1  = mul->nb[1]/sizeof(float);
    float * mx_out = (float *) mixn->data;
    int64_t mx_s1  = mixn->nb[1]/sizeof(float);
    if (yn_temp) {
        yn_out = yt.alloc(nt*n_embd);
        yn_s1  = n_embd;
        mx_out = mt.alloc(nt*HCMIX_NMIX);
        mx_s1  = HCMIX_NMIX;
    }
    const bool cur_pre = ppre->view_src == mixn;
    const float * ppre_ptr = yn_temp && cur_pre ? (const float *) mx_out : (const float *) ppre->data;
    const int64_t ppre_s0  = ppre->nb[0]/sizeof(float);
    const int64_t ppre_s1  = yn_temp && cur_pre ? mx_s1 : (int64_t) (ppre->nb[1]/sizeof(float));
    // grid barrier + distributed collapse for small batches (opt-in GGML_CUDA_HC_STEP_GB=1: tg 92.7 vs 93.0 without; the
    // last workgroup's mix finish, not the collapse, is the tail, and the barrier adds a release/poll round trip)
    static const bool gb_env = [] { const char * e = getenv("GGML_CUDA_HC_STEP_GB"); return e && atoi(e) != 0; }();
    const bool cur = ppre->view_src == mixn; // GLM-5-Next: the pre gate is this mix's (see supported)
    const bool gb  = gb_env && nt <= 8 && !cur;
    if (gb && ctx.hc_step_gen == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &ctx.hc_step_gen, HCMIX_MAX_TOK*sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.hc_step_gen, 0, HCMIX_MAX_TOK*sizeof(int), stream));
    }

    const bool q8 = fn->type == GGML_TYPE_Q8_0;
    const bool f16w = fn->type == GGML_TYPE_F16;
#define HCSTEP_LAUNCH(MAXP) do { if (q8) { if (gb) { HCSTEP_LAUNCH_GB(MAXP, true, true, false, false); } else { HCSTEP_LAUNCH_GB(MAXP, false, true, false, false); } } \
                                 else    { if (gb) { HCSTEP_LAUNCH_GB(MAXP, true, false, false, false); } else { HCSTEP_LAUNCH_GB(MAXP, false, false, false, false); } } } while (0)
#define HCSTEP_LAUNCH_GB(MAXP, GB, Q8, CUR, F16W) dsv4_hc_step_f32<MAXP, GB, Q8, CUR, F16W><<<dim3(nwg, nt), 256, 0, stream>>>( \
        (const float *) f->data, (const float *) X->data, (const float *) pw->data, (const float *) cb->data, \
        (const float *) ppre_ptr, (const char *) fn->data, (const float *) mixn->src[2]->data, \
        (const float *) mixn->src[3]->data, (const float *) w->data, \
        (float *) post->data, mx_out, yn_out, ybuf.get(), part.get(), ctx.hc_mix_counters, ctx.hc_step_gen, \
        n_embd, f->nb[1]/sizeof(float), X->nb[1]/sizeof(float), X->nb[2]/sizeof(float), \
        pw->nb[0]/sizeof(float), pw->nb[1]/sizeof(float), \
        cb->nb[0]/sizeof(float), cb->nb[1]/sizeof(float), cb->nb[2]/sizeof(float), \
        ppre_s0, ppre_s1, fn->nb[1], \
        post->nb[1]/sizeof(float), post->nb[2]/sizeof(float), mx_s1, yn_s1, \
        ggml_get_op_params_f32(mixn, 0), ggml_get_op_params_f32(mixn, 1), ggml_get_op_params_i32(mixn, 2), \
        ggml_get_op_params_f32(rn, 0))
    if (cur) {
        GGML_ASSERT(nwg <= 16);
        if (f16w) {
            HCSTEP_LAUNCH_GB(16, false, false, true, true);
        } else if (q8) {
            HCSTEP_LAUNCH_GB(16, false, true, true, false);
        } else {
            HCSTEP_LAUNCH_GB(16, false, false, true, false);
        }
    } else if (nwg <= 16) {
        HCSTEP_LAUNCH(16);
    } else if (nwg <= 20) {
        HCSTEP_LAUNCH(20);
    } else if (nwg <= 28) {
        HCSTEP_LAUNCH(28);
    } else {
        HCSTEP_LAUNCH(32);
    }
#undef HCSTEP_LAUNCH
#undef HCSTEP_LAUNCH_GB
    CUDA_CHECK(cudaGetLastError());
    if (yn_temp) {
        CUDA_CHECK(cudaMemcpy2DAsync(mixn->data, mixn->nb[1], mt.get(), HCMIX_NMIX*sizeof(float), HCMIX_NMIX*sizeof(float), nt,
                                     cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(mul->data, mul->nb[1], yt.get(), n_embd*sizeof(float), n_embd*sizeof(float), nt,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#if HCSTEP_TRACE
    static int64_t tr_seen = 0;
    static const int64_t tr_skip = [] { const char * e = getenv("GGML_CUDA_HC_STEP_TRACE_SKIP"); return e ? atoll(e) : 0; }();
    static const int64_t tr_want = [] { const char * e = getenv("GGML_CUDA_HC_STEP_TRACE"); return e ? atoll(e) : 0; }();
    if (tr_want > 0 && ctx.device == 0 && tr_seen++ >= tr_skip && tr_seen <= tr_skip + tr_want) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        static unsigned long long h[1024][8];
        CUDA_CHECK(hipMemcpyFromSymbol(h, HIP_SYMBOL(hcs_trace), sizeof(h)));
        const int nb_ = nwg*(int) nt;
        unsigned long long t0 = ~0ull, t2max = 0, t3max = 0, tend = 0;
        int last = -1;
        for (int i = 0; i < nb_; ++i) {
            t0 = std::min(t0, h[i][0]);
            t2max = std::max(t2max, h[i][2]);
            t3max = std::max(t3max, h[i][3]);
            if (h[i][6] > h[i][3] && h[i][6] - h[i][3] < 100000) { last = i; tend = std::max(tend, std::max(h[i][6], cur ? h[i][5] : 0ull)); }
        }
        double s01 = 0, s12 = 0, s0min = 1e9, s0max = 0;
        for (int i = 0; i < nb_; ++i) {
            s01 += (h[i][1] - h[i][0])*HCS_US; s12 += (h[i][2] - h[i][1])*HCS_US;
            s0min = std::min(s0min, (h[i][0] - t0)*HCS_US); s0max = std::max(s0max, (h[i][0] - t0)*HCS_US);
        }
        fprintf(stderr, "HCS nt %d nwg %d cur %d: starts %.2f..%.2f us, load+X' avg %.2f, dots+part avg %.2f, all parts %.2f, "
                "atomics %.2f", (int) nt, nwg, (int) cur, s0min, s0max, s01/nb_, s12/nb_, (t2max - t0)*HCS_US, (t3max - t0)*HCS_US);
        if (last >= 0) {
            // CUR: wave 0 finishes the mix first, the collapse ends after it
            fprintf(stderr, " | last wg %d: merge %.2f, mix finish %.2f, collapse end %.2f, end %.2f us\n", last,
                (h[last][4] - h[last][3])*HCS_US, (h[last][6] - h[last][4])*HCS_US,
                cur ? (h[last][5] - h[last][4])*HCS_US : 0.0, (tend - t0)*HCS_US);
        } else {
            fprintf(stderr, "\n");
        }
    }
#endif
}
