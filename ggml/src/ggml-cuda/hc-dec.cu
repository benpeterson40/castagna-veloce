#include "hc-dec.cuh"

#include <cstdlib>

// Decode-sized hyper-connection kernels for GCN (few tokens). At batch 1 the generic path spends most of the time
// on launch overhead and short-K matrix-vector products that leave most lanes idle; these kernels fuse the chains
// and give every lane one weight row.

static int hc_dec_env(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

// SCALE -> SILU -> q8_0 up projection -> RESHAPE -> gated DSV4_HC_PRE:
//   a[k,t]     = silu(s_lo*lo[k,t])
//   g[c,e,t]   = sum_k W[c*E + e, k] * a[k,t]
//   pre[e,t]   = s_pre * sum_c xn[e,c,t] * sigmoid(g[c,e,t])
// Block = 64 output columns x HC_DEC_NS streams (one wave per stream, one lane per weight row); the activations sit in
// LDS (broadcast reads), the weights are read as raw q8_0 (pairs of blocks = 17 dwords) and multiplied in float, so
// no activation quantization kernel is needed. The gate is never stored.
#define HC_DEC_NS 4
#ifndef HC_DEC_WIDE
#define HC_DEC_WIDE 0
#endif
#ifndef HC_DEC_TUNROLL
#define HC_DEC_TUNROLL 4
#endif
#ifndef HC_DEC_PRELOAD
#define HC_DEC_PRELOAD 5
#endif

// T lanes share a weight row and split its block pairs (lane = row*T + sub: the T lanes of a row read adjacent pairs),
// so a block covers 64/T output columns and the grid has T times more blocks than one lane per row would give
template <int NT, int T>
__launch_bounds__(64*HC_DEC_NS)
static __global__ void hc_up_pre_dec(
        const char * __restrict__ W, const float * __restrict__ lo, const float * __restrict__ xn, float * __restrict__ dst,
        const int K, const int64_t w_stride, const int E, const float s_lo, const float s_pre,
        const int64_t lo_st, const int64_t xn_sc, const int64_t xn_st, const int64_t d_st) {
    constexpr int CPB = 64/T; // output columns per block
    extern __shared__ float hc_act[]; // [NT][K]
    __shared__ float red[HC_DEC_NS][NT][CPB];

    const int tid = threadIdx.x;
    for (int i = tid; i < NT*K; i += 64*HC_DEC_NS) {
        const int t = i / K;
        const int k = i - t*K;
        const float v = s_lo*lo[t*lo_st + k];
        hc_act[i] = v/(1.0f + expf(-v));
    }
    __syncthreads();

    const int  c     = tid / 64;
    const int  lane  = tid % 64;
    const int  r     = lane / T;
    const int  sub   = lane % T;
    const int  e     = blockIdx.x*CPB + r;
    const bool valid = e < E;
    const int * wr   = (const int *) (W + (int64_t) (c*E + (valid ? e : E - 1))*w_stride);

    float acc[NT] = {0.0f};
    const int npairs = K / 64;
    // one lane per row (T == 1) with K = 320: all 5 pairs are loaded before any is used, so the lane waits for one
    // memory round trip instead of five (few waves per CU: the loop was bound by the per-pair load latency)
    if (T == 1 && NT == 1 && npairs == HC_DEC_PRELOAD) {
        int wall[HC_DEC_PRELOAD][17];
#pragma unroll
        for (int p = 0; p < HC_DEC_PRELOAD; ++p) {
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                wall[p][i] = wr[17*p + i];
            }
        }
#pragma unroll
        for (int p = 0; p < HC_DEC_PRELOAD; ++p) {
            const int * w = wall[p];
            const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
            const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
            int q0[8], q1[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                q0[j] = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2);
                q1[j] = w[9 + j];
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 * a = (const float4 *) (hc_act + t*K + 64*p);
                float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float4 a0 = a[j];
                    const float4 a1 = a[8 + j];
                    s0 += (float) (int8_t) (q0[j]      )*a0.x + (float) (int8_t) (q0[j] >>  8)*a0.y +
                          (float) (int8_t) (q0[j] >> 16)*a0.z + (float) (int8_t) (q0[j] >> 24)*a0.w;
                    s1 += (float) (int8_t) (q1[j]      )*a1.x + (float) (int8_t) (q1[j] >>  8)*a1.y +
                          (float) (int8_t) (q1[j] >> 16)*a1.z + (float) (int8_t) (q1[j] >> 24)*a1.w;
                }
                acc[t] += d0*s0 + d1*s1;
            }
        }
    } else
    for (int p = sub; p < npairs; p += T) {
        int w[17];
        if (HC_DEC_WIDE) {
            // 68 bytes as 4 x 16-byte + 1 x 4-byte loads (dword-aligned, gfx9 unaligned mode): each load instruction
            // touches 64 cache lines (one row per lane), so fewer, wider loads cut the L1/TA line requests ~3x
            const int4 * w4 = (const int4 *) (wr + 17*p);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int4 v = w4[i];
                w[4*i + 0] = v.x; w[4*i + 1] = v.y; w[4*i + 2] = v.z; w[4*i + 3] = v.w;
            }
            w[16] = wr[17*p + 16];
        } else {
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                w[i] = wr[17*p + i];
            }
        }
        const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
        int q0[8], q1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q0[j] = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2); // quants of block 0 start at byte 2
            q1[j] = w[9 + j];                                     // block 1: d at bytes 34-35, quants from byte 36
        }
        // tokens fully unrolled (131-147 VGPRs, 1 wave per SIMD at 4 tokens, but limiting the unroll was slower)
#pragma unroll HC_DEC_TUNROLL
        for (int t = 0; t < NT; ++t) {
            const float4 * a = (const float4 *) (hc_act + t*K + 64*p);
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 a0 = a[j];
                const float4 a1 = a[8 + j];
                s0 += (float) (int8_t) (q0[j]      )*a0.x + (float) (int8_t) (q0[j] >>  8)*a0.y +
                      (float) (int8_t) (q0[j] >> 16)*a0.z + (float) (int8_t) (q0[j] >> 24)*a0.w;
                s1 += (float) (int8_t) (q1[j]      )*a1.x + (float) (int8_t) (q1[j] >>  8)*a1.y +
                      (float) (int8_t) (q1[j] >> 16)*a1.z + (float) (int8_t) (q1[j] >> 24)*a1.w;
            }
            acc[t] += d0*s0 + d1*s1;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = T/2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }

    if (sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            red[c][t][r] = valid ? xn[e + c*xn_sc + t*xn_st] / (1.0f + expf(-acc[t])) : 0.0f;
        }
    }
    __syncthreads();
    if (c == 0 && sub == 0 && valid) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float s = 0.0f;
#pragma unroll
            for (int cc = 0; cc < HC_DEC_NS; ++cc) {
                s += red[cc][t][r];
            }
            dst[e + t*d_st] = s_pre*s;
        }
    }
}

bool ggml_cuda_hc_up_pre_dec_supported(int cc, const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * mm,
                                       const ggml_tensor * pre) {
    static const int enabled = hc_dec_env("GGML_CUDA_HC_DEC", 1);
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * xn = pre->src[0];
    const int64_t nt = lo->ne[1];
    return ggml_get_op_params_f32(scale, 1) == 0.0f && // no bias
        lo->type == GGML_TYPE_F32 && lo->ne[2] == 1 && lo->ne[3] == 1 && lo->nb[0] == sizeof(float) &&
        nt >= 1 && nt <= 4 && mm->src[1] == silu &&
        w->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(w) && w->ne[0] == lo->ne[0] && w->ne[0] % 64 == 0 &&
        w->ne[0]*nt*sizeof(float) <= 32768 && ((uintptr_t) w->data) % 4 == 0 &&
        ggml_get_op_params_i32(pre, 1) != 0 && // gated
        xn->type == GGML_TYPE_F32 && xn->ne[1] == HC_DEC_NS && xn->ne[2] == nt && xn->nb[0] == sizeof(float) &&
        w->ne[1] == HC_DEC_NS*xn->ne[0] && pre->type == GGML_TYPE_F32 && pre->nb[0] == sizeof(float) &&
        pre->ne[0] == xn->ne[0] && pre->ne[1] == nt;
}

void ggml_cuda_hc_up_pre_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * mm, ggml_tensor * pre) {
    const ggml_tensor * lo = scale->src[0];
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * xn = pre->src[0];
    const int K  = (int) w->ne[0];
    const int E  = (int) xn->ne[0];
    const int nt = (int) lo->ne[1];
    static const int t_env = hc_dec_env("GGML_CUDA_HC_DEC_T", 0);
    // lanes per weight row: gfx906, E = 2560, K = 320: 8 at one token (more blocks), 1 at 2-4 tokens (verify step
    // 89.0 -> 93.5 t/s against 4). Tried and slower or equal: dp4a on int8-quantized activations, weights staged
    // through LDS with coalesced loads (the kernel is bound by its chain of dependent loads, not ALU or access pattern)
    // one token: 1 lane per row with all 5 block pairs preloaded (one memory round trip): 14.0 -> 10.6 us per call
    // (8 lanes per row before). Several tokens: 2 lanes per row sharing the tokens, and fewer unrolled tokens, both
    // measured no better (16.0 us at 4 tokens).
    const int T = t_env ? t_env : 1;
    const dim3 grid((E + 64/T - 1)/(64/T));
    const size_t smem = (size_t) nt*K*sizeof(float);
    const float s_lo  = ggml_get_op_params_f32(scale, 0);
    const float s_pre = ggml_get_op_params_f32(pre, 0);
    cudaStream_t stream = ctx.stream();
#define HC_UP_PRE(NT, T_) hc_up_pre_dec<NT, T_><<<grid, 64*HC_DEC_NS, smem, stream>>>((const char *) w->data, (const float *) lo->data, \
        (const float *) xn->data, (float *) pre->data, K, w->nb[1], E, s_lo, s_pre, lo->nb[1]/sizeof(float), xn->nb[1]/sizeof(float), \
        xn->nb[2]/sizeof(float), pre->nb[1]/sizeof(float))
#define HC_UP_PRE_T(NT) if (T == 1) { HC_UP_PRE(NT, 1); } else if (T == 2) { HC_UP_PRE(NT, 2); } else if (T == 4) { HC_UP_PRE(NT, 4); } \
        else { HC_UP_PRE(NT, 8); }
    switch (nt) {
        case 1:  HC_UP_PRE_T(1); break;
        case 2:  HC_UP_PRE_T(2); break;
        case 3:  HC_UP_PRE_T(3); break;
        default: HC_UP_PRE_T(4); break;
    }
#undef HC_UP_PRE_T
#undef HC_UP_PRE
    CUDA_CHECK(cudaGetLastError());
}

// GDN gate inputs at decode: MUL_MAT(alpha) -> RESHAPE -> ADD(dt) -> SOFTPLUS -> MUL(A) and MUL_MAT(beta) -> RESHAPE ->
// SIGMOID share the activation: one wave per output row (alpha rows first, then beta rows), f32 weights
template <int NT>
__launch_bounds__(256)
static __global__ void gdn_ab_dec(
        const float * __restrict__ x, const float * __restrict__ wa, const float * __restrict__ wb, const float * __restrict__ dt,
        const float * __restrict__ A, float * __restrict__ gate, float * __restrict__ beta, const int K, const int H,
        const int64_t x_st, const int64_t wa_s, const int64_t wb_s, const int64_t g_st, const int64_t b_st) {
    const int r    = blockIdx.x*4 + threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    if (r >= 2*H) {
        return;
    }
    const bool is_b = r >= H;
    const int  h    = is_b ? r - H : r;
    const float4 * w4 = (const float4 *) (is_b ? wb + h*wb_s : wa + h*wa_s);
    float acc[NT] = {0.0f};
    for (int k = lane; k < K/4; k += 64) {
        const float4 w = w4[k];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 v = ((const float4 *) (x + t*x_st))[k];
            acc[t] += w.x*v.x + w.y*v.y + w.z*v.z + w.w*v.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (is_b) {
                beta[h + t*b_st] = 1.0f/(1.0f + expf(-acc[t]));
            } else {
                const float v  = acc[t] + dt[h];
                const float sp = v > 20.0f ? v : logf(1.0f + expf(v)); // as ggml softplus
                gate[h + t*g_st] = sp*A[h];
            }
        }
    }
}

// nodes: mm_a, (reshape), add, softplus, mul, (reshape), mm_b, (reshape), sigmoid
bool ggml_cuda_gdn_ab_dec_supported(int cc, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * sp,
                                    const ggml_tensor * mul, const ggml_tensor * mm_b, const ggml_tensor * sig) {
    static const int enabled = hc_dec_env("GGML_CUDA_GDN_AB_DEC", 1);
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * dt = add->src[1];
    const ggml_tensor * A  = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    const int64_t nt = x->ne[1];
    return mm_b->src[1] == x && x->type == GGML_TYPE_F32 && wa->type == GGML_TYPE_F32 && wb->type == GGML_TYPE_F32 &&
        nt >= 1 && nt <= 4 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->ne[0] % 4 == 0 &&
        (x->nb[1] % 16) == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ggml_is_contiguous(wa) && ggml_is_contiguous(wb) && ggml_are_same_shape(wa, wb) && wa->ne[0] == x->ne[0] &&
        wa->ne[2] == 1 && ((uintptr_t) wa->data) % 16 == 0 && ((uintptr_t) wb->data) % 16 == 0 &&
        add->src[0]->view_src == mm_a && ggml_is_contiguous(add) && ggml_is_contiguous(mm_a) &&
        dt->type == GGML_TYPE_F32 && ggml_nelements(dt) == wa->ne[1] && ggml_is_contiguous(dt) &&
        sp->src[0] == add && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS &&
        (mul->src[0] == sp || mul->src[1] == sp) && A->type == GGML_TYPE_F32 && ggml_nelements(A) == wa->ne[1] &&
        ggml_is_contiguous(A) && ggml_is_contiguous(mul) && ggml_nelements(mul) == wa->ne[1]*nt &&
        ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0]->view_src == mm_b && ggml_is_contiguous(mm_b) &&
        ggml_is_contiguous(sig) && ggml_nelements(sig) == wa->ne[1]*nt && mul->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32;
}

void ggml_cuda_gdn_ab_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * mm_b,
                          ggml_tensor * gate, ggml_tensor * sig) {
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * sp = gate->src[0]->op == GGML_OP_UNARY ? gate->src[0] : gate->src[1];
    const ggml_tensor * A  = gate->src[0] == sp ? gate->src[1] : gate->src[0];
    const int K = (int) x->ne[0], H = (int) wa->ne[1], nt = (int) x->ne[1];
    const dim3 grid((2*H + 3)/4);
    cudaStream_t stream = ctx.stream();
#define GDN_AB(NT) gdn_ab_dec<NT><<<grid, 256, 0, stream>>>((const float *) x->data, (const float *) wa->data, (const float *) wb->data, \
        (const float *) add->src[1]->data, (const float *) A->data, (float *) gate->data, (float *) sig->data, K, H, \
        x->nb[1]/sizeof(float), wa->nb[1]/sizeof(float), wb->nb[1]/sizeof(float), H, H)
    switch (nt) {
        case 1:  GDN_AB(1); break;
        case 2:  GDN_AB(2); break;
        case 3:  GDN_AB(3); break;
        default: GDN_AB(4); break;
    }
#undef GDN_AB
    CUDA_CHECK(cudaGetLastError());
}

// Shared-expert gate tail: MUL_MAT(w_g [K], x) -> SIGMOID -> MUL(shexp, .) -> ADD(moe, .) in one kernel:
//   out[e,t] = moe[e,t] + shexp[e,t] * sigmoid(dot(w_g, x[:,t]))
// One block per token (the dot product needs the whole row first). Four launches -> one at decode (<= 8 tokens).
static __global__ void __launch_bounds__(256) shexp_gate_add(
        const float * __restrict__ wg, const float * __restrict__ x, const float * __restrict__ shexp,
        const float * __restrict__ moe, float * __restrict__ out, const int K, const int E,
        const int64_t s_x, const int64_t s_sh, const int64_t s_moe, const int64_t s_out) {
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 * xp = (const float4 *) (x + t*s_x);
    const float4 * wp = (const float4 *) wg;
    float acc = 0.0f;
    for (int k = tid; k < K/4; k += 256) {
        const float4 a = xp[k], b = wp[k];
        acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    __shared__ float red[256/ws];
    acc = warp_reduce_sum<ws>(acc);
    if (tid % ws == 0) {
        red[tid / ws] = acc;
    }
    __syncthreads();
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 256/ws; ++i) {
        dot += red[i];
    }
    const float g = 1.0f / (1.0f + expf(-dot));
    for (int e = tid; e < E; e += 256) {
        out[t*s_out + e] = moe[t*s_moe + e] + shexp[t*s_sh + e]*g;
    }
}

static bool shexp_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

bool ggml_cuda_shexp_gate_add_supported(const ggml_tensor * mm, const ggml_tensor * sig, const ggml_tensor * mul,
                                        const ggml_tensor * add) {
    static const int enabled = hc_dec_env("GGML_CUDA_SHEXP_GATE_FUSE", 1);
    if (!enabled) {
        return false;
    }
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    if (sig->src[0] != mm || !(mul->src[0] == sig || mul->src[1] == sig) || !(add->src[0] == mul || add->src[1] == mul)) {
        return false;
    }
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    const int64_t K = wg->ne[0], E = sh->ne[0], N = x->ne[1];
    // same-index elementwise tail (out may replace moe/shexp exactly); the dot inputs may not be overwritten
    // by another token's block
    const auto same_or_disjoint = [](const ggml_tensor * o, const ggml_tensor * in) {
        return !shexp_overlap(o, in) || (o->data == in->data && o->nb[1] == in->nb[1]);
    };
    // decode / verify only: at prefill the launches do not matter and the unfused path keeps prefill bit-identical
    return N <= 8 && wg->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && sh->type == GGML_TYPE_F32 && moe->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32 && wg->ne[1] == 1 && ggml_is_contiguous(wg) && K % 4 == 0 &&
        x->ne[0] == K && x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ((uintptr_t) wg->data) % 16 == 0 && x->ne[2] == 1 && x->ne[3] == 1 &&
        mm->ne[0] == 1 && mm->ne[1] == N && ggml_are_same_shape(sh, moe) && ggml_are_same_shape(add, sh) &&
        sh->ne[1] == N && sh->ne[2] == 1 && sh->ne[3] == 1 && sh->nb[0] == sizeof(float) && moe->nb[0] == sizeof(float) &&
        add->nb[0] == sizeof(float) && E > 0 &&
        !shexp_overlap(add, wg) && same_or_disjoint(add, x) && same_or_disjoint(add, sh) && same_or_disjoint(add, moe);
}

void ggml_cuda_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig,
                              const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    shexp_gate_add<<<(unsigned) x->ne[1], 256, 0, ctx.stream()>>>(
        (const float *) wg->data, (const float *) x->data, (const float *) sh->data, (const float *) moe->data,
        (float *) add->data, (int) wg->ne[0], (int) sh->ne[0], x->nb[1]/sizeof(float), sh->nb[1]/sizeof(float),
        moe->nb[1]/sizeof(float), add->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}
