#include "gcn-q8-matvec.cuh"
#include "quantize.cuh"
#include "mmvq.cuh"

// Dense q8_0 x q8_1 matvec for 1..4 tokens on GCN (gfx906). MMVQ at 2-4 tokens reaches only ~55% of the DRAM bandwidth
// on large matrices (10240 x 2560: 61 us, 454 GB/s) and no block/warp shape of it did better. Here the tokens'
// quantized activation rows sit in LDS (loaded once per block), LPR lanes share a weight row and read adjacent 68-byte
// block pairs (17 dwords: two q8_0 blocks), and a block walks RG groups of 256/LPR consecutive rows. gfx906, 4 tokens:
// 10240 x 2560 61.5 -> 52.4 us, 2560 x 6144 39.1 -> 36.5, 6144 x 2560 38.8 -> 35.7. Used for 2-4 tokens and >= 4096
// rows (1280 rows: too few blocks, 12.0 -> 16.7; one token: MMVQ is at ~82% already and this kernel is 2x slower).
// Opt-in (GGML_CUDA_GCN_Q8_MV=4): faster in isolation, but the MTP server measured 64.0 -> 62.9 t/s (mixed prompt,
// same draft acceptance), so not the default.
template <int NT, int LPR, int RG>
__launch_bounds__(256)
static __global__ void gcn_q8_mv(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride) {
    extern __shared__ int ys[]; // NT rows of K/32 block_q8_1 (9 ints each)
    const int nyb = K / QK8_1;
    for (int t = 0; t < NT; ++t) {
        const int * src = (const int *) (y + t*y_stride);
        for (int i = threadIdx.x; i < nyb*9; i += 256) {
            ys[t*nyb*9 + i] = src[i];
        }
    }
    __syncthreads();
    constexpr int RP = 256/LPR;
    const int sub = threadIdx.x % LPR;
    const int npairs = K / 64;
#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*RP + threadIdx.x / LPR;
        if (row >= M) {
            break;
        }
        const int * wr = (const int *) (W + (int64_t) row*w_stride);
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[t] = 0.0f;
        }
        for (int p = sub; p < npairs; p += LPR) {
            int w[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                w[i] = wr[17*p + i];
            }
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
                const int * b0 = ys + (t*nyb + 2*p)*9;
                const int * b1 = b0 + 9;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s0 = ggml_cuda_dp4a(q0[j], b0[1 + j], s0);
                    s1 = ggml_cuda_dp4a(q1[j], b1[1 + j], s1);
                }
                const float dy0 = __low2float(*(const half2 *) b0);
                const float dy1 = __low2float(*(const half2 *) b1);
                acc[t] += d0*dy0*(float) s0 + d1*dy1*(float) s1;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[t];
#pragma unroll
            for (int off = LPR/2; off > 0; off >>= 1) {
                a += __shfl_xor(a, off, LPR);
            }
            if (sub == 0) {
                dst[t*d_stride + row] = a;
            }
        }
    }
}

// Split-K q8_0 x q8_1 matvec for 1..8 tokens on GCN (DeepSeek V4 TP4 verify: q_a / kv / shared gate,up 512..1024 x
// 4096, where MMVQ reaches 180-260 GB/s at 6 tokens and gcn_q8_mv has too few row groups). Workgroup = RPB = 256/LPR rows
// x one K slice of 64*LPR quants: lane = one 68-byte block pair of its row, loaded before the slice of every token's
// activation is staged in LDS (both in flight at once). With several slices the partial sums go to a scratch buffer and
// the slice that arrives last (counter per row group) adds them in slice order and writes the rows (deterministic).
template <int NT, int LPR>
__launch_bounds__(256)
static __global__ void gcn_q8_mvk(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        float * __restrict__ part, int * __restrict__ counters, const int M, const int64_t w_stride, const int64_t y_stride,
        const int64_t d_stride) {
    constexpr int RPB = 256/LPR;             // rows per workgroup
    constexpr int YB  = 2*LPR;               // q8_1 blocks of the slice
    constexpr int NY  = NT*YB*9;             // ints of the staged activation
    __shared__ int ys[NY];
    __shared__ int s_last;
    const int nsplit = gridDim.y;
    const int split  = blockIdx.y;
    const int sub    = threadIdx.x % LPR;
    const int r      = threadIdx.x / LPR;
    const int row    = blockIdx.x*RPB + r;

    const int * wr = (const int *) (W + (int64_t) min(row, M - 1)*w_stride) + 17*(split*LPR + sub);
    int w[17];
#pragma unroll
    for (int i = 0; i < 17; ++i) {
        w[i] = wr[i];
    }
    int yv[(NY + 255)/256];
#pragma unroll
    for (int u = 0; u < (NY + 255)/256; ++u) {
        const int i = 256*u + threadIdx.x;
        yv[u] = i < NY ? ((const int *) (y + (i/(YB*9))*y_stride + split*YB))[i % (YB*9)] : 0;
    }
#pragma unroll
    for (int u = 0; u < (NY + 255)/256; ++u) {
        const int i = 256*u + threadIdx.x;
        if (i < NY) {
            ys[i] = yv[u];
        }
    }
    __syncthreads();

    const float d0 = __half2float(__ushort_as_half((unsigned short) (w[0] & 0xFFFF)));
    const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[8] >> 16)));
    int q0[8], q1[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        q0[j] = __builtin_amdgcn_alignbyte(w[j + 1], w[j], 2);
        q1[j] = w[9 + j];
    }
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int * b0 = ys + (t*YB + 2*sub)*9;
        const int * b1 = b0 + 9;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s0 = ggml_cuda_dp4a(q0[j], b0[1 + j], s0);
            s1 = ggml_cuda_dp4a(q1[j], b1[1 + j], s1);
        }
        acc[t] = d0*__low2float(*(const half2 *) b0)*(float) s0 + d1*__low2float(*(const half2 *) b1)*(float) s1;
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, LPR);
        }
    }
    if (nsplit == 1) {
        if (sub == 0 && row < M) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                dst[t*d_stride + row] = acc[t];
            }
        }
        return;
    }
    float * pb = part + (int64_t) (blockIdx.x*nsplit + split)*(NT*RPB);
    if (sub == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            pb[t*RPB + r] = acc[t];
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        s_last = atomicAdd(counters + blockIdx.x, 1) == nsplit - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    for (int i = threadIdx.x; i < NT*RPB; i += 256) {
        const float * pr = part + (int64_t) blockIdx.x*nsplit*(NT*RPB) + i;
        float a = 0.0f;
        for (int k = 0; k < nsplit; ++k) {
            a += pr[k*(NT*RPB)];
        }
        const int rw = blockIdx.x*RPB + i % RPB;
        if (rw < M) {
            dst[(i / RPB)*d_stride + rw] = a;
        }
    }
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = 0;
    }
}

// One token, gfx906 (e.g. the attention / GDN output projections, 2560 x 3072 per device: MMVQ ~20 us in the graph for
// 8.4 MB, floor ~11 us). A row takes 16 lanes; lane l loads block pairs l, l+16, ... (PPL of them, 17 dwords each) all
// before the first use, and the small activation load is issued first so its in-order return does not wait on the
// weights. The activation goes to LDS behind an LDS-only barrier (__syncthreads would drain the weight loads), and the
// 16-lane row sum uses DPP moves. 4 rows per 64-thread block (640 blocks balance better over 60 CUs than 160).
template <int CTRL, int BANKS>
static __device__ __forceinline__ int q8mv1_dpp(const int old, const int v) {
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xF, BANKS, false);
}
static __device__ __forceinline__ float q8mv1_row16_sum(float x) {
    int v = __float_as_int(x);
    x += __int_as_float(q8mv1_dpp<0x118, 0xC>(q8mv1_dpp<0x108, 0x3>(v, v), v)); v = __float_as_int(x); // xor 8
    x += __int_as_float(q8mv1_dpp<0x114, 0xA>(q8mv1_dpp<0x104, 0x5>(v, v), v)); v = __float_as_int(x); // xor 4
    x += __int_as_float(q8mv1_dpp<0x4E, 0xF>(v, v));                            v = __float_as_int(x); // xor 2
    x += __int_as_float(q8mv1_dpp<0xB1, 0xF>(v, v));                                                   // xor 1
    return x;
}

template <int PPL, int RPB = 4>
__launch_bounds__(RPB*16)
static __global__ void gcn_q8_mv1(const char * __restrict__ W, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride) {
    extern __shared__ int ys[]; // K/32 block_q8_1 (9 ints each)
    const int nyi = (K / QK8_1) * 9;
    const int l16 = threadIdx.x % 16;
    const int row = blockIdx.x * RPB + threadIdx.x / 16;
    // activation first (in-order return), then every weight load of this lane
    constexpr int NTH = RPB*16;
    constexpr int YPT = (1024 + NTH - 1)/NTH; // ints per thread; the launcher checks nyi <= 1024
    int yv[YPT];
#pragma unroll
    for (int u = 0; u < YPT; ++u) {
        const int i = threadIdx.x + NTH*u;
        yv[u] = i < nyi ? ((const int *) y)[i] : 0;
    }
    const int * wr = (const int *) (W + (int64_t) min(row, M - 1) * w_stride);
    int w[PPL][17];
#pragma unroll
    for (int j = 0; j < PPL; ++j) {
#pragma unroll
        for (int i = 0; i < 17; ++i) {
            w[j][i] = wr[17*(l16 + 16*j) + i];
        }
    }
#pragma unroll
    for (int u = 0; u < YPT; ++u) {
        const int i = threadIdx.x + NTH*u;
        if (i < nyi) { ys[i] = yv[u]; }
    }
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PPL; ++j) {
        const int p = l16 + 16*j;
        const float d0 = __half2float(__ushort_as_half((unsigned short) (w[j][0] & 0xFFFF)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned int) w[j][8] >> 16)));
        const int * b0 = ys + (2*p)*9;
        const int * b1 = b0 + 9;
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s0 = ggml_cuda_dp4a(__builtin_amdgcn_alignbyte(w[j][i + 1], w[j][i], 2), b0[1 + i], s0);
            s1 = ggml_cuda_dp4a(w[j][9 + i], b1[1 + i], s1);
        }
        acc += d0*__low2float(*(const half2 *) b0)*(float) s0 + d1*__low2float(*(const half2 *) b1)*(float) s1;
    }
    acc = q8mv1_row16_sum(acc);
    if (l16 == 0 && row < M) {
        dst[row] = acc;
    }
}

static bool ggml_cuda_gcn_q8_matvec1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // opt-in (GGML_CUDA_GCN_Q8_MV1=1): 2560 x 3072 bench 15.3 us vs an 11.5 us ideal 8.4 MB read, in the graph only ~2% per call
    // faster than MMVQ (not bit-identical), so not the default. Lessons: the per-lane 68 B chunk pattern caps small
    // matrices at ~550 GB/s; only lane-coalesced dwordx4 streams near peak (12.3 us), and staging those through LDS
    // (one wave per block) serialized each wave and was 3-4x slower.
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MV1"); return e && atoi(e) != 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1];
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 ||
            src0->ne[3] != 1 || !ggml_is_contiguous(src0) || src0->nb[1] % 4 != 0 || ((uintptr_t) src0->data) % 4 != 0 ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || M % 16 != 0 || M < 1024 ||
            (K/32)*9 > 256*4) {
        return false;
    }
    const int npairs = (int) (K / 64);
    if (K % 64 != 0 || npairs % 16 != 0 || npairs/16 > 4) {
        return false;
    }
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    const char * yq = ggml_cuda_q8_1_activation(ctx, src1, src0->type, K_pad, yq_local);
    const size_t smem = (size_t) (K/32)*9*sizeof(int);
    const dim3 grid((unsigned) (M/4));
    switch (npairs/16) {
        case 1: gcn_q8_mv1<1><<<grid, 64, smem, ctx.stream()>>>((const char *) src0->data, (const block_q8_1 *) yq, (float *) dst->data, (int) K, (int) M, src0->nb[1]); break;
        case 2: gcn_q8_mv1<2><<<grid, 64, smem, ctx.stream()>>>((const char *) src0->data, (const block_q8_1 *) yq, (float *) dst->data, (int) K, (int) M, src0->nb[1]); break;
        case 3: gcn_q8_mv1<3><<<grid, 64, smem, ctx.stream()>>>((const char *) src0->data, (const block_q8_1 *) yq, (float *) dst->data, (int) K, (int) M, src0->nb[1]); break;
        default: gcn_q8_mv1<4><<<grid, 64, smem, ctx.stream()>>>((const char *) src0->data, (const block_q8_1 *) yq, (float *) dst->data, (int) K, (int) M, src0->nb[1]); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// GGML_CUDA_GCN_Q8_MVK: 0 (default) off, 1 the shapes where it measured faster, 2 every supported 1..8-token case.
// Opt-in: DeepSeek V4 6-token verify only +0.5% in the graph (pp6 170.8 -> 171.7), and the changed summation order moved
// a greedy DSpark answer (sky prompt: 98 -> 74 tokens, drafts 66/160 -> 45/145).
// gfx906 (us, MMVQ / gcn_q8_mv -> this): 1 token 8192 x 1024 15.6 -> 13.3, 4096 x 512 8.0 -> 4.8 (K 4096: MMVQ ahead);
// 3-4 tokens: all faster (8192 x 1024 at 4: 30.1 -> 17.2); 6 tokens 1024 x 4096 17.0 -> 16.2, 512 x 4096 12.6 -> 11.0,
// 4096 x 512 8.5 -> 7.4, but 8192 x 1024 22.5 -> 24.9 (gcn_q8_mv); 8 tokens: all faster. 1-2 tokens stay on MMVQ: the
// short-K wins in isolation made DeepSeek V4 decode slower in the graph (tg 52.4 -> 52.0)
static bool ggml_cuda_gcn_q8_matvec_k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int env = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MVK"); return e ? atoi(e) : 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (env <= 0 || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < 1 || N > 8 || K % 512 != 0 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0) || src0->nb[1] % 4 != 0 ||
            ((uintptr_t) src0->data) % 4 != 0 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || M > 65535*16) {
        return false;
    }
    if (env == 1) {
        const bool use = N <= 2 ? false :
                         N <= 4 ? true :
                         N <= 7 ? M < 4096 || K <= 512 : true;
        if (!use) {
            return false;
        }
    }
    const int lpr    = K % 1024 == 0 ? 16 : 8;
    const int nsplit = (int) (K / (64*lpr));
    const int rpb    = 256/lpr;
    const int nblk   = (int) ((M + rpb - 1)/rpb);
    if (nsplit > 1 && nblk > 65535) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const char * yq = ggml_cuda_q8_1_activation(ctx, src1, src0->type, K_pad, yq_local);
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (nsplit > 1) {
        part.alloc((size_t) nblk*nsplit*N*rpb);
        if (ctx.hc_mix_counters == nullptr) {
            CUDA_CHECK(cudaMalloc((void **) &ctx.hc_mix_counters, 65535*sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(ctx.hc_mix_counters, 0, 65535*sizeof(int), stream));
        }
    }
    const int64_t ys_stride = K_pad/QK8_1;
    const int64_t ds = dst->nb[1]/sizeof(float);
    const dim3 grid((unsigned) nblk, (unsigned) nsplit);
#define Q8MVK(NT_, LPR_) gcn_q8_mvk<NT_, LPR_><<<grid, 256, 0, stream>>>((const char *) src0->data, (const block_q8_1 *) yq, \
        (float *) dst->data, part.get(), ctx.hc_mix_counters, (int) M, src0->nb[1], ys_stride, ds)
#define Q8MVK_L(NT_) if (lpr == 16) { Q8MVK(NT_, 16); } else { Q8MVK(NT_, 8); }
    switch (N) {
        case 1:  Q8MVK_L(1); break;
        case 2:  Q8MVK_L(2); break;
        case 3:  Q8MVK_L(3); break;
        case 4:  Q8MVK_L(4); break;
        case 5:  Q8MVK_L(5); break;
        case 6:  Q8MVK_L(6); break;
        case 7:  Q8MVK_L(7); break;
        default: Q8MVK_L(8); break;
    }
#undef Q8MVK_L
#undef Q8MVK
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_gcn_q8_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_gcn_q8_matvec_k(ctx, src0, src1, dst)) {
        return true;
    }
    // max / min tokens (GGML_CUDA_GCN_Q8_MV=0: off). Default 5..16 (2026-10-03, GLM-5.3 Q4 on 8 MI50s, batched decode
    // S_TG: 6 seqs 112.7 -> 118.7, 8: 125.7 -> 142.5, 12: 99.1 -> 120.1, 16: 120.0 -> 145.9; 1..4 tokens neutral)
    static const int env_max = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MV"); return e ? atoi(e) : 16; }();
    static const int env_min = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MV_MIN"); return e ? atoi(e) : 5; }();
    static const int env_min_rows = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MV_MIN_ROWS"); return e ? atoi(e) : 4096; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N == 1 && ggml_cuda_gcn_q8_matvec1(ctx, src0, src1, dst)) {
        return true;
    }
    if (env_max <= 0 || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < env_min || N > env_max || K % 64 != 0 || M < env_min_rows ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0) ||
            src0->nb[1] % 4 != 0 || ((uintptr_t) src0->data) % 4 != 0 || src1->nb[0] != sizeof(float) ||
            dst->nb[0] != sizeof(float) || (size_t) std::min<int64_t>(N, 8)*(K/QK8_1)*sizeof(block_q8_1) > 48*1024) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> yq_local(ctx.pool());
    const char * yq = ggml_cuda_q8_1_activation(ctx, src1, src0->type, K_pad, yq_local);
    static const int lpr_env = [] { const char * e = getenv("GGML_CUDA_GCN_Q8_MV_LPR"); return e ? atoi(e) : 8; }();
    constexpr int RP8 = 256/8;
    const int rg = (int) std::max<int64_t>(1, std::min<int64_t>(4, M / (RP8*240)));
    const int64_t ys_stride = K_pad/QK8_1;
    const int64_t ds = dst->nb[1]/sizeof(float);
    // up to 8 tokens per launch (the kernel's NT is the number of tokens it computes; until 2026-10-03 5..16 tokens all
    // took NT = 4 and left tokens 5+ unwritten: DeepSeek V4 PPL at 8-token ubatches 14.9 -> 4100)
    for (int64_t t0 = 0; t0 < N; t0 += 8) {
        const int nt = (int) std::min<int64_t>(8, N - t0);
        const size_t smem = (size_t) nt*(K/QK8_1)*sizeof(block_q8_1);
        const block_q8_1 * yt = (const block_q8_1 *) yq + t0*ys_stride;
        float * dt = (float *) dst->data + t0*ds;
#define Q8MV(NT_, LPR_, RG_) gcn_q8_mv<NT_, LPR_, RG_><<<(unsigned) ((M + (256/LPR_)*RG_ - 1)/((256/LPR_)*RG_)), 256, smem, stream>>>( \
        (const char *) src0->data, yt, dt, (int) K, (int) M, src0->nb[1], ys_stride, ds)
#define Q8MV_RG(NT_, LPR_) switch (rg) { case 1: Q8MV(NT_, LPR_, 1); break; case 2: Q8MV(NT_, LPR_, 2); break; default: Q8MV(NT_, LPR_, 4); break; }
#define Q8MV_L(NT_) if (lpr_env == 16) { Q8MV_RG(NT_, 16) } else if (lpr_env == 4) { Q8MV_RG(NT_, 4) } else { Q8MV_RG(NT_, 8) }
        switch (nt) {
            case 1:  Q8MV_L(1); break;
            case 2:  Q8MV_L(2); break;
            case 3:  Q8MV_L(3); break;
            case 4:  Q8MV_L(4); break;
            case 5:  Q8MV_L(5); break;
            case 6:  Q8MV_L(6); break;
            case 7:  Q8MV_L(7); break;
            default: Q8MV_L(8); break;
        }
#undef Q8MV_L
#undef Q8MV_RG
#undef Q8MV
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
