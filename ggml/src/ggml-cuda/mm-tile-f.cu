#include "mm-tile-f.cuh"

// Each wave computes an R (rows) x T (tokens) tile. Lanes stride K with 4 consecutive values per load (16 bytes of
// F32 or 8 bytes of BF16), so every weight row and activation column is read with coalesced loads, and each loaded
// value is reused T (weights) or R (activations) times from registers. The 4 waves of a block split K; the R*T
// partial sums of a wave are combined with a reduce-scatter over the lanes (R*T - 1 shuffles instead of
// R*T*log2(64)), leaving the total of tile entry i in lane i.
//
// Versus the few-rows kernels (all rows, 2-4 tokens per block) this cuts the weight re-reads from L2 by T/2..T/4,
// which is what bounds them at n ~ 320 (gfx906: 24 x 2560 at 320 tokens 52 us, of which ~5 us is DRAM traffic).
// BF16 weights up to 256 rows: rocBLAS has no fast BF16 GEMM on gfx906 (128 x 2560: 299 us); at 512 rows this
// tile is FMA-bound and slower than rocBLAS.

template <typename T_w>
static __device__ __forceinline__ float4 mm_tile_f_load_w(const T_w * p);

template <>
__device__ __forceinline__ float4 mm_tile_f_load_w<float>(const float * p) {
    return *(const float4 *) p;
}

template <>
__device__ __forceinline__ float4 mm_tile_f_load_w<nv_bfloat16>(const nv_bfloat16 * p) {
    const uint2 u = *(const uint2 *) p;
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
}

template <typename T_w, int R, int T, int NW>
static __global__ void __launch_bounds__(NW*64)
mm_tile_f_kernel(const T_w * __restrict__ w, const float * __restrict__ x, float * __restrict__ dst,
                 const int M, const int N, const int K, const int64_t sw, const int64_t sx, const int64_t sd) {
    constexpr int RT = R*T;
    static_assert(RT <= 64 && (RT & (RT - 1)) == 0, "tile must be a power of two <= 64");
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int m0   = blockIdx.x*R;
    const int n0   = blockIdx.y*T;

    const T_w   * wp[R];
    const float * xp[T];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        wp[r] = w + (int64_t) min(m0 + r, M - 1)*sw;
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        xp[t] = x + (int64_t) min(n0 + t, N - 1)*sx;
    }

    float v[RT];
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        v[i] = 0.0f;
    }

    for (int k = 4*(wave*64 + lane); k < K; k += 4*64*NW) {
        float4 xv[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            xv[t] = *(const float4 *) (xp[t] + k);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float4 wv = mm_tile_f_load_w<T_w>(wp[r] + k);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                v[r*T + t] += wv.x*xv[t].x + wv.y*xv[t].y + wv.z*xv[t].z + wv.w*xv[t].w;
            }
        }
    }

    // reduce-scatter over the 64 lanes: halve the live values each step, lane bit b selects the kept half
#pragma unroll
    for (int h = RT/2; h >= 1; h /= 2) {
        const bool upper = lane & h;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            const float send = upper ? v[i] : v[i + h];
            const float keep = upper ? v[i + h] : v[i];
            v[i] = keep + __shfl_xor(send, h, 64);
        }
    }
    // lanes l and l ^ (multiples of RT) now hold partial totals of entry l % RT: fold the remaining lane bits
#pragma unroll
    for (int off = RT; off < 64; off *= 2) {
        v[0] += __shfl_xor(v[0], off, 64);
    }

    __shared__ float part[NW][RT];
    if (lane < RT) {
        part[wave][lane] = v[0];
    }
    __syncthreads();
    if (threadIdx.x < RT) {
        float s = 0.0f;
#pragma unroll
        for (int q = 0; q < NW; ++q) {
            s += part[q][threadIdx.x];
        }
        const int r = threadIdx.x / T, t = threadIdx.x % T;
        if (m0 + r < M && n0 + t < N) {
            dst[(int64_t) (n0 + t)*sd + m0 + r] = s;
        }
    }
}

static int mm_tile_f_env() {
    static const int env = [] { const char * e = getenv("GGML_CUDA_MM_TILE_F"); return e ? atoi(e) : -1; }();
    return env;
}

bool ggml_cuda_mm_tile_f_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    const int env = mm_tile_f_env();
    if (env == 0 || (!GGML_CUDA_CC_IS_GCN(cc) && env != 1)) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // F32 weights: only where the few-rows kernels / rocBLAS are slower (2..64 rows); BF16: up to 256 rows (FMA-bound beyond).
    // 2..8 tokens (MTP verify): F32 weights with 128..4096 rows (router 512 x 2560 at 4 tokens: mmvf 17 us); fewer rows
    // take the split-K rows_small_t kernel (GGML_CUDA_MM_TILE_F_SMALL_N=1 on)
    static const bool small_n = [] { const char * e = getenv("GGML_CUDA_MM_TILE_F_SMALL_N"); return e && atoi(e) != 0; }(); // opt-in: 14 vs 18 us isolated, neutral in-model
    const bool small = small_n && N >= 2 && N <= 8 && src0->type == GGML_TYPE_F32 && M >= 128 && M <= 4096;
    if ((N <= 8 && !small) || M < 2 || (!small && M > (src0->type == GGML_TYPE_F32 ? 64 : 256)) || K % 4 != 0 || K > INT_MAX || M > INT_MAX || N > INT_MAX) {
        return false;
    }
    const size_t ts = ggml_type_size(src0->type);
    return src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->nb[0] == ts && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src0->nb[1] % (4*ts) == 0 && src1->nb[1] % 16 == 0 &&
        ((uintptr_t) src0->data) % (4*ts) == 0 && ((uintptr_t) src1->data) % 16 == 0;
}

template <typename T_w>
static void mm_tile_f_launch(const T_w * w, const float * x, float * d, int64_t M, int64_t N, int64_t K,
                             int64_t sw, int64_t sx, int64_t sd, cudaStream_t stream) {
#define MM_TILE_F_CASE(R, T, NW) do { \
        const dim3 grid((M + (R) - 1)/(R), (N + (T) - 1)/(T), 1); \
        mm_tile_f_kernel<T_w, R, T, NW><<<grid, (NW)*64, 0, stream>>>(w, x, d, (int) M, (int) N, (int) K, sw, sx, sd); \
    } while (0)
    // gfx906, 320 tokens: 4 x 2560 10 us (few-rows 18), 4 x 10240 31 us with 8 waves (44), 24 x 2560 21 us (52),
    // 48 x 2560 38 us (60), BF16 128 x 2560 81 us (rocBLAS 299). 16 waves spill (64 VGPRs at 1024 threads).
    if (N <= 8 && M >= 128) {
        // small batches of wide matrices: tokens in one tile
        static const int cfg = [] { const char * e = getenv("GGML_CUDA_MM_TILE_F_SN_CFG"); return e ? atoi(e) : 0; }();
        if (N <= 4) {
            switch (cfg) {
                case 1:  MM_TILE_F_CASE(1, 4, 4); break;
                case 7:  MM_TILE_F_CASE(4, 4, 4); break;
                case 3:  MM_TILE_F_CASE(2, 4, 8); break;
                case 4:  MM_TILE_F_CASE(1, 4, 8); break;
                case 5:  MM_TILE_F_CASE(4, 4, 8); break;
                case 6:  MM_TILE_F_CASE(8, 4, 4); break;
                default: MM_TILE_F_CASE(2, 4, 4); break; // gfx906 512 x 2560 at 4 tokens: 14.0 us (mmvf 17.8, 4x4 20.1)
            }
        } else {
            MM_TILE_F_CASE(2, 8, 4);
        }
    } else if (M <= 2) {
        MM_TILE_F_CASE(2, 8, 4);
    } else if (M <= 4) {
        if (K >= 8192) {
            MM_TILE_F_CASE(4, 2, 8);
        } else {
            MM_TILE_F_CASE(4, 2, 4);
        }
    } else {
        MM_TILE_F_CASE(8, 8, 4);
    }
#undef MM_TILE_F_CASE
}

void ggml_cuda_mm_tile_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t sw = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t sx = src1->nb[1] / sizeof(float), sd = dst->nb[1] / sizeof(float);
    cudaStream_t stream = ctx.stream();
    if (src0->type == GGML_TYPE_F32) {
        mm_tile_f_launch((const float *) src0->data, (const float *) src1->data, (float *) dst->data, M, N, K, sw, sx, sd, stream);
    } else {
        mm_tile_f_launch((const nv_bfloat16 *) src0->data, (const float *) src1->data, (float *) dst->data, M, N, K, sw, sx, sd, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
