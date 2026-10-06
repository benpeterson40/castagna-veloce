#include "gcn-f16-matvec.cuh"

// Dense F16 x F32 matvec for 1..8 tokens on GCN (gfx906). MMVF gives each row one block of <= 128 threads that walk the
// row in 4-byte half2 steps and sum in half precision (default precision); on DeepSeek V4 Flash (TP4 decode: compressor,
// indexer, router weights, all F16 and replicated on every GPU, ~1 GB per token) that ran at ~200 GB/s.
// Here a block converts the tokens' activations to f16 once into LDS (MMVF rounds them to f16 as well), each wave owns
// R rows per row group and streams them in lane-coalesced 16-byte loads (a wave reads 1 KB of one row per load), and
// the products are summed by v_dot2_f32_f16 into f32. KS waves can share a row group, each summing a slice of K (more
// waves in flight for short matrices and several tokens; partial sums meet in LDS). A block walking several row groups
// to spread its LDS fill measured slower (the groups run back to back). Reading the activations from global memory
// instead (f32, every wave the whole vector) left 4096-wide rows at ~250 GB/s: 16 KB per token does not stay in the
// 16 KB L1 next to the weight stream.
// GGML_CUDA_GCN_F16_MV=0: off (MMVF); GGML_CUDA_GCN_F16_MV_R=1/2/4 rows per wave, GGML_CUDA_GCN_F16_MV_KS=1/2/4 waves per row group.
typedef _Float16 f16mv_h2 __attribute__((ext_vector_type(2)));
#ifndef F16MV_UNROLL
#define F16MV_UNROLL 4
#endif

template <int NT, int R, int KS, int NW, bool MULTI = false>
__launch_bounds__(64*NW)
static __global__ void gcn_f16_mv(const char * __restrict__ W, const float * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride,
        const ggml_cuda_f16mv_list L) {
    extern __shared__ uint4 f16mv_ys[]; // NT rows of K halves
    __shared__ float red[KS > 1 ? NW*R*NT : 1]; // per-wave partial sums of the K slices
    const int nc = K / 8; // 16-byte chunks (8 halves) per row
    for (int t = 0; t < NT; ++t) {
        for (int c = threadIdx.x; c < nc; c += 64*NW) {
            const float4 a = *(const float4 *) (y + t*y_stride + 8*c);
            const float4 b = *(const float4 *) (y + t*y_stride + 8*c + 4);
            const f16mv_h2 h0 = {(_Float16) a.x, (_Float16) a.y};
            const f16mv_h2 h1 = {(_Float16) a.z, (_Float16) a.w};
            const f16mv_h2 h2 = {(_Float16) b.x, (_Float16) b.y};
            const f16mv_h2 h3 = {(_Float16) b.z, (_Float16) b.w};
            f16mv_ys[t*nc + c] = make_uint4(__builtin_bit_cast(uint32_t, h0), __builtin_bit_cast(uint32_t, h1),
                                            __builtin_bit_cast(uint32_t, h2), __builtin_bit_cast(uint32_t, h3));
        }
    }
    __syncthreads();

    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int ks   = wave % KS;                                // K slice of this wave
    const int row0 = (blockIdx.x*(NW/KS) + wave / KS)*R;      // first of the wave's R rows
    const bool live = row0 < M;                                // no early exit: KS > 1 meets at a barrier
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const uint4 * wr[R];
    float * dr[R];    // MULTI: the row's output (its matrix's dst + local row)
    int64_t dsr[R];   // MULTI: its token stride
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int gr = min(row0 + r, M - 1); // rows past M read row M-1, never stored
        if constexpr (MULTI) {
            int m = 0;
            while (m < L.nm - 1 && gr >= L.row_end[m]) {
                ++m;
            }
            const int rs = m > 0 ? L.row_end[m - 1] : 0;
            wr[r]  = (const uint4 *) (L.W[m] + (int64_t) (gr - rs)*L.w_stride[m]);
            dr[r]  = L.dst[m] + (gr - rs);
            dsr[r] = L.d_stride[m];
        } else {
            wr[r]  = (const uint4 *) (W + (int64_t) gr*w_stride);
            dr[r]  = nullptr;
            dsr[r] = 0;
        }
    }
    // the row's 1 KB steps (64 lanes x 16 B) split over KS waves; rows of 2^n bytes put every wave's same offset on the
    // same DRAM channels (8 KB rows: ~265 GB/s), so each wave starts its walk at a different step of its slice
    const int ns   = (nc + 63) / 64;
    const int s_lo = ks*ns/KS;
    const int sn   = (ks + 1)*ns/KS - s_lo;
    const int rot  = sn > 0 ? (row0 / R) % sn : 0;
    if (live) {
#pragma unroll F16MV_UNROLL
        for (int s0 = 0; s0 < sn; ++s0) {
            const int sc = s_lo + (s0 + rot < sn ? s0 + rot : s0 + rot - sn);
            const int c  = sc*64 + lane;
            if (c >= nc) {
                continue;
            }
            uint4 q[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q[r] = wr[r][c];
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const uint4 yv = f16mv_ys[t*nc + c];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float s = acc[r][t];
                    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16mv_h2, q[r].x), __builtin_bit_cast(f16mv_h2, yv.x), s, false);
                    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16mv_h2, q[r].y), __builtin_bit_cast(f16mv_h2, yv.y), s, false);
                    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16mv_h2, q[r].z), __builtin_bit_cast(f16mv_h2, yv.z), s, false);
                    s = __builtin_amdgcn_fdot2(__builtin_bit_cast(f16mv_h2, q[r].w), __builtin_bit_cast(f16mv_h2, yv.w), s, false);
                    acc[r][t] = s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) {
                v += __shfl_xor(v, off, 64);
            }
            if constexpr (KS == 1) {
                if (lane == r*NT + t && live && row0 + r < M) {
                    if constexpr (MULTI) {
                        dr[r][t*dsr[r]] = v;
                    } else {
                        dst[t*d_stride + row0 + r] = v;
                    }
                }
            } else {
                if (lane == r*NT + t) {
                    red[wave*R*NT + r*NT + t] = v;
                }
            }
        }
    }
    if constexpr (KS > 1) {
        __syncthreads();
        const int j = lane; // (r, t) = (j / NT, j % NT)
        if (ks == 0 && j < R*NT && live && row0 + j / NT < M) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v += red[(wave + k)*R*NT + j];
            }
            if constexpr (MULTI) {
                // lane j's row is r = j / NT: pick its pointer without dynamic register indexing
                float * o = dr[0];
                int64_t os = dsr[0];
#pragma unroll
                for (int r = 1; r < R; ++r) {
                    if (j / NT == r) {
                        o  = dr[r];
                        os = dsr[r];
                    }
                }
                o[(j % NT)*os] = v;
            } else {
                dst[(j % NT)*d_stride + row0 + j / NT] = v;
            }
        }
    }
}

bool ggml_cuda_gcn_f16_matvec_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int env    = [] { const char * e = getenv("GGML_CUDA_GCN_F16_MV"); return e ? atoi(e) : 1; }();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    // the activations are rounded to f16 here (as MMVF does at default precision); ops that ask for F32 precision
    // (e.g. the mtmd encoders' matmuls) keep MMVF's f32 path
    if (ggml_prec(dst->op_params[0]) == GGML_PREC_F32) {
        return false;
    }
    return !(!env || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < 1 || N > 8 || K % 8 != 0 || N*K*sizeof(half) > 48*1024 || M > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[0] != sizeof(half) || src0->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 ||
            src1->nb[0] != sizeof(float) || (N > 1 && src1->nb[1] % 16 != 0) || ((uintptr_t) src1->data) % 16 != 0 ||
            dst->nb[0] != sizeof(float) || (N > 1 && dst->nb[1] % sizeof(float) != 0));
}

static void ggml_cuda_gcn_f16_matvec_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cuda_f16mv_list * list, const int64_t M_total);

bool ggml_cuda_gcn_f16_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_gcn_f16_matvec_supported(ctx.device, src0, src1, dst)) {
        return false;
    }
    ggml_cuda_gcn_f16_matvec_launch(ctx, src0, src1, dst, nullptr, src0->ne[1]);
    return true;
}

// several F16 matrices of one K on the same activations (e.g. DeepSeek V4's compressor, indexer compressor and indexer
// projections of attn_norm): one launch over their concatenated rows, each row finding its matrix in the list
bool ggml_cuda_gcn_f16_matvec_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mms, const int n) {
    if (n < 1 || n > GGML_CUDA_F16MV_MAXM) {
        return false;
    }
    ggml_cuda_f16mv_list L = {};
    int64_t rows = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * mm = mms[k];
        if (!ggml_cuda_gcn_f16_matvec_supported(ctx.device, mm->src[0], mm->src[1], mm) || mm->src[1] != mms[0]->src[1] ||
                mm->src[0]->ne[0] != mms[0]->src[0]->ne[0] || rows + mm->src[0]->ne[1] > INT_MAX) {
            return false;
        }
        rows += mm->src[0]->ne[1];
        L.W[k]        = (const char *) mm->src[0]->data;
        L.dst[k]      = (float *) mm->data;
        L.w_stride[k] = mm->src[0]->nb[1];
        L.d_stride[k] = mm->src[1]->ne[1] > 1 ? mm->nb[1] / sizeof(float) : 0;
        L.row_end[k]  = (int) rows;
    }
    L.nm = n;
    ggml_cuda_gcn_f16_matvec_launch(ctx, mms[0]->src[0], mms[0]->src[1], (ggml_tensor *) mms[0], &L, rows);
    return true;
}

static void ggml_cuda_gcn_f16_matvec_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cuda_f16mv_list * list, const int64_t M_total) {
    static const int env_r  = [] { const char * e = getenv("GGML_CUDA_GCN_F16_MV_R"); return e ? atoi(e) : 0; }();
    static const int env_ks = [] { const char * e = getenv("GGML_CUDA_GCN_F16_MV_KS"); return e ? atoi(e) : 0; }();
    const int64_t K = src0->ne[0], M = M_total, N = src1->ne[1];
    // rows per wave (R rows share each activation read: worth it from 2 tokens) and waves per row group (KS: the row's K
    // split over KS waves of the block, summed in LDS; more waves in flight for few rows / many tokens)
    // (1 token: 2 waves per row group on rows of >= 2048 halves, 1024 x 4096 18.0 -> 17.0 us, 64 x 4096 4.9 -> 4.1; 8192 x
    // 1024 rows lose with it, 25.7 -> 30.0. Several tokens: 1, more slices measured slower. Reading the activations from
    // global memory for several tokens instead of the LDS tile (one block per CU at 6 tokens) was slower on all but one
    // shape, so the tile stays.)
    // 3+ tokens: waves per block and rows per wave by row count (gfx906, 6 tokens, us: 1024 x 4096 8 waves x 4 rows 26.4
    // (4 x 4: 43.2), 512 x 4096 16 x 1 15.7 (4 x 4: 22.1), 256 / 64 x 4096 8 x 1 11.8 / 10.6, 8192 x 1024 4 x 2 45.2)
    const int r  = env_r ? env_r : N == 1 ? 1 : N == 2 ? 2 : M >= 4096 ? 2 : M >= 1024 ? 4 : 1;
    const int ks = env_ks ? env_ks : N == 1 && K >= 2048 ? 2 : 1;
    // waves per block (GGML_CUDA_GCN_F16_MV_NW): from 3 tokens the LDS tile (N*K halves) holds a CU to one or two blocks,
    // so 8..16 waves share it
    static const int env_nw = [] { const char * e = getenv("GGML_CUDA_GCN_F16_MV_NW"); return e ? atoi(e) : 0; }();
    const int nw = env_nw ? env_nw : N <= 2 || M >= 4096 ? 4 : M >= 1024 || M < 512 ? 8 : 16;
    const int64_t ys = N > 1 ? src1->nb[1] / sizeof(float) : 0;
    const int64_t ds = N > 1 ? dst->nb[1] / sizeof(float) : 0;
    const size_t smem = (size_t) N*K*sizeof(half);
    cudaStream_t stream = ctx.stream();
#define F16MV_GO(NT_, R_, KS_, NW_) do { const unsigned nb_ = (unsigned) ((M + ((NW_)/(KS_))*(R_) - 1)/(((NW_)/(KS_))*(R_))); \
        if (list) { gcn_f16_mv<NT_, R_, KS_, NW_, true><<<nb_, 64*(NW_), smem, stream>>>(nullptr, (const float *) src1->data, \
            nullptr, (int) K, (int) M, 0, ys, 0, *list); } \
        else { gcn_f16_mv<NT_, R_, KS_, NW_><<<nb_, 64*(NW_), smem, stream>>>((const char *) src0->data, (const float *) src1->data, \
            (float *) dst->data, (int) K, (int) M, src0->nb[1], ys, ds, ggml_cuda_f16mv_list{}); } } while (0)
#define F16MV(NT_, R_, KS_) do { if (nw == 16) { F16MV_GO(NT_, R_, KS_, 16); } else if (nw == 8) { F16MV_GO(NT_, R_, KS_, 8); } \
        else { F16MV_GO(NT_, R_, KS_, 4); } } while (0)
#define F16MV_KS(NT_, R_) switch (ks) { case 4: F16MV(NT_, R_, 4); break; case 2: F16MV(NT_, R_, 2); break; \
        default: F16MV(NT_, R_, 1); break; }
#define F16MV_R(NT_) switch (r) { case 4: F16MV_KS(NT_, 4) break; case 2: F16MV_KS(NT_, 2) break; default: F16MV_KS(NT_, 1) break; }
    switch (N) {
        case 1:  F16MV_R(1); break;
        case 2:  F16MV_R(2); break;
        case 3:  F16MV_R(3); break;
        case 4:  F16MV_R(4); break;
        case 5:  F16MV_R(5); break;
        case 6:  F16MV_R(6); break;
        case 7:  F16MV_R(7); break;
        default: F16MV_R(8); break;
    }
#undef F16MV_R
#undef F16MV_KS
#undef F16MV
#undef F16MV_GO
    CUDA_CHECK(cudaGetLastError());
}

// F32 weights with 65..8192 rows (e.g. GLM-5.3's router, 288 x 4096 per GPU, replicated): MMVF gave each row one block
// walking it in float2 steps (~300 GB/s). Same layout as gcn_f16_mv without the LDS tile and in full f32 (the router's
// rounding decides the experts): KS waves per row, each streaming a rotated slice of the row in 16-byte loads; the
// activations come from global memory (L1/L2) as float4. Fewer rows than 65 keep mul_mat_f32_rows_small_t.
template <int NT, int KS>
__launch_bounds__(256)
static __global__ void gcn_f32_mv(const float * __restrict__ W, const float * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride) {
    __shared__ float red[KS > 1 ? 4*NT : 1];
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int ks   = wave % KS;
    const int row  = blockIdx.x*(4/KS) + wave / KS;
    const bool live = row < M;
    const int nc   = K / 4; // 16-byte chunks
    const int ns   = (nc + 63) / 64;
    const int s_lo = ks*ns/KS;
    const int sn   = (ks + 1)*ns/KS - s_lo;
    const int rot  = sn > 0 ? row % sn : 0;
    const float4 * wr = (const float4 *) (W + (int64_t) min(row, M - 1)*w_stride);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
    if (live) {
#pragma unroll 4
        for (int s0 = 0; s0 < sn; ++s0) {
            const int c = (s_lo + (s0 + rot < sn ? s0 + rot : s0 + rot - sn))*64 + lane;
            if (c >= nc) {
                continue;
            }
            const float4 q = wr[c];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 a = *(const float4 *) (y + t*y_stride + 4*c);
                float s = acc[t];
                s = fmaf(q.x, a.x, s);
                s = fmaf(q.y, a.y, s);
                s = fmaf(q.z, a.z, s);
                s = fmaf(q.w, a.w, s);
                acc[t] = s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if constexpr (KS == 1) {
            if (lane == t && live) {
                dst[t*d_stride + row] = v;
            }
        } else if (lane == t) {
            red[wave*NT + t] = v;
        }
    }
    if constexpr (KS > 1) {
        __syncthreads();
        if (ks == 0 && lane < NT && live) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v += red[(wave + k)*NT + lane];
            }
            dst[lane*d_stride + row] = v;
        }
    }
}

bool ggml_cuda_gcn_f32_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int env    = [] { const char * e = getenv("GGML_CUDA_GCN_F32_MV"); return e ? atoi(e) : 1; }();
    static const int env_ks = [] { const char * e = getenv("GGML_CUDA_GCN_F32_MV_KS"); return e ? atoi(e) : 0; }();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < 1 || N > 4 || M <= 64 || M > 8192 || K % 4 != 0 || K < 1024 || K > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[0] != sizeof(float) || src0->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 ||
            src1->nb[0] != sizeof(float) || (N > 1 && src1->nb[1] % 16 != 0) || ((uintptr_t) src1->data) % 16 != 0 ||
            dst->nb[0] != sizeof(float) || ggml_cuda_is_f16_inplace(ctx, src1)) {
        return false;
    }
    const int ks = env_ks ? env_ks : M < 1024 ? 4 : M < 4096 ? 2 : 1;
    const int64_t ws = src0->nb[1] / sizeof(float);
    const int64_t ys = N > 1 ? src1->nb[1] / sizeof(float) : 0;
    const int64_t ds = N > 1 ? dst->nb[1] / sizeof(float) : 0;
    cudaStream_t stream = ctx.stream();
#define F32MV(NT_, KS_) gcn_f32_mv<NT_, KS_><<<(unsigned) ((M + 4/(KS_) - 1)/(4/(KS_))), 256, 0, stream>>>( \
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) K, (int) M, ws, ys, ds)
#define F32MV_KS(NT_) switch (ks) { case 4: F32MV(NT_, 4); break; case 2: F32MV(NT_, 2); break; default: F32MV(NT_, 1); break; }
    switch (N) {
        case 1:  F32MV_KS(1); break;
        case 2:  F32MV_KS(2); break;
        case 3:  F32MV_KS(3); break;
        default: F32MV_KS(4); break;
    }
#undef F32MV_KS
#undef F32MV
    CUDA_CHECK(cudaGetLastError());
    return true;
}
