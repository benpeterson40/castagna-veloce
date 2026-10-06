#include <algorithm>
#include <vector>
#include "moe-vec.cuh"
#include "mmid.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"

// Grouped MUL_MAT_ID for few tokens per expert (GCN): the tokens are sorted by expert (mm_ids_helper), each
// expert's token list is cut into tiles of up to NC tokens, and a block computes rows x NC outputs of one tile.
// A weight int is loaded and unpacked once and dotted against all NC tokens, so the weights are read once per
// tile instead of once per token (MMVQ) and no LDS tile is built (MMQ, which pays for 128-row tiles whose K loop
// is too short to amortize them at ~10 tokens per expert).

typedef float (*moe_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

template <ggml_type type> struct moe_vec_traits;
template <> struct moe_vec_traits<GGML_TYPE_Q4_0> { static constexpr int vdr = VDR_Q4_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q4_1> { static constexpr int vdr = VDR_Q4_1_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_1_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_0> { static constexpr int vdr = VDR_Q5_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_1> { static constexpr int vdr = VDR_Q5_1_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_1_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q8_0> { static constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q8_0_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q4_K> { static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q4_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q5_K> { static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q5_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q6_K> { static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q6_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_Q2_K> { static constexpr int vdr = VDR_Q2_K_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_q2_K_q8_1; };
template <> struct moe_vec_traits<GGML_TYPE_IQ2_XXS> { static constexpr int vdr = VDR_IQ2_XXS_Q8_1_MMVQ; static constexpr moe_vec_dot_t dot = vec_dot_iq2_xxs_q8_1; };

static constexpr int MOE_VEC_NWARPS = 4;

// one thread per value: gathers src1 rows in expert order and quantizes them to q8_1 (K padded with zeros)
static __global__ void moe_vec_quantize_gather(const float * __restrict__ x, const int32_t * __restrict__ ids_src1,
        block_q8_1 * __restrict__ y, const int64_t ne10, const int64_t ne10_padded, const int64_t s_col) {
    const int64_t i0  = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (i0 >= ne10_padded) {
        return;
    }
    const int64_t src_row = ids_src1 ? (int64_t) ids_src1[row] : row;
    const float xi = i0 < ne10 ? x[src_row*s_col + i0] : 0.0f;

    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    block_q8_1 * yb = y + (row*ne10_padded + i0) / QK8_1;
    yb->qs[i0 % QK8_1] = q;
    if (i0 % QK8_1 == 0) {
        yb->ds = make_half2(d, sum);
    }
}

// activations for the K-quant row-lane path: one scale per 256 values plus integer sums per 32, so a super-block's
// eight sub-block dot products combine with its 6-bit scales in integer math (one float step per super-block)
struct block_q8_moek {
    int8_t  qs[QK_K];
    float   d;
    int32_t bsum[QK_K/32];
};
static_assert(sizeof(block_q8_moek) == QK_K + 4 + 4*(QK_K/32), "unexpected block_q8_moek size");

// block of 256 threads per (super-block, row): gathers src1 rows in expert order and quantizes them to block_q8_moek
static __global__ void moe_vec_quantize_gather_q8k(const float * __restrict__ x, const int32_t * __restrict__ ids_src1,
        block_q8_moek * __restrict__ y, const int64_t ne10, const int64_t s_col) {
    const int64_t i0  = (int64_t) blockIdx.x*QK_K + threadIdx.x;
    const int64_t row = blockIdx.y;
    const float xi = i0 < ne10 ? x[(int64_t) ids_src1[row]*s_col + i0] : 0.0f;

    __shared__ float amax_w[QK_K/32];
    float amax = warp_reduce_max<32>(fabsf(xi));
    if (threadIdx.x % 32 == 0) {
        amax_w[threadIdx.x/32] = amax;
    }
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int w = 0; w < QK_K/32; ++w) {
        amax = fmaxf(amax, amax_w[w]);
    }
    const float  d = amax / 127.0f;
    const int    q = amax == 0.0f ? 0 : (int) roundf(xi / d);
    const int  sum = warp_reduce_sum<32>(q);

    block_q8_moek * yb = y + row*gridDim.x + blockIdx.x;
    yb->qs[threadIdx.x] = (int8_t) q;
    if (threadIdx.x % 32 == 0) {
        yb->bsum[threadIdx.x/32] = sum;
    }
    if (threadIdx.x == 0) {
        yb->d = d;
    }
}

// tile map: tile t covers tokens [col0, min(col0 + nc, end)) of one expert; tiles[0] of the header holds the count
static __global__ void moe_vec_tile_map(const int32_t * __restrict__ expert_bounds, int2 * __restrict__ tiles,
        int32_t * __restrict__ n_tiles, const int n_experts, const int nc) {
    extern __shared__ int moe_vec_scan[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // per-thread range of experts, serial count, then a block scan over threads
    const int per = (n_experts + nt - 1) / nt;
    const int e0  = min(tid*per, n_experts);
    const int e1  = min(e0 + per, n_experts);
    int cnt = 0;
    for (int e = e0; e < e1; ++e) {
        cnt += (expert_bounds[e + 1] - expert_bounds[e] + nc - 1) / nc;
    }
    moe_vec_scan[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < nt; off <<= 1) {
        const int v = tid >= off ? moe_vec_scan[tid - off] : 0;
        __syncthreads();
        moe_vec_scan[tid] += v;
        __syncthreads();
    }
    int t = moe_vec_scan[tid] - cnt;
    for (int e = e0; e < e1; ++e) {
        const int beg = expert_bounds[e];
        const int end = expert_bounds[e + 1];
        for (int c = beg; c < end; c += nc) {
            tiles[t++] = make_int2(e, c);
        }
    }
    if (tid == nt - 1) {
        *n_tiles = moe_vec_scan[tid];
    }
}

// T lanes share one row (T divides the warp); the block covers rows_per_block = nwarps*warp_size/T rows
template <ggml_type type, int NC, int T>
__launch_bounds__(MOE_VEC_NWARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void moe_vec_q(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, const int2 * __restrict__ tiles,
        const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int64_t stride_row_x, const int64_t stride_expert_x,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = moe_vec_traits<type>::vdr;
    constexpr moe_vec_dot_t vec_dot = moe_vec_traits<type>::dot;
    constexpr int rows_per_block = MOE_VEC_NWARPS*warp_size/T;

    const int tile = blockIdx.y;
    if (tile >= *n_tiles) {
        return;
    }
    const int2 te    = tiles[tile];
    const int expert = te.x;
    const int col0   = te.y;
    const int ncols  = min(NC, expert_bounds[expert + 1] - col0);

    const int tid  = threadIdx.y*warp_size + threadIdx.x;
    const int lane = tid % T;
    const int row  = blockIdx.x*rows_per_block + tid / T;
    if (row >= nrows_x) {
        return; // whole T-lane groups exit together
    }

    const char * x = vx + expert*stride_expert_x;
    const int kbx_row = row*stride_row_x;

    const block_q8_1 * y[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        y[j] = vy + (col0 + min(j, ncols - 1))*stride_col_y; // idle columns repeat the last token
    }

    float tmp[NC] = {0.0f};

    const int blocks_per_row = ncols_x / qk;
    constexpr int items_per_block = qi/vdr;
    const int items = blocks_per_row*items_per_block;

    for (int it = lane; it < items; it += T) {
        const int kbx = it / items_per_block;
        const int kqs = vdr*(it % items_per_block);
        const int kby = kbx*(qk/QK8_1);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j < ncols) { // uniform: skips the idle columns of a partial tile
                tmp[j] += vec_dot(x, y[j] + kby, kbx_row + kbx, kqs);
            }
        }
    }

    if constexpr (T > 1) {
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            tmp[j] = warp_reduce_sum<T>(tmp[j]);
        }
    }

#pragma unroll
    for (int j = 0; j < NC; ++j) {
        if (lane == j % T && j < ncols) {
            const int d = ids_dst[col0 + j];
            dst[(d / n_expert_used)*s2_dst + (d % n_expert_used)*s1_dst + row] = tmp[j];
        }
    }
}


// Row-lane variant (GCN): one lane per weight row, whole quant blocks pulled with 16-byte loads and unpacked once in
// registers, then dotted against each token of the tile. The tile's activations are the same for every lane, so they
// come through scalar loads and cost no vector memory traffic; the only vector loads are the weights, read once.
// Idle columns of a partial tile are skipped with a uniform branch instead of being computed as padding.
static __device__ __forceinline__ float2 moe_rl_h2f(const int v) {
    const half2 h = *reinterpret_cast<const half2 *>(&v);
    return __half22float2(h);
}

template <int J>
static __device__ __forceinline__ void moe_rl_q5_1_block(const int * w, const block_q8_1 * __restrict__ y, const int64_t stride_col_y,
        const int kb, const int ncols, float * acc) {
    const float2 dm = moe_rl_h2f(w[0]);
    const int    qh = w[1];
    int v[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vl = w[2 + i];
        const int vh = qh >> (4*i);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0    |= (vh <<  4) & 0x00000010;
        vi0    |= (vh << 11) & 0x00001000;
        vi0    |= (vh << 18) & 0x00100000;
        vi0    |= (vh << 25) & 0x10000000;
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1    |= (vh >> 12) & 0x00000010;
        vi1    |= (vh >>  5) & 0x00001000;
        vi1    |= (vh <<  2) & 0x00100000;
        vi1    |= (vh <<  9) & 0x10000000;
        v[i] = vi0; v[4 + i] = vi1;
    }
#pragma unroll
    for (int j = 0; j < J; ++j) {
        if (j >= ncols) {
            break;
        }
        {
            const block_q8_1 * yb = y + j*stride_col_y + kb;
            const int * yq = (const int *) yb->qs;
            int sumi = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sumi = ggml_cuda_dp4a(v[i], yq[i], sumi);
            }
            const float2 ds = __half22float2(yb->ds);
            acc[j] += sumi*(dm.x*ds.x) + dm.y*ds.y;
        }
    }
}

// decode without the expert sort: ids of the MUL_MAT_ID (row stride s1), activation row = slot / y_div
struct moe_direct_t {
    const int32_t * ids = nullptr;
    int64_t s1 = 0;
    int y_div = 1;
    // gate/up pairs: SwiGLU-clamp limit (GLM-5-Next: 10); INFINITY = plain SwiGLU (same arithmetic as before)
    float glu_limit = INFINITY;
};

// silu(min(g, limit)) * clamp(u, -limit, limit), as ggml_cuda_op_swiglu_clamp_single; with limit INFINITY exactly the plain
// SwiGLU expression these kernels used
static __device__ __forceinline__ float moe_glu(float g, float u, const float limit) {
    g = fminf(g, limit);
    u = fmaxf(fminf(u, limit), -limit);
    return g/(1.0f + expf(-g))*u;
}

#ifndef MOE_DEC_KS
#define MOE_DEC_KS 8
#endif
#ifndef MOE_DEC_KS_DEFAULT
#define MOE_DEC_KS_DEFAULT 4
#endif

// decode (few tokens): the row-lane grid alone has too few waves to cover the memory latency, so K is also split over
// blockDim.y waves of the block (steps = the kernel's K loop iterations) and the partial sums are reduced through LDS
static __device__ __forceinline__ void moe_rl_krange(const int nsteps, int & s0, int & s1) {
    s0 = (nsteps*(int) threadIdx.y)/(int) blockDim.y;
    s1 = (nsteps*((int) threadIdx.y + 1))/(int) blockDim.y;
}

template <int J, int KS>
static __device__ __forceinline__ void moe_rl_kreduce(float * acc) {
    if constexpr (KS > 1) {
        __shared__ float red[KS - 1][J][64];
        if (blockDim.y > 1) {
            __syncthreads(); // (the previous chunk's reads are done)
            if (threadIdx.y > 0) {
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    red[threadIdx.y - 1][j][threadIdx.x] = acc[j];
                }
            }
            __syncthreads();
            if (threadIdx.y == 0) {
                for (int sl = 1; sl < (int) blockDim.y; ++sl) {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        acc[j] += red[sl - 1][j][threadIdx.x];
                    }
                }
            }
        }
    }
}

// tok_map (gate/up with one activation row per token): column j of the tile reads activation row tok_map[col0 + j]
// (activations quantized once per token instead of once per token-expert slot)
template <ggml_type type, int J, bool Q8K = false, int KS = 1>
__launch_bounds__(64*KS)
static __global__ void moe_rowlane(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, const int2 * __restrict__ tiles,
        const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int64_t stride_row_b, const int64_t stride_expert_b,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst,
        const int32_t * __restrict__ tok_map = nullptr, const int glu_epi = 0, const int tile_cols = J,
        const moe_direct_t direct = {}) {
    int expert, col_beg, col_tile_end, dslot = -1;
    if (direct.ids) {
        // decode: one block per slot straight from ids (no expert sort / tile map), activation row slot / y_div
        const int slot = blockIdx.y;
        expert       = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
        col_beg      = slot / direct.y_div;
        col_tile_end = col_beg + 1;
        dslot        = slot;
    } else {
        const int tile = blockIdx.y;
        if (tile >= *n_tiles) {
            return;
        }
        const int2 te = tiles[tile];
        expert  = te.x;
        col_beg = te.y;
        // a tile covers up to tile_cols tokens of one expert, processed J at a time by this block: the later chunks re-read
        // weight rows the block just streamed (L2) instead of another block re-reading them from DRAM later
        col_tile_end = min(te.y + tile_cols, expert_bounds[expert + 1]);
    }
    for (int col0 = col_beg; col0 < col_tile_end; col0 += J) {
    const int ncols  = min(J, col_tile_end - col0);

    const int  row   = blockIdx.x*64 + threadIdx.x;
    const bool valid = row < nrows_x;
    const char * xr  = vx + expert*stride_expert_b + (valid ? row : nrows_x - 1)*stride_row_b;

    const block_q8_1 * y = vy + col0*stride_col_y; // column j of the tile at y + j*stride_col_y (only j < ncols is read)
    float acc[J] = {0.0f};

    if constexpr (type == GGML_TYPE_Q5_1) {
        // 4 blocks (96 bytes, 16-byte aligned: the row stride is a multiple of 16) per step
        const int nb = ncols_x / QK5_1;
        int st0 = 0, st1 = nb/4;
        if constexpr (KS > 1) { moe_rl_krange(nb/4, st0, st1); }
        for (int kb = 4*st0; kb < 4*st1; kb += 4) {
            int w[24];
            const int4 * p = (const int4 *) (xr + kb*sizeof(block_q5_1));
#pragma unroll
            for (int i = 0; i < 6; ++i) {
                const int4 t = p[i];
                w[4*i + 0] = t.x; w[4*i + 1] = t.y; w[4*i + 2] = t.z; w[4*i + 3] = t.w;
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                moe_rl_q5_1_block<J>(w + 6*b, y, stride_col_y, kb + b, ncols, acc);
            }
        }
        // K % 128 != 0 (e.g. a 320-wide expert slice under -sm tensor): the last K slice takes the remaining 1-3 blocks,
        // one 24-byte (8-byte aligned) block at a time
        if (st1 == nb/4) {
            int kb0 = 4*(nb/4);
            if (nb - kb0 >= 2) {
                // two blocks (48 bytes, 16-byte aligned: kb0 is a multiple of 4) in one step
                int w[12];
                const int4 * p = (const int4 *) (xr + kb0*sizeof(block_q5_1));
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    const int4 t = p[i];
                    w[4*i + 0] = t.x; w[4*i + 1] = t.y; w[4*i + 2] = t.z; w[4*i + 3] = t.w;
                }
                moe_rl_q5_1_block<J>(w,     y, stride_col_y, kb0,     ncols, acc);
                moe_rl_q5_1_block<J>(w + 6, y, stride_col_y, kb0 + 1, ncols, acc);
                kb0 += 2;
            }
            for (int kb = kb0; kb < nb; ++kb) {
                const int2 * p = (const int2 *) (xr + kb*sizeof(block_q5_1));
                const int2 t0 = p[0], t1 = p[1], t2 = p[2];
                const int w[6] = {t0.x, t0.y, t1.x, t1.y, t2.x, t2.y};
                moe_rl_q5_1_block<J>(w, y, stride_col_y, kb, ncols, acc);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // 4 blocks (136 bytes, 8-byte aligned since K % 128 == 0) per step; odd blocks start dword-aligned at byte 2,
        // even blocks need their quants shifted by 2 bytes
        const int nb = ncols_x / QK8_0;
        int st0 = 0, st1 = nb/4;
        if constexpr (KS > 1) { moe_rl_krange(nb/4, st0, st1); }
        for (int kb = 4*st0; kb < 4*st1; kb += 4) {
            int w[34];
            const int2 * p = (const int2 *) (xr + kb*sizeof(block_q8_0));
#pragma unroll
            for (int i = 0; i < 17; ++i) {
                const int2 t = p[i];
                w[2*i + 0] = t.x; w[2*i + 1] = t.y;
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte0 = 34*b;             // d at byte0, quants at byte0 + 2
                const int dw    = byte0 / 4;
                const float d   = __half2float(*reinterpret_cast<const half *>(
                    reinterpret_cast<const char *>(w) + byte0));
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if ((byte0 + 2) % 4 == 0) {
                        v[i] = w[(byte0 + 2)/4 + i];
                    } else {
                        v[i] = __builtin_amdgcn_alignbyte(w[dw + i + 1], w[dw + i], 2);
                    }
                }
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    if (j >= ncols) {
                        break;
                    }
                    {
                        const block_q8_1 * yb = y + j*stride_col_y + kb + b;
                        const int * yq = (const int *) yb->qs;
                        int sumi = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sumi = ggml_cuda_dp4a(v[i], yq[i], sumi);
                        }
                        acc[j] += sumi*(d*__low2float(yb->ds));
                    }
                }
            }
        }
    } else {
        static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K, "unsupported type");
        // q5_K = q4_K layout plus a 32-byte plane of 5th bits between the scales and the quants
        constexpr bool Q5 = type == GGML_TYPE_Q5_K;
        constexpr int  NI = Q5 ? 11 : 9;  // 16-byte loads per super-block
        constexpr int  QS = Q5 ? 12 : 4;  // dword offset of the quants
        constexpr size_t SB_BYTES = Q5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
        const int nsb = ncols_x / QK_K;
        int sb0 = 0, sb1 = nsb;
        if constexpr (KS > 1) { moe_rl_krange(nsb, sb0, sb1); }
#ifndef MOE_RL_JIT
#define MOE_RL_JIT 1
#endif
        if constexpr (!Q8K && MOE_RL_JIT && J <= 8) { // (16-wide tiles: the whole-super-block load below is faster)
            // quants loaded per sub-block pair right before use and scales decoded on the fly: fewer live registers,
            // so more waves hide the load latency (the kernel is latency-bound)
            // activation row of every tile column, resolved once (full tiles run the column loop straight through so the
            // scalar loads of consecutive columns issue together; the clamp keeps the unused entries of partial tiles valid)
            const block_q8_1 * ycol[J];
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int jj = min(j, ncols - 1);
                ycol[j] = tok_map ? vy + (int64_t) tok_map[col0 + jj]*stride_col_y : y + jj*stride_col_y;
            }
            for (int sb = sb0; sb < sb1; ++sb) {
                const int4 * p = (const int4 *) (xr + sb*SB_BYTES);
                const int4 hd = p[0];
                const float2 dm = moe_rl_h2f(hd.x);
                const int scw[3] = {hd.y, hd.z, hd.w};
                const uint8_t * sc8 = (const uint8_t *) scw;
                int qh[8];
                if constexpr (Q5) {
                    const int4 h0 = p[1], h1 = p[2];
                    qh[0] = h0.x; qh[1] = h0.y; qh[2] = h0.z; qh[3] = h0.w;
                    qh[4] = h1.x; qh[5] = h1.y; qh[6] = h1.z; qh[7] = h1.w;
                }
#pragma unroll
                for (int pr = 0; pr < 4; ++pr) {
                    const int4 qa = p[QS/4 + 2*pr], qb = p[QS/4 + 2*pr + 1];
                    const int qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                    int lo[8], hi[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        lo[i] = qw[i] & 0x0F0F0F0F;
                        hi[i] = (qw[i] >> 4) & 0x0F0F0F0F;
                        if constexpr (Q5) {
                            lo[i] |= ((qh[i] >> (2*pr    )) & 0x01010101) << 4;
                            hi[i] |= ((qh[i] >> (2*pr + 1)) & 0x01010101) << 4;
                        }
                    }
                    float dsc2[2], dmn2[2];
#pragma unroll
                    for (int t = 0; t < 2; ++t) {
                        const int s_ = 2*pr + t;
                        int scv, mv;
                        if (s_ < 4) {
                            scv = sc8[s_] & 63;
                            mv  = sc8[s_ + 4] & 63;
                        } else {
                            scv = (sc8[s_ + 4] & 0xF) | ((sc8[s_ - 4] >> 6) << 4);
                            mv  = (sc8[s_ + 4] >>  4) | ((sc8[s_    ] >> 6) << 4);
                        }
                        dsc2[t] = dm.x*scv;
                        dmn2[t] = dm.y*mv;
                    }
                    auto col = [&](const int j) {
                        const block_q8_1 * y0 = ycol[j] + sb*8 + 2*pr;
                        const block_q8_1 * y1 = y0 + 1;
                        const int * q0 = (const int *) y0->qs;
                        const int * q1 = (const int *) y1->qs;
                        int s0 = 0, s1 = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            s0 = ggml_cuda_dp4a(lo[i], q0[i], s0);
                            s1 = ggml_cuda_dp4a(hi[i], q1[i], s1);
                        }
                        const float2 ds0 = __half22float2(y0->ds);
                        const float2 ds1 = __half22float2(y1->ds);
                        acc[j] += dsc2[0]*ds0.x*s0 - dmn2[0]*ds0.y + dsc2[1]*ds1.x*s1 - dmn2[1]*ds1.y;
                    };
                    if (ncols == J) {
#pragma unroll
                        for (int j = 0; j < J; ++j) {
                            col(j);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < J; ++j) {
                            if (j >= ncols) {
                                break;
                            }
                            col(j);
                        }
                    }
                }
            }
        } else
        for (int sb = sb0; sb < sb1; ++sb) {
            int w[NI*4];
            const int4 * p = (const int4 *) (xr + sb*SB_BYTES);
#pragma unroll
            for (int i = 0; i < NI; ++i) {
                const int4 t = p[i];
                w[4*i + 0] = t.x; w[4*i + 1] = t.y; w[4*i + 2] = t.z; w[4*i + 3] = t.w;
            }
            const float2 dm = moe_rl_h2f(w[0]);
            const uint8_t * sc8 = (const uint8_t *) (w + 1);
            if constexpr (Q8K) {
                int sci[8], mi[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    if (s < 4) {
                        sci[s] = sc8[s] & 63;
                        mi[s]  = sc8[s + 4] & 63;
                    } else {
                        sci[s] = (sc8[s + 4] & 0xF) | ((sc8[s - 4] >> 6) << 4);
                        mi[s]  = (sc8[s + 4] >>  4) | ((sc8[s    ] >> 6) << 4);
                    }
                }
                // y is block_q8_moek here, one block per super-block, stride_col_y in blocks
                const block_q8_moek * yk = (const block_q8_moek *) vy + col0*stride_col_y;
                int isum[J];
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    isum[j] = 0;
                }
#pragma unroll
                for (int pr = 0; pr < 4; ++pr) {
                    int lo[8], hi[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int q = w[QS + 8*pr + i];
                        lo[i] = q & 0x0F0F0F0F;
                        hi[i] = (q >> 4) & 0x0F0F0F0F;
                        if constexpr (Q5) {
                            const int qh = w[4 + i];
                            lo[i] |= ((qh >> (2*pr    )) & 0x01010101) << 4;
                            hi[i] |= ((qh >> (2*pr + 1)) & 0x01010101) << 4;
                        }
                    }
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        if (j >= ncols) {
                            break;
                        }
                        const int * q = (const int *) yk[j*stride_col_y + sb].qs + 16*pr;
                        int s0 = 0, s1 = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            s0 = ggml_cuda_dp4a(lo[i], q[i],     s0);
                            s1 = ggml_cuda_dp4a(hi[i], q[8 + i], s1);
                        }
                        isum[j] += sci[2*pr]*s0 + sci[2*pr + 1]*s1;
                    }
                }
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    if (j >= ncols) {
                        break;
                    }
                    const block_q8_moek * b = yk + j*stride_col_y + sb;
                    int msum = 0;
#pragma unroll
                    for (int s = 0; s < 8; ++s) {
                        msum += mi[s]*b->bsum[s];
                    }
                    acc[j] += b->d*(dm.x*isum[j] - dm.y*msum);
                }
                continue;
            }
            float dsc[8], dmn[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                int scv, mv;
                if (s < 4) {
                    scv = sc8[s] & 63;
                    mv  = sc8[s + 4] & 63;
                } else {
                    scv = (sc8[s + 4] & 0xF) | ((sc8[s - 4] >> 6) << 4);
                    mv  = (sc8[s + 4] >>  4) | ((sc8[s    ] >> 6) << 4);
                }
                dsc[s] = dm.x*scv;
                dmn[s] = dm.y*mv;
            }
#pragma unroll
            for (int pr = 0; pr < 4; ++pr) {
                int lo[8], hi[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int q = w[QS + 8*pr + i];
                    lo[i] = q & 0x0F0F0F0F;
                    hi[i] = (q >> 4) & 0x0F0F0F0F;
                    if constexpr (Q5) {
                        const int qh = w[4 + i]; // 5th bits of values 4i..4i+3 of every pair, bit 2*pr / 2*pr+1
                        lo[i] |= ((qh >> (2*pr    )) & 0x01010101) << 4;
                        hi[i] |= ((qh >> (2*pr + 1)) & 0x01010101) << 4;
                    }
                }
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    if (j >= ncols) {
                        break;
                    }
                    {
                        const block_q8_1 * y0 = (tok_map ? vy + (int64_t) tok_map[col0 + j]*stride_col_y : y + j*stride_col_y) + sb*8 + 2*pr;
                        const block_q8_1 * y1 = y0 + 1;
                        const int * q0 = (const int *) y0->qs;
                        const int * q1 = (const int *) y1->qs;
                        int s0 = 0, s1 = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            s0 = ggml_cuda_dp4a(lo[i], q0[i], s0);
                            s1 = ggml_cuda_dp4a(hi[i], q1[i], s1);
                        }
                        const float2 ds0 = __half22float2(y0->ds);
                        const float2 ds1 = __half22float2(y1->ds);
                        acc[j] += dsc[2*pr]*ds0.x*s0 - dmn[2*pr]*ds0.y + dsc[2*pr + 1]*ds1.x*s1 - dmn[2*pr + 1]*ds1.y;
                    }
                }
            }
        }
    }

    moe_rl_kreduce<J, KS>(acc);
    if (valid && (KS == 1 || threadIdx.y == 0)) {
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (j < ncols) {
                const int d = dslot >= 0 ? dslot : ids_dst[col0 + j];
                float * o = dst + (d / n_expert_used)*s2_dst + (d % n_expert_used)*s1_dst + row;
                if (glu_epi) {
                    // fused SwiGLU: o holds the gate projection written by the previous launch of this pair
                    *o = moe_glu(*o, acc[j], direct.glu_limit);
                } else {
                    *o = acc[j];
                }
            }
        }
    }
    } // chunk loop
}

// gate/up pair + SwiGLU in one row-lane pass (K-quants, tok_map activations): each lane computes the gate and the up row
// of the same output row, so every activation load feeds both projections and glu is written once
template <ggml_type type_g, ggml_type type_u, int J, int KS = 1>
__launch_bounds__(64*KS)
static __global__ void moe_rowlane_glu(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, const int2 * __restrict__ tiles,
        const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ ids_dst,
        float * __restrict__ dst, const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp,
        const int64_t su_row, const int64_t su_exp, const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst,
        const int64_t s2_dst, const int32_t * __restrict__ tok_map, const moe_direct_t direct = {}) {
    int expert, col0, ncols, dslot = -1;
    if (direct.ids) { // (see moe_rowlane)
        const int slot = blockIdx.y;
        expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
        col0   = slot / direct.y_div;
        ncols  = 1;
        dslot  = slot;
    } else {
        const int tile = blockIdx.y;
        if (tile >= *n_tiles) {
            return;
        }
        const int2 te = tiles[tile];
        expert = te.x;
        col0   = te.y;
        ncols  = min(J, expert_bounds[expert + 1] - col0);
    }

    const int  row   = blockIdx.x*64 + threadIdx.x;
    const bool valid = row < nrows_x;
    const int  rr    = valid ? row : nrows_x - 1;
    const char * xg  = vg + expert*sg_exp + rr*sg_row;
    const char * xu  = vu + expert*su_exp + rr*su_row;

    const block_q8_1 * ycol[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int jj = min(j, ncols - 1); // (clamped: never read past ncols)
        ycol[j] = vy + (tok_map ? (int64_t) tok_map[col0 + jj] : (int64_t) (col0 + jj))*stride_col_y;
    }
    float ag[J] = {0.0f};
    float au[J] = {0.0f};

    constexpr bool G5 = type_g == GGML_TYPE_Q5_K;
    constexpr bool U5 = type_u == GGML_TYPE_Q5_K;
    constexpr size_t GB = G5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    constexpr size_t UB = U5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    const int nsb = ncols_x / QK_K;
    int sb0 = 0, sb1 = nsb;
    if constexpr (KS > 1) { moe_rl_krange(nsb, sb0, sb1); }

    // one sub-block pair of one matrix: quants (lo/hi) and the per-pair scale/min products
    struct half_t { int lo[8], hi[8]; float dsc[2], dmn[2]; };
    auto decode = [&](const int4 * p, const bool q5, const int pr, half_t & h) {
        const int4 hd = p[0];
        const float2 dm = moe_rl_h2f(hd.x);
        const int scw[3] = {hd.y, hd.z, hd.w};
        const uint8_t * sc8 = (const uint8_t *) scw;
        const int qs = q5 ? 3 : 1; // int4 offset of the quants
        const int4 qa = p[qs + 2*pr], qb = p[qs + 2*pr + 1];
        const int qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        int qh[8] = {0};
        if (q5) {
            const int4 h0 = p[1], h1 = p[2];
            qh[0] = h0.x; qh[1] = h0.y; qh[2] = h0.z; qh[3] = h0.w;
            qh[4] = h1.x; qh[5] = h1.y; qh[6] = h1.z; qh[7] = h1.w;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            h.lo[i] = qw[i] & 0x0F0F0F0F;
            h.hi[i] = (qw[i] >> 4) & 0x0F0F0F0F;
            if (q5) {
                h.lo[i] |= ((qh[i] >> (2*pr    )) & 0x01010101) << 4;
                h.hi[i] |= ((qh[i] >> (2*pr + 1)) & 0x01010101) << 4;
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int s_ = 2*pr + t;
            int scv, mv;
            if (s_ < 4) {
                scv = sc8[s_] & 63;
                mv  = sc8[s_ + 4] & 63;
            } else {
                scv = (sc8[s_ + 4] & 0xF) | ((sc8[s_ - 4] >> 6) << 4);
                mv  = (sc8[s_ + 4] >>  4) | ((sc8[s_    ] >> 6) << 4);
            }
            h.dsc[t] = dm.x*scv;
            h.dmn[t] = dm.y*mv;
        }
    };

    for (int sb = sb0; sb < sb1; ++sb) {
        const int4 * pg = (const int4 *) (xg + sb*GB);
        const int4 * pu = (const int4 *) (xu + sb*UB);
#pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
            half_t g, u;
            decode(pg, G5, pr, g);
            decode(pu, U5, pr, u);
            auto col = [&](const int j) {
                const block_q8_1 * y0 = ycol[j] + sb*8 + 2*pr;
                const block_q8_1 * y1 = y0 + 1;
                const int * q0 = (const int *) y0->qs;
                const int * q1 = (const int *) y1->qs;
                int g0 = 0, g1 = 0, u0 = 0, u1 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    g0 = ggml_cuda_dp4a(g.lo[i], q0[i], g0);
                    g1 = ggml_cuda_dp4a(g.hi[i], q1[i], g1);
                    u0 = ggml_cuda_dp4a(u.lo[i], q0[i], u0);
                    u1 = ggml_cuda_dp4a(u.hi[i], q1[i], u1);
                }
                const float2 ds0 = __half22float2(y0->ds);
                const float2 ds1 = __half22float2(y1->ds);
                ag[j] += g.dsc[0]*ds0.x*g0 - g.dmn[0]*ds0.y + g.dsc[1]*ds1.x*g1 - g.dmn[1]*ds1.y;
                au[j] += u.dsc[0]*ds0.x*u0 - u.dmn[0]*ds0.y + u.dsc[1]*ds1.x*u1 - u.dmn[1]*ds1.y;
            };
            // full tiles run straight through (the loads of all columns issue together); partial ones stop at ncols
            if (ncols == J) {
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    col(j);
                }
            } else {
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    if (j >= ncols) {
                        break;
                    }
                    col(j);
                }
            }
        }
    }

    moe_rl_kreduce<J, KS>(ag);
    moe_rl_kreduce<J, KS>(au);
    if (valid && (KS == 1 || threadIdx.y == 0)) {
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (j < ncols) {
                const int d = dslot >= 0 ? dslot : ids_dst[col0 + j];
                dst[(d / n_expert_used)*s2_dst + (d % n_expert_used)*s1_dst + row] = moe_glu(ag[j], au[j], direct.glu_limit);
            }
        }
    }
}

// Decode (one token per slot, direct ids): coalesced K-quant MoE kernel. 32 lanes per output row read the row's
// quant chunks as consecutive 16-byte pieces (a row-lane layout, one lane per weight row, touches 64 scattered rows
// per load and reaches only ~60% of the DRAM bandwidth for these few, randomly placed expert matrices). Work item
// = one 16-byte quant chunk (half of a sub-block pair: 16 values of sub-block 2p and 16 of 2p+1); the super-block
// header (and the q5_K high-bit chunk) are small broadcast loads. GLU: the same lanes also run the up row and store
// silu(gate)*up.
template <bool Q5>
static __device__ __forceinline__ float moe_coal_kq_item(const char * __restrict__ row, const block_q8_1 * __restrict__ y, const int it) {
    constexpr int SB = Q5 ? sizeof(block_q5_K) : sizeof(block_q4_K);
    constexpr int QS = Q5 ? 48 : 16; // byte offset of the quants in the super-block
    const int sb = it >> 3;
    const int q  = it & 7;
    const int p  = q >> 1;
    const int h  = q & 1;
    const char * b = row + sb*SB;
    const int4 hd = *(const int4 *) b;
    const int4 qw = *(const int4 *) (b + QS + 16*q);
    const int qa[4] = {qw.x, qw.y, qw.z, qw.w};
    int lo[4], hi[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo[i] = qa[i] & 0x0F0F0F0F;
        hi[i] = (qa[i] >> 4) & 0x0F0F0F0F;
    }
    if constexpr (Q5) {
        const int4 qh4 = *(const int4 *) (b + 16 + 16*h); // 5th bits of values 16h..16h+15 of every sub-block
        const int qh[4] = {qh4.x, qh4.y, qh4.z, qh4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] |= ((qh[i] >> (2*p    )) & 0x01010101) << 4;
            hi[i] |= ((qh[i] >> (2*p + 1)) & 0x01010101) << 4;
        }
    }
    const block_q8_1 * y0 = y + sb*8 + 2*p;
    const int4 a0 = *(const int4 *) (y0->qs + 16*h);
    const int4 a1 = *(const int4 *) ((y0 + 1)->qs + 16*h);
    const float d0 = __low2float(y0->ds);
    const float d1 = __low2float((y0 + 1)->ds);
    int s0 = 0, s1 = 0, m0 = 0, m1 = 0;
    s0 = ggml_cuda_dp4a(lo[0], a0.x, s0); s0 = ggml_cuda_dp4a(lo[1], a0.y, s0);
    s0 = ggml_cuda_dp4a(lo[2], a0.z, s0); s0 = ggml_cuda_dp4a(lo[3], a0.w, s0);
    s1 = ggml_cuda_dp4a(hi[0], a1.x, s1); s1 = ggml_cuda_dp4a(hi[1], a1.y, s1);
    s1 = ggml_cuda_dp4a(hi[2], a1.z, s1); s1 = ggml_cuda_dp4a(hi[3], a1.w, s1);
    m0 = ggml_cuda_dp4a(0x01010101, a0.x, m0); m0 = ggml_cuda_dp4a(0x01010101, a0.y, m0);
    m0 = ggml_cuda_dp4a(0x01010101, a0.z, m0); m0 = ggml_cuda_dp4a(0x01010101, a0.w, m0);
    m1 = ggml_cuda_dp4a(0x01010101, a1.x, m1); m1 = ggml_cuda_dp4a(0x01010101, a1.y, m1);
    m1 = ggml_cuda_dp4a(0x01010101, a1.z, m1); m1 = ggml_cuda_dp4a(0x01010101, a1.w, m1);
    // 6-bit scales/mins of sub-blocks 2p and 2p+1 (p is not a compile-time constant: bytes by shifts, no local array)
    const float2 dm = moe_rl_h2f(hd.x);
    auto sc8 = [&](const int k) {
        const int w = k < 4 ? hd.y : (k < 8 ? hd.z : hd.w);
        return (w >> (8*(k & 3))) & 0xFF;
    };
    float r = 0.0f;
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int s_ = 2*p + t;
        int scv, mv;
        if (s_ < 4) {
            scv = sc8(s_) & 63;
            mv  = sc8(s_ + 4) & 63;
        } else {
            scv = (sc8(s_ + 4) & 0xF) | ((sc8(s_ - 4) >> 6) << 4);
            mv  = (sc8(s_ + 4) >>  4) | ((sc8(s_    ) >> 6) << 4);
        }
        const float da = t == 0 ? d0 : d1;
        const int   sv = t == 0 ? s0 : s1;
        const int   mm = t == 0 ? m0 : m1;
        r += da*(dm.x*scv*sv - dm.y*mv*mm);
    }
    return r;
}

#ifndef MOE_COAL_UNROLL
#define MOE_COAL_UNROLL 1
#endif
template <bool G5, bool U5, bool GLU, int LPR>
__launch_bounds__(256)
static __global__ void moe_coal_kq(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    const int slot   = blockIdx.y;
    const int expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
    const block_q8_1 * y = vy + (int64_t) (slot / direct.y_div)*stride_col_y;
    const int sub = threadIdx.x % LPR; // LPR lanes per row
    const int row = blockIdx.x*(256/LPR) + threadIdx.x / LPR;
    if (row >= nrows_x) {
        return; // whole 32-lane groups
    }
    const char * rg = vg + expert*sg_exp + row*sg_row;
    const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;
    const int nit = (ncols_x / QK_K)*8;
    float ag = 0.0f, au = 0.0f;
    // (unrolled so the chunk loads of consecutive items are in flight together)
    for (int it0 = 0; it0 < nit; it0 += LPR*MOE_COAL_UNROLL) {
#pragma unroll
        for (int u = 0; u < MOE_COAL_UNROLL; ++u) {
            const int it = it0 + LPR*u + sub;
            if (it < nit) {
                ag += moe_coal_kq_item<G5>(rg, y, it);
                if constexpr (GLU) {
                    au += moe_coal_kq_item<U5>(ru, y, it);
                }
            }
        }
    }
#pragma unroll
    for (int off = LPR/2; off > 0; off >>= 1) {
        ag += __shfl_xor(ag, off, LPR);
        if constexpr (GLU) {
            au += __shfl_xor(au, off, LPR);
        }
    }
    if (sub == 0) {
        float * o = dst + (slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row;
        *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
    }
}

// Long-span variant: the slot's quantized activation row is staged in shared memory once (in moe_coal_kq every
// 16-byte weight chunk came with 32 bytes of activation loads from global memory), and each block walks RG row
// groups of 256/LPR rows, so it streams one long stretch of the expert matrices (gfx906, 20 random 0.92 MB
// matrices: long spans per block reach ~740 GB/s, 16-row blocks ~500). LDS: ncols_x/QK8_1 blocks of block_q8_1.
template <bool G5, bool U5, bool GLU, int LPR, int RG>
__launch_bounds__(256)
static __global__ void moe_coal_kq_ls(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    extern __shared__ int4 ys_raw[];
    block_q8_1 * ys = (block_q8_1 *) ys_raw;
    const int slot   = blockIdx.y;
    const int expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
    const int nyb = ncols_x / QK8_1;
    const int sub = threadIdx.x % LPR;
    const int nit = (ncols_x / QK_K)*8;
    {
        const block_q8_1 * y = vy + (int64_t) (slot / direct.y_div)*stride_col_y;
        // 36-byte blocks copied as ints
        const int * src = (const int *) y;
        int * d = (int *) ys;
        for (int i = threadIdx.x; i < nyb*(int) (sizeof(block_q8_1)/4); i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();
#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*(256/LPR) + threadIdx.x / LPR;
        if (row >= nrows_x) {
            break; // whole lane groups; nothing after the loop needs the block
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;
        float ag = 0.0f, au = 0.0f;
        for (int it = sub; it < nit; it += LPR) {
            ag += moe_coal_kq_item<G5>(rg, ys, it);
            if constexpr (GLU) {
                au += moe_coal_kq_item<U5>(ru, ys, it);
            }
        }
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            ag += __shfl_xor(ag, off, LPR);
            if constexpr (GLU) {
                au += __shfl_xor(au, off, LPR);
            }
        }
        if (sub == 0) {
            float * o = dst + (slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row;
            *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
        }
    }
}

// Several tokens per MoE call (MTP verify): tokens often share experts (real text, 4-token verify: ~26 distinct of 40
// slots), and the per-slot kernels read a shared expert once per slot. Dedup: the block of a slot whose expert already
// occurs at a lower slot exits; the first one handles every slot of that expert (one weight pass, one dot product per
// routed row; the weight chunk re-reads hit L1). Lists up to MOE_DD_MAX slots per expert (a token routes to an expert
// once, so <= tokens). Opt-in (GGML_CUDA_MOE_DEDUP=1): MTP verify 63.3/83.0 -> 56.9/74.1 t/s. The duplicate expert
// reads were nearly free (concurrent slots of one expert hit L2), while one block doing m rows' dot products and
// the uneven blocks cost more.
static constexpr int MOE_DD_MAX = 8;

static __device__ __forceinline__ int moe_dd_list(const moe_direct_t & direct, const int n_slots, const int n_expert_used,
        const int slot, int * s_list) {
    // all slots' experts into LDS in parallel (a serial walk over global ids by one thread cost ~12 us per block)
    __shared__ int s_e[256];
    __shared__ int s_m;
    for (int j = threadIdx.x; j < n_slots && j < 256; j += blockDim.x) {
        s_e[j] = direct.ids[(j / n_expert_used)*direct.s1 + j % n_expert_used];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int e = s_e[slot];
        int m = 0;
        bool dup = false;
        for (int j = 0; j < n_slots && !dup; ++j) {
            if (s_e[j] == e) {
                if (j < slot) {
                    dup = true;
                } else if (m < MOE_DD_MAX) {
                    s_list[m++] = j;
                }
            }
        }
        s_m = dup ? 0 : m;
    }
    __syncthreads();
    return s_m;
}

template <bool G5, bool U5, bool GLU, int LPR, int RG>
__launch_bounds__(256)
static __global__ void moe_coal_kq_ls_dd(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    extern __shared__ int4 ys_raw[]; // m rows of ncols_x/QK8_1 block_q8_1
    __shared__ int s_list[MOE_DD_MAX];
    const int m = moe_dd_list(direct, gridDim.y, n_expert_used, blockIdx.y, s_list);
    if (m == 0) {
        return;
    }
    const int expert = direct.ids[(blockIdx.y / n_expert_used)*direct.s1 + blockIdx.y % n_expert_used];
    const int nyb = ncols_x / QK8_1;
    const int rowi = nyb*(int) (sizeof(block_q8_1)/4); // ints per staged row
    for (int t = 0; t < m; ++t) {
        const int * src = (const int *) (vy + (int64_t) (s_list[t] / direct.y_div)*stride_col_y);
        int * d = (int *) ys_raw + t*rowi;
        for (int i = threadIdx.x; i < rowi; i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();
    const int sub = threadIdx.x % LPR;
    const int nit = (ncols_x / QK_K)*8;
#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*(256/LPR) + threadIdx.x / LPR;
        if (row >= nrows_x) {
            break;
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;
        float ag[MOE_DD_MAX], au[MOE_DD_MAX];
#pragma unroll
        for (int t = 0; t < MOE_DD_MAX; ++t) {
            ag[t] = 0.0f;
            au[t] = 0.0f;
        }
        for (int it = sub; it < nit; it += LPR) {
#pragma unroll
            for (int t = 0; t < MOE_DD_MAX; ++t) {
                if (t < m) {
                    const block_q8_1 * ys = (const block_q8_1 *) ((const int *) ys_raw + t*rowi);
                    ag[t] += moe_coal_kq_item<G5>(rg, ys, it);
                    if constexpr (GLU) {
                        au[t] += moe_coal_kq_item<U5>(ru, ys, it);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MOE_DD_MAX; ++t) {
            if (t < m) {
                float a = ag[t], u = au[t];
#pragma unroll
                for (int off = LPR/2; off > 0; off >>= 1) {
                    a += __shfl_xor(a, off, LPR);
                    if constexpr (GLU) {
                        u += __shfl_xor(u, off, LPR);
                    }
                }
                if (sub == 0) {
                    const int slot = s_list[t];
                    float * o = dst + (slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row;
                    *o = GLU ? moe_glu(a, u, direct.glu_limit) : a;
                }
            }
        }
    }
}

// ---- i-quant decode (iq2_xs / iq3_xxs), direct per slot ----
// GLM-5.3-Flash UD-Q2_K_XL: gate/up experts iq2_xs, down iq3_xxs (MMVQ took them before: tg 41 t/s at TP4 vs 57 for
// the larger Q4_K_XL). moe_coal_kq_ls with one 32-value sub-block per item: LPR lanes per row walk the row's sub-blocks
// (adjacent lanes, adjacent sub-blocks), the activation row sits in LDS. Signs are applied
// bytewise as in moe_rowlane_iq2xxs (every grid byte is > 0, so (g ^ m) + (m & 1) negates without carries); the
// sub-block sums are scaled in integer math (iq2_xs: (2*ls + 1) per 16 values, / 8; iq3_xxs: (2*ls + 1), / 4), which is
// the arithmetic of vec_dot_iq2_xs_q8_1 / vec_dot_iq3_xxs_q8_1 without their integer rounding.
static __device__ __forceinline__ uint32_t moe_iq_u16x2(const char * p) { // 4 bytes at a 2-byte aligned address
    return (uint32_t) *(const uint16_t *) p | ((uint32_t) *(const uint16_t *) (p + 2) << 16);
}

// one item's weight bytes (loaded ahead of their use): iq2_xs: d, qs of the sub-block (4 x 16 bits), its scale byte;
// iq3_xxs: d, the 8 grid indices, the signs | scale word; iq4_xs: d, the sub-block's 16 qs bytes (a, b, c, e), its
// 6-bit scale - 32 (s)
struct moe_iq_raw {
    uint32_t a, b, c, e;
    uint16_t d;
    int      s;
};

template <ggml_type type>
static __device__ __forceinline__ moe_iq_raw moe_iq_load(const char * __restrict__ row, const int it) {
    const int sb = it / 8, ib = it % 8;
    moe_iq_raw r;
    if constexpr (type == GGML_TYPE_IQ2_XS) {
        // 74 bytes: d, qs[32] (9-bit grid index | 7-bit sign index << 9), scales[8] (4 bits per 16 values)
        const char * b = row + sb*sizeof(block_iq2_xs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
        r.c = (uint8_t) b[2 + QK_K/4 + ib];
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        // 136 bytes: d, scales_h (2 bits per sub-block), scales_l[4] (4 bits per sub-block), qs[128] (16 bytes per
        // sub-block: value j in the low nibble of byte j, value 16 + j in its high nibble); rows are 8-byte aligned
        const char * b = row + sb*sizeof(block_iq4_xs);
        const uint32_t h = *(const uint32_t *) b;
        const uint2 q0 = *(const uint2 *) (b + 8 + 16*ib);
        const uint2 q1 = *(const uint2 *) (b + 16 + 16*ib);
        r.d = h & 0xFFFF;
        r.s = (int) ((((uint8_t) b[4 + ib/2] >> (4*(ib & 1))) & 0xF) | (((h >> (16 + 2*ib)) & 3) << 4)) - 32;
        r.a = q0.x; r.b = q0.y; r.c = q1.x; r.e = q1.y;
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        // 66 bytes: d, qs[32] (per sub-block: 4 grid indices (8 bits), then 4 x 7-bit sign sets | 4-bit scale << 28)
        const char * b = row + sb*sizeof(block_iq2_xxs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
    } else {
        static_assert(type == GGML_TYPE_IQ3_XXS, "unsupported type");
        // 98 bytes: d, qs[64] (grid indices, 4 values each), then per sub-block 4 x 7-bit sign sets | scale << 28
        const char * b = row + sb*sizeof(block_iq3_xxs);
        r.d = *(const uint16_t *) b;
        r.a = moe_iq_u16x2(b + 2 + 8*ib);
        r.b = moe_iq_u16x2(b + 6 + 8*ib);
        r.c = moe_iq_u16x2(b + 2 + QK_K/4 + 4*ib);
    }
    return r;
}

// The kernels are bound by LDS and VALU work per item (gfx906 counters, gate at 1 token: ~76 VALU, ~14 LDS and ~3.4
// VMEM instructions per 32 values), not by memory; computing the sign masks in registers instead of the LDS table
// measured no gain (tg 61.3 -> 60.9 t/s).
template <ggml_type type>
static __device__ __forceinline__ float moe_iq_dot(const moe_iq_raw & r, const block_q8_1 * __restrict__ ys, const int it,
        const uint2 * __restrict__ g2, const uint32_t * __restrict__ g3, const uint2 * __restrict__ sg) {
    const block_q8_1 * yb = ys + it;
    const int * yq = (const int *) yb->qs;
    const float d = __half2float(__ushort_as_half(r.d));
    if constexpr (type == GGML_TYPE_IQ2_XS) {
        int s[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = ((k < 2 ? r.a : r.b) >> (16*(k & 1))) & 0xFFFF;
            const uint2 g = g2[q & 0x1FF];
            const uint2 m = sg[q >> 9];
            const int v0 = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
            const int v1 = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            s[k] = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], 0));
        }
        const int si = (2*(int) (r.c & 0xF) + 1)*(s[0] + s[1]) + (2*(int) (r.c >> 4) + 1)*(s[2] + s[3]);
        return d*0.125f*__low2float(yb->ds)*(float) si;
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        // non-linear 4-bit values from kvalues_iq4nl by byte permutes (no tables in LDS)
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = get_int_from_table_16((int) (k == 0 ? r.a : k == 1 ? r.b : k == 2 ? r.c : r.e), kvalues_iq4nl);
            s = ggml_cuda_dp4a(v.y, yq[k + 4], ggml_cuda_dp4a(v.x, yq[k], s));
        }
        return d*__low2float(yb->ds)*(float) (r.s*s);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        // 8 values per grid entry (iq2xxs_grid in g2), signs as for iq2_xs; d*(0.5 + ls)/4 = d*(2*ls + 1)/8
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint2 g = g2[(r.a >> (8*k)) & 0xFF];
            const uint2 m = sg[(r.b >> (7*k)) & 0x7F];
            const int v0 = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
            const int v1 = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            s = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], s));
        }
        return d*0.125f*__low2float(yb->ds)*(float) ((2*(int) (r.b >> 28) + 1)*s);
    } else {
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t ii = k < 2 ? r.a : r.b;
            const uint32_t gl = g3[(ii >> (16*(k & 1)))     & 0xFF];
            const uint32_t gh = g3[(ii >> (16*(k & 1) + 8)) & 0xFF];
            const uint2 m = sg[(r.c >> (7*k)) & 0x7F];
            const int v0 = (int) ((gl ^ m.x) + (m.x & 0x01010101u));
            const int v1 = (int) ((gh ^ m.y) + (m.y & 0x01010101u));
            s = ggml_cuda_dp4a(v1, yq[2*k + 1], ggml_cuda_dp4a(v0, yq[2*k], s));
        }
        return d*0.25f*__low2float(yb->ds)*(float) ((2*(int) (r.c >> 28) + 1)*s);
    }
}

// GLU: gate (vg) and up (vu) of one type, dst = SwiGLU(-clamp) of the pair. A block stages the grid / sign tables and
// the activation row in LDS (dynamic part: ncols_x/QK8_1 blocks of block_q8_1) and walks RG row groups of 256/LPR rows,
// so that setup and its barrier are paid once per RG groups: with one group per block it was the cost of the down
// projection's many small blocks (TP4, K 512: 8192 blocks at 4 tokens, 142 us). The tables stay in LDS: gfx906 serves
// the lanes' scattered grid lookups from global memory (L1) far slower (gate 21 -> 32 us at 1 token). Each lane keeps
// PF items in flight (the next PF load while the current ones are summed).
#ifndef MOE_IQ_PF
#define MOE_IQ_PF 2
#endif
template <ggml_type type, bool GLU, int LPR, int RG>
__launch_bounds__(256)
static __global__ void moe_coal_iq(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    constexpr bool X2 = type == GGML_TYPE_IQ2_XS;
    constexpr bool I4 = type == GGML_TYPE_IQ4_XS; // no LDS tables
    constexpr bool XX = type == GGML_TYPE_IQ2_XXS;
    constexpr int  PF = MOE_IQ_PF;
    __shared__ uint2    g2[X2 ? 512 : XX ? 256 : 1];     // iq2xs_grid / iq2xxs_grid
    __shared__ uint32_t g3[X2 || I4 || XX ? 1 : 256];    // iq3xxs_grid
    __shared__ uint2    sg[I4 ? 1 : 128];         // ksigns_iq2xs expanded to byte masks (0x00 / 0xFF per value)
    extern __shared__ int4 ys_raw[];
    block_q8_1 * ys = (block_q8_1 *) ys_raw;

    const int slot   = blockIdx.y;
    const int expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
    const int nyb    = ncols_x / QK8_1;
    const int sub    = threadIdx.x % LPR;
    if constexpr (X2) {
        for (int i = threadIdx.x; i < 512; i += 256) {
            const uint64_t g = iq2xs_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (XX) {
        const uint64_t g = iq2xxs_grid[threadIdx.x];
        g2[threadIdx.x] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    } else if constexpr (!I4) {
        g3[threadIdx.x] = iq3xxs_grid[threadIdx.x];
    }
    if (!I4 && threadIdx.x < 128) {
        const uint32_t s = ksigns_iq2xs[threadIdx.x];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((s >> b)       & 1u) ? (0xFFu << (8*b)) : 0u;
            hi |= ((s >> (b + 4)) & 1u) ? (0xFFu << (8*b)) : 0u;
        }
        sg[threadIdx.x] = make_uint2(lo, hi);
    }
    {
        const int * src = (const int *) (vy + (int64_t) (slot / direct.y_div)*stride_col_y);
        int * d = (int *) ys;
        for (int i = threadIdx.x; i < nyb*(int) (sizeof(block_q8_1)/4); i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();

#pragma unroll 1
    for (int g = 0; g < RG; ++g) {
        const int row = (blockIdx.x*RG + g)*(256/LPR) + threadIdx.x / LPR;
        if (row >= nrows_x) {
            break; // whole lane groups; nothing after the loop needs the block
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;

        moe_iq_raw rgb[PF], rub[PF];
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int it = sub + p*LPR;
            if (it < nyb) {
                rgb[p] = moe_iq_load<type>(rg, it);
                if constexpr (GLU) {
                    rub[p] = moe_iq_load<type>(ru, it);
                }
            }
        }
        float ag = 0.0f, au = 0.0f;
        for (int it0 = sub; it0 < nyb; it0 += PF*LPR) {
            moe_iq_raw ngb[PF], nub[PF];
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + (PF + p)*LPR;
                if (it < nyb) {
                    ngb[p] = moe_iq_load<type>(rg, it);
                    if constexpr (GLU) {
                        nub[p] = moe_iq_load<type>(ru, it);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + p*LPR;
                if (it < nyb) {
                    ag += moe_iq_dot<type>(rgb[p], ys, it, g2, g3, sg);
                    if constexpr (GLU) {
                        au += moe_iq_dot<type>(rub[p], ys, it, g2, g3, sg);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                rgb[p] = ngb[p];
                if constexpr (GLU) {
                    rub[p] = nub[p];
                }
            }
        }
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            ag += __shfl_xor(ag, off, LPR);
            if constexpr (GLU) {
                au += __shfl_xor(au, off, LPR);
            }
        }
        if (sub == 0) {
            float * o = dst + (slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row;
            *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
        }
    }
}

// One token (every slot reads the same activation row): moe_coal_iq gave each (row group, slot) its own block, so the
// table + activation staging was paid by every block and a decode MUL_MAT_ID (6..8 slots of 512 rows) ran ~100 blocks:
// 1.5 waves per SIMD, latency bound (gfx906: the per-item work is ~76 VALU + ~14 LDS for 32 values). Here a grid of a
// few blocks per CU stages once and walks (slot, row group) tasks with a block stride; LPR lanes per row (16 / 32: short
// per-lane chains). GGML_CUDA_MOE_IQ_FLAT=0 off, GGML_CUDA_MOE_IQ_FLAT_LPR, GGML_CUDA_MOE_IQ_FLAT_BPC (blocks per CU).
template <ggml_type type, bool GLU, int LPR>
__launch_bounds__(256)
static __global__ void moe_coal_iq_flat(
        const char * __restrict__ vg, const char * __restrict__ vu, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t sg_row, const int64_t sg_exp, const int64_t su_row, const int64_t su_exp,
        const int n_expert_used, const int64_t s1_dst, const moe_direct_t direct, const int n_tok, const int64_t s2_dst,
        const int64_t stride_col_y) {
    constexpr bool X2 = type == GGML_TYPE_IQ2_XS;
    constexpr bool I4 = type == GGML_TYPE_IQ4_XS;
    constexpr bool XX = type == GGML_TYPE_IQ2_XXS;
    constexpr int  PF = 2;
    constexpr int  RPT = 256/LPR; // rows per task
    __shared__ uint2    g2[X2 ? 512 : XX ? 256 : 1];
    __shared__ uint32_t g3[X2 || I4 || XX ? 1 : 256];
    __shared__ uint2    sg[I4 ? 1 : 128];
    extern __shared__ int4 ys_raw[];
    block_q8_1 * ys = (block_q8_1 *) ys_raw;

    const int nyb = ncols_x / QK8_1;
    const int sub = threadIdx.x % LPR;
    if constexpr (X2) {
        for (int i = threadIdx.x; i < 512; i += 256) {
            const uint64_t g = iq2xs_grid[i];
            g2[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
        }
    } else if constexpr (XX) {
        const uint64_t g = iq2xxs_grid[threadIdx.x];
        g2[threadIdx.x] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    } else if constexpr (!I4) {
        g3[threadIdx.x] = iq3xxs_grid[threadIdx.x];
    }
    if (!I4 && threadIdx.x < 128) {
        const uint32_t sv = ksigns_iq2xs[threadIdx.x];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((sv >> b)       & 1u) ? (0xFFu << (8*b)) : 0u;
            hi |= ((sv >> (b + 4)) & 1u) ? (0xFFu << (8*b)) : 0u;
        }
        sg[threadIdx.x] = make_uint2(lo, hi);
    }
    // the tokens' activation rows, nyb blocks each, back to back
    for (int t = 0; t < n_tok; ++t) {
        const int * src = (const int *) (vy + (int64_t) t*stride_col_y);
        int * d = (int *) (ys + t*nyb);
        for (int i = threadIdx.x; i < nyb*(int) (sizeof(block_q8_1)/4); i += 256) {
            d[i] = src[i];
        }
    }
    __syncthreads();

    const int n_rg    = (nrows_x + RPT - 1)/RPT;
    const int n_tasks = n_rg*n_expert_used*n_tok;
#pragma unroll 1
    for (int task = blockIdx.x; task < n_tasks; task += gridDim.x) {
        const int slot   = task / n_rg;          // token*n_expert_used + k
        const int tok    = slot / n_expert_used;
        const int kk     = slot % n_expert_used;
        const int row    = (task % n_rg)*RPT + threadIdx.x / LPR;
        const int expert = direct.ids[tok*direct.s1 + kk];
        const block_q8_1 * yt = ys + tok*nyb;
        if (row >= nrows_x) {
            continue; // whole lane groups; the loop has no barrier
        }
        const char * rg = vg + expert*sg_exp + row*sg_row;
        const char * ru = GLU ? vu + expert*su_exp + row*su_row : nullptr;

        moe_iq_raw rgb[PF], rub[PF];
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int it = sub + p*LPR;
            if (it < nyb) {
                rgb[p] = moe_iq_load<type>(rg, it);
                if constexpr (GLU) {
                    rub[p] = moe_iq_load<type>(ru, it);
                }
            }
        }
        float ag = 0.0f, au = 0.0f;
        for (int it0 = sub; it0 < nyb; it0 += PF*LPR) {
            moe_iq_raw ngb[PF], nub[PF];
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + (PF + p)*LPR;
                if (it < nyb) {
                    ngb[p] = moe_iq_load<type>(rg, it);
                    if constexpr (GLU) {
                        nub[p] = moe_iq_load<type>(ru, it);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                const int it = it0 + p*LPR;
                if (it < nyb) {
                    ag += moe_iq_dot<type>(rgb[p], yt, it, g2, g3, sg);
                    if constexpr (GLU) {
                        au += moe_iq_dot<type>(rub[p], yt, it, g2, g3, sg);
                    }
                }
            }
#pragma unroll
            for (int p = 0; p < PF; ++p) {
                rgb[p] = ngb[p];
                if constexpr (GLU) {
                    rub[p] = nub[p];
                }
            }
        }
#pragma unroll
        for (int off = LPR/2; off > 0; off >>= 1) {
            ag += __shfl_xor(ag, off, LPR);
            if constexpr (GLU) {
                au += __shfl_xor(au, off, LPR);
            }
        }
        if (sub == 0) {
            float * o = dst + tok*s2_dst + kk*s1_dst + row;
            *o = GLU ? moe_glu(ag, au, direct.glu_limit) : ag;
        }
    }
}

template <int LPR>
__launch_bounds__(256)
static __global__ void moe_coal_q51_dd(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t s_row, const int64_t s_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    __shared__ int s_list[MOE_DD_MAX];
    const int m = moe_dd_list(direct, gridDim.y, n_expert_used, blockIdx.y, s_list);
    if (m == 0) {
        return;
    }
    const int expert = direct.ids[(blockIdx.y / n_expert_used)*direct.s1 + blockIdx.y % n_expert_used];
    const int sub = threadIdx.x % LPR;
    const int row = blockIdx.x*(256/LPR) + threadIdx.x / LPR;
    if (row >= nrows_x) {
        return;
    }
    const block_q8_1 * ys[MOE_DD_MAX];
#pragma unroll
    for (int t = 0; t < MOE_DD_MAX; ++t) {
        ys[t] = vy + (int64_t) (s_list[t < m ? t : 0] / direct.y_div)*stride_col_y;
    }
    const char * xr = vx + expert*s_exp + row*s_row;
    const int nb = ncols_x / QK5_1;
    float acc[MOE_DD_MAX];
#pragma unroll
    for (int t = 0; t < MOE_DD_MAX; ++t) {
        acc[t] = 0.0f;
    }
#pragma unroll 2
    for (int kb = sub; kb < nb; kb += LPR) {
        const int2 * p = (const int2 *) (xr + kb*sizeof(block_q5_1));
        const int2 t0 = p[0], t1 = p[1], t2 = p[2];
        const int w[6] = {t0.x, t0.y, t1.x, t1.y, t2.x, t2.y};
#pragma unroll
        for (int t = 0; t < MOE_DD_MAX; ++t) {
            if (t < m) {
                float tmp[1] = {0.0f};
                moe_rl_q5_1_block<1>(w, ys[t], 0, kb, 1, tmp);
                acc[t] += tmp[0];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MOE_DD_MAX; ++t) {
        if (t < m) {
            float a = acc[t];
#pragma unroll
            for (int off = LPR/2; off > 0; off >>= 1) {
                a += __shfl_xor(a, off, LPR);
            }
            if (sub == 0) {
                const int slot = s_list[t];
                dst[(slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row] = a;
            }
        }
    }
}

// q5_1 counterpart (decode down projection): work item = one 24-byte block, LPR lanes per row read consecutive blocks
template <int LPR>
__launch_bounds__(256)
static __global__ void moe_coal_q51(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t s_row, const int64_t s_exp,
        const int64_t stride_col_y, const int n_expert_used, const int64_t s1_dst, const int64_t s2_dst, const moe_direct_t direct) {
    const int slot   = blockIdx.y;
    const int expert = direct.ids[(slot / n_expert_used)*direct.s1 + slot % n_expert_used];
    const block_q8_1 * y = vy + (int64_t) (slot / direct.y_div)*stride_col_y;
    const int sub = threadIdx.x % LPR;
    const int row = blockIdx.x*(256/LPR) + threadIdx.x / LPR;
    if (row >= nrows_x) {
        return;
    }
    const char * xr = vx + expert*s_exp + row*s_row;
    const int nb = ncols_x / QK5_1;
    float acc = 0.0f;
#pragma unroll 4
    for (int kb = sub; kb < nb; kb += LPR) {
        const int2 * p = (const int2 *) (xr + kb*sizeof(block_q5_1)); // 8-byte aligned (24-byte blocks, K % 128 == 0)
        const int2 t0 = p[0], t1 = p[1], t2 = p[2];
        const int w[6] = {t0.x, t0.y, t1.x, t1.y, t2.x, t2.y};
        float tmp[1] = {0.0f};
        moe_rl_q5_1_block<1>(w, y, 0, kb, 1, tmp);
        acc += tmp[0];
    }
#pragma unroll
    for (int off = LPR/2; off > 0; off >>= 1) {
        acc += __shfl_xor(acc, off, LPR);
    }
    if (sub == 0) {
        dst[(slot / n_expert_used)*s2_dst + (slot % n_expert_used)*s1_dst + row] = acc;
    }
}

static int moe_vec_env(const char * name, int def);


typedef short moe_short2 __attribute__((ext_vector_type(2)));

// ---- q2_K row-lane (prefill tiles) ----
// q2_K: 84-byte super-blocks (scales[16]: 4-bit scale | 4-bit min per 16 values, qs[64]: 2-bit values, d, dmin). A lane
// owns one weight row; per 32-value sub-block the two 16-value dot products (4 dp4a each) combine with the 4-bit scales in
// integer math, and the min term uses the activations' integer sums per 16 values (int16 pairs written by the quantize
// kernel), one v_dot2_i32_i16 against the packed mins. Activation scales stay per 32 values (q8_1), as in MMQ.

// q8_1 activations plus the integer sums of each 16-value half (int16, [row][K/16])
static __global__ void moe_vec_quantize_gather_s16(const float * __restrict__ x, const int32_t * __restrict__ ids_src1,
        block_q8_1 * __restrict__ y, int16_t * __restrict__ s16, const int64_t ne10, const int64_t ne10_padded, const int64_t s_col) {
    const int64_t i0  = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (i0 >= ne10_padded) {
        return;
    }
    const int64_t src_row = ids_src1 ? (int64_t) ids_src1[row] : row;
    const float xi = i0 < ne10 ? x[src_row*s_col + i0] : 0.0f;

    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int  q16 = warp_reduce_sum<16>((int) q);

    block_q8_1 * yb = y + (row*ne10_padded + i0) / QK8_1;
    yb->qs[i0 % QK8_1] = q;
    if (i0 % QK8_1 == 0) {
        yb->ds = make_half2(d, sum);
    }
    if (i0 % 16 == 0) {
        s16[(row*ne10_padded + i0) / 16] = (int16_t) q16;
    }
}

// KS > 1 (decode-sized batches): the block is KS waves on the same 64 rows, wave w takes super-blocks
// [w*nsb/KS, (w+1)*nsb/KS) and wave 0 adds the partials from LDS in wave order (deterministic)
template <int J, bool DENSE = false, int KS = 1>
__launch_bounds__(64*KS)
static __global__ void moe_rowlane_q2k(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, const int16_t * __restrict__ vs16,
        const int2 * __restrict__ tiles, const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ ids_dst, float * __restrict__ dst, const int ncols_x, const int nrows_x,
        const int64_t stride_row_b, const int64_t stride_expert_b, const int64_t stride_col_y, const int n_expert_used,
        const int64_t s1_dst, const int64_t s2_dst, const int32_t * __restrict__ tok_map, const int glu_epi) {
    // DENSE: a plain matrix (one "expert", no ids): tile t = columns t*J.., n_expert_used carries the column count
    const int tile = blockIdx.y;
    int expert, col0, ncols;
    if constexpr (DENSE) {
        expert = 0;
        col0   = tile*J;
        ncols  = min(J, n_expert_used - col0);
    } else {
        if (tile >= *n_tiles) {
            return;
        }
        const int2 te = tiles[tile];
        expert = te.x;
        col0   = te.y;
        ncols  = min(J, expert_bounds[expert + 1] - col0);
    }

    const int  lane  = threadIdx.x % 64;
    const int  wave  = KS > 1 ? threadIdx.x / 64 : 0;
    const int  row   = blockIdx.x*64 + lane;
    const bool valid = row < nrows_x;
    const char * xr  = vx + expert*stride_expert_b + (valid ? row : nrows_x - 1)*stride_row_b;

    // activation rows of the tile columns (the clamp keeps partial tiles' unused entries valid); s16 rows have
    // 2*stride_col_y int16 entries (one per 16 values)
    const block_q8_1 * ycol[J];
    const int16_t    * scol[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int jj = min(j, ncols - 1);
        const int64_t r = tok_map ? (int64_t) tok_map[col0 + jj] : (int64_t) (col0 + jj);
        ycol[j] = vy   + r*stride_col_y;
        scol[j] = vs16 + r*2*stride_col_y;
    }
    float acc[J] = {0.0f};

    const int nsb = ncols_x / QK_K;
    const int sb0 = KS > 1 ? wave*nsb/KS : 0;
    const int sb1 = KS > 1 ? (wave + 1)*nsb/KS : nsb;
    for (int sb = sb0; sb < sb1; ++sb) {
        const int * p = (const int *) (xr + sb*sizeof(block_q2_K));
        // the 84-byte block as five 16-byte loads + one dword (blocks are only 4-byte aligned: dword-aligned dwordx4 is
        // fine on gfx9); 21 dword loads per lane, each touching 64 cache lines across the wave's rows, thrashed L1
        typedef int moe_int4a __attribute__((ext_vector_type(4), aligned(4)));
        moe_int4a pv[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) {
            pv[i] = ((const moe_int4a *) p)[i];
        }
        int scw[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            scw[i] = pv[0][i];
        }
        const float2 dm = moe_rl_h2f(p[20]); // d, dmin
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int qd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qd[i] = pv[1 + 2*h + i/4][i % 4];
            }
#pragma unroll
            for (int jb = 0; jb < 4; ++jb) {
                const int s  = 4*h + jb;                   // 32-value sub-block of the super-block
                const int is = 8*h + 2*jb;                 // its two 16-value scale bytes
                const int b0 = (scw[is/4] >> (8*(is % 4))) & 0xFF;
                const int b1 = (scw[is/4] >> (8*(is % 4) + 8)) & 0xFF;
                const int sc0 = b0 & 0xF, sc1 = b1 & 0xF;
                const int mpair = (b0 >> 4) | ((b1 >> 4) << 16);
                int lo[4], hi[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    lo[i] = (qd[i]     >> (2*jb)) & 0x03030303;
                    hi[i] = (qd[4 + i] >> (2*jb)) & 0x03030303;
                }
                auto col = [&](const int j) {
                    const block_q8_1 * yb = ycol[j] + sb*8 + s;
                    // two dword-aligned 16-byte loads instead of eight dword loads (the qs of a block_q8_1 start at byte 4)
                    typedef int moe_int4y __attribute__((ext_vector_type(4), aligned(4)));
                    const moe_int4y y0 = ((const moe_int4y *) yb->qs)[0];
                    const moe_int4y y1 = ((const moe_int4y *) yb->qs)[1];
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s0 = ggml_cuda_dp4a(lo[i], y0[i], s0);
                        s1 = ggml_cuda_dp4a(hi[i], y1[i], s1);
                    }
                    const int isum  = __mul24(sc0, s0) + __mul24(sc1, s1); // |s| < 2^13: 24-bit multiplies
                    const int spair = *(const int *) (scol[j] + 2*(sb*8 + s));
                    const int msum  = __builtin_amdgcn_sdot2(__builtin_bit_cast(moe_short2, mpair),
                                                             __builtin_bit_cast(moe_short2, spair), 0, false);
                    const float da = __low2float(yb->ds);
                    acc[j] = fmaf(da, fmaf(dm.x, (float) isum, -dm.y*(float) msum), acc[j]);
                };
                if (ncols == J) {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        col(j);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        if (j >= ncols) {
                            break;
                        }
                        col(j);
                    }
                }
            }
        }
    }

    if constexpr (KS > 1) {
        __shared__ float red[KS > 1 ? KS - 1 : 1][J][64];
        if (wave > 0) {
#pragma unroll
            for (int j = 0; j < J; ++j) {
                red[wave - 1][j][lane] = acc[j];
            }
        }
        __syncthreads();
        if (wave > 0) {
            return;
        }
#pragma unroll
        for (int w = 0; w < KS - 1; ++w) {
#pragma unroll
            for (int j = 0; j < J; ++j) {
                acc[j] += red[w][j][lane];
            }
        }
    }

    if (valid) {
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (j < ncols) {
                float * o;
                if constexpr (DENSE) {
                    o = dst + (int64_t) (col0 + j)*s1_dst + row;
                } else {
                    const int d = ids_dst[col0 + j];
                    o = dst + (d / n_expert_used)*s2_dst + (d % n_expert_used)*s1_dst + row;
                }
                if (glu_epi) {
                    const float g = *o;
                    *o = g/(1.0f + expf(-g))*acc[j];
                } else {
                    *o = acc[j];
                }
            }
        }
    }
}

static bool moe_rowlane_ok(const ggml_tensor * src0) {
    static const int env = moe_vec_env("GGML_CUDA_MOE_ROWLANE", 1);
    if (env == 0) {
        return false;
    }
    // whole-block vector loads: 16-byte aligned rows and experts (8-byte for q8_0)
    const size_t align = src0->type == GGML_TYPE_Q8_0 ? 8 : 16;
    if (src0->nb[1] % align != 0 || src0->nb[2] % align != 0 || ((uintptr_t) src0->data) % align != 0) {
        return false;
    }
    return (src0->type == GGML_TYPE_Q5_1 && src0->ne[0] % QK5_1 == 0) || // (row stride checked above; K tail handled)
           (src0->type == GGML_TYPE_Q8_0 && src0->ne[0] % 128 == 0) ||
           ((src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K) && src0->ne[0] % QK_K == 0);
}

static int moe_vec_env(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

static bool moe_vec_type_ok(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K: case GGML_TYPE_IQ2_XXS:
            return true;
        default:
            return false;
    }
}

// ---- iq2_xxs row-lane (prefill tiles) ----
// iq2_xxs: 66-byte super-blocks (d, then per 32-value sub-block 8 bytes: 4 grid indices + 4 x 7-bit sign sets and a
// 4-bit scale). A lane owns one weight row: the grid rows (8 magnitudes) and the expanded sign masks come from LDS
// tables, signs are applied bytewise ((g ^ m) + (m & 1) negates, g <= 43 so no byte carries), and each sub-block's
// dot product is scaled by d*(2*ls + 1)/8 against the q8_1 activation scale of the same 32 values.
template <int J>
__launch_bounds__(64)
static __global__ void moe_rowlane_iq2xxs(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        const int2 * __restrict__ tiles, const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ ids_dst, float * __restrict__ dst, const int ncols_x, const int nrows_x,
        const int64_t stride_row_b, const int64_t stride_expert_b, const int64_t stride_col_y, const int n_expert_used,
        const int64_t s1_dst, const int64_t s2_dst, const int32_t * __restrict__ tok_map, const int glu_epi) {
    __shared__ uint2 grid_lds[256];  // iq2xxs_grid
    __shared__ uint2 sign_lds[128];  // ksigns_iq2xs expanded to byte masks (0x00 / 0xFF per value)
    for (int i = threadIdx.x; i < 256; i += 64) {
        const uint64_t g = iq2xxs_grid[i];
        grid_lds[i] = make_uint2((uint32_t) g, (uint32_t) (g >> 32));
    }
    for (int i = threadIdx.x; i < 128; i += 64) {
        const uint32_t s = ksigns_iq2xs[i];
        uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            lo |= ((s >> b)       & 1u) ? (0xFFu << (8*b)) : 0u;
            hi |= ((s >> (b + 4)) & 1u) ? (0xFFu << (8*b)) : 0u;
        }
        sign_lds[i] = make_uint2(lo, hi);
    }
    __syncthreads();

    const int tile = blockIdx.y;
    if (tile >= *n_tiles) {
        return;
    }
    const int2 te     = tiles[tile];
    const int  expert = te.x;
    const int  col0   = te.y;
    const int  ncols  = min(J, expert_bounds[expert + 1] - col0);

    const int  row   = blockIdx.x*64 + threadIdx.x;
    const bool valid = row < nrows_x;
    const char * xr  = vx + expert*stride_expert_b + (valid ? row : nrows_x - 1)*stride_row_b;

    const block_q8_1 * ycol[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int jj = min(j, ncols - 1);
        ycol[j] = vy + (tok_map ? (int64_t) tok_map[col0 + jj] : (int64_t) (col0 + jj))*stride_col_y;
    }
    float acc[J] = {0.0f};

    const int nsb = ncols_x / QK_K;
    for (int sb = 0; sb < nsb; ++sb) {
        // 66 bytes at a 2-byte aligned address a (the alignment is the same for every row: the row stride is a multiple
        // of 4 for K % 512 == 0; otherwise per lane, which the selects below also handle). From the dword base a & ~3:
        // a % 4 == 2: d is the high half of dword 0 and sub-block ib's aux words are dwords 1+2ib, 2+2ib exactly;
        // a % 4 == 0: d is the low half, aux words straddle dwords (realigned by 2 bytes) and the last dword is only
        // loaded as 16 bits so nothing past the 66 bytes is read
        const uintptr_t a  = (uintptr_t) (xr + sb*sizeof(block_iq2_xxs));
        const int *     q  = (const int *) (a & ~(uintptr_t) 3);
        const bool      odd = (a & 3) != 0;
        int w[17];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            w[i] = q[i];
        }
        w[16] = odd ? q[16] : (int) *(const unsigned short *) (q + 16);
        const float d = __half2float(__ushort_as_half((unsigned short) (odd ? ((uint32_t) w[0] >> 16) : (w[0] & 0xFFFF))));
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            // aux0 = 4 grid indices, aux1 = signs (4 x 7 bits) | scale << 28
            const uint32_t aux0 = odd ? (uint32_t) w[1 + 2*ib] : (uint32_t) __builtin_amdgcn_alignbyte(w[1 + 2*ib], w[2*ib], 2);
            const uint32_t aux1 = odd ? (uint32_t) w[2 + 2*ib] : (uint32_t) __builtin_amdgcn_alignbyte(w[2 + 2*ib], w[1 + 2*ib], 2);
            const float db = d*(float) (2*(aux1 >> 28) + 1)*0.125f;
            int v[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint2 g = grid_lds[(aux0 >> (8*k)) & 0xFF];
                const uint2 m = sign_lds[(aux1 >> (7*k)) & 0x7F];
                v[2*k + 0] = (int) ((g.x ^ m.x) + (m.x & 0x01010101u));
                v[2*k + 1] = (int) ((g.y ^ m.y) + (m.y & 0x01010101u));
            }
            auto col = [&](const int j) {
                const block_q8_1 * yb = ycol[j] + sb*8 + ib;
                const int * yq = (const int *) yb->qs;
                int s = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s = ggml_cuda_dp4a(v[i], yq[i], s);
                }
                acc[j] = fmaf(db*__low2float(yb->ds), (float) s, acc[j]);
            };
            if (ncols == J) {
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    col(j);
                }
            } else {
#pragma unroll
                for (int j = 0; j < J; ++j) {
                    if (j >= ncols) {
                        break;
                    }
                    col(j);
                }
            }
        }
    }

    if (valid) {
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (j < ncols) {
                const int dd = ids_dst[col0 + j];
                float * o = dst + (dd / n_expert_used)*s2_dst + (dd % n_expert_used)*s1_dst + row;
                if (glu_epi) {
                    const float g = *o;
                    *o = g/(1.0f + expf(-g))*acc[j];
                } else {
                    *o = acc[j];
                }
            }
        }
    }
}


// bytes 0..127 -> signed byte - off without borrows between bytes
static __device__ __forceinline__ int gkq_sub_bytes_mv(const int v, const unsigned off4) {
    return (int) (((unsigned) (v | 0x80808080) - off4) ^ 0x80808080u);
}
// ---- q3_K row-lane (prefill tiles) ----
// q3_K: 110-byte super-blocks (hmask[32]: high bit, qs[64]: low 2 bits, scales[12]: 16 6-bit scales, d). A lane owns one
// weight row; values are (q2 | h << 2) - 4 per byte (-4..3), each 16-value half dot product (4 dp4a) is scaled by its
// signed 6-bit scale in integer math (24-bit multiplies), and the super-block's d and the q8_1 activation scale of the 32
// values are applied once per sub-block. Super-blocks start 2-byte aligned (110 bytes): every lane reads the 28 dwords
// from the aligned base at or below its block unconditionally (divergent per-lane branches serialized each load) and
// realigns them by 0 or 2 bytes; the last dword holds block bytes 108..109, so nothing past the block's last dword is read.
template <int J, bool DENSE = false>
__launch_bounds__(64)
static __global__ void moe_rowlane_q3k(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy,
        const int2 * __restrict__ tiles, const int32_t * __restrict__ n_tiles, const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ ids_dst, float * __restrict__ dst, const int ncols_x, const int nrows_x,
        const int64_t stride_row_b, const int64_t stride_expert_b, const int64_t stride_col_y, const int n_expert_used,
        const int64_t s1_dst, const int64_t s2_dst, const int32_t * __restrict__ tok_map, const int glu_epi) {
    // DENSE: a plain matrix (one "expert", no ids): tile t = columns t*J.., n_expert_used carries the column count
    const int tile = blockIdx.y;
    int expert, col0, ncols;
    if constexpr (DENSE) {
        expert = 0;
        col0   = tile*J;
        ncols  = min(J, n_expert_used - col0);
    } else {
        if (tile >= *n_tiles) {
            return;
        }
        const int2 te = tiles[tile];
        expert = te.x;
        col0   = te.y;
        ncols  = min(J, expert_bounds[expert + 1] - col0);
    }

    const int  row   = blockIdx.x*64 + threadIdx.x;
    const bool valid = row < nrows_x;
    const char * xr  = vx + expert*stride_expert_b + (valid ? row : nrows_x - 1)*stride_row_b;

    const block_q8_1 * ycol[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int jj = min(j, ncols - 1);
        ycol[j] = vy + (tok_map ? (int64_t) tok_map[col0 + jj] : (int64_t) (col0 + jj))*stride_col_y;
    }
    float acc[J] = {0.0f};

    const int nsb = ncols_x / QK_K;
    for (int sb = 0; sb < nsb; ++sb) {
        // all 28 dwords from the aligned base (a pointer off xr, so the loads stay global and merge), realigned per lane
        const char * bp  = xr + sb*sizeof(block_q3_K);
        const int    mis = (int) ((uintptr_t) bp & 3); // 0 or 2
        const int  * q   = (const int *) (bp - mis);
        int w[28];
#pragma unroll
        for (int k = 0; k < 28; ++k) {
            w[k] = q[k];
        }
        auto dw = [&](const int k) { // super-block dword k
            return (int) __builtin_amdgcn_alignbyte(w[k + 1], w[k], mis); // byte shift
        };
        int hm[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            hm[i] = dw(i);           // hmask bytes 0..31
        }
        const unsigned a0 = dw(24), a1 = dw(25), tmp = dw(26); // scales bytes 96..107
        const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) w[27] >> (8*mis))));
        // 16 signed 6-bit scales (bytes of 4 words), minus 32
        const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        int scw[4];
        scw[0] = gkq_sub_bytes_mv((int) ((a0 & km2)        | (((tmp >> 0) & km1) << 4)), 0x20202020u);
        scw[1] = gkq_sub_bytes_mv((int) ((a1 & km2)        | (((tmp >> 2) & km1) << 4)), 0x20202020u);
        scw[2] = gkq_sub_bytes_mv((int) (((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)), 0x20202020u);
        scw[3] = gkq_sub_bytes_mv((int) (((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)), 0x20202020u);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int qd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qd[i] = dw(8 + 8*h + i); // qs bytes 32 + 32h ..
            }
#pragma unroll
            for (int jb = 0; jb < 4; ++jb) {
                const int s  = 4*h + jb;
                const int is = 8*h + 2*jb;
                const int sc0 = (int8_t) ((scw[is/4] >> (8*(is % 4)))     & 0xFF);
                const int sc1 = (int8_t) ((scw[is/4] >> (8*(is % 4) + 8)) & 0xFF);
                // x = q2 | h << 2 per byte (0..7), then one byte permute maps x -> x - 4
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int hb = s >= 2 ? hm[i] >> (s - 2) : hm[i] << (2 - s);
                    const int x  = (hb & 0x04040404) | ((qd[i] >> (2*jb)) & 0x03030303);
                    v[i] = (int) __builtin_amdgcn_perm(0x03020100u, 0xFFFEFDFCu, (unsigned) x);
                }
                auto col = [&](const int j) {
                    const block_q8_1 * yb = ycol[j] + sb*8 + s;
                    const int * yq = (const int *) yb->qs;
                    int s0 = 0, s1 = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        s0 = ggml_cuda_dp4a(v[i],     yq[i],     s0);
                        s1 = ggml_cuda_dp4a(v[4 + i], yq[4 + i], s1);
                    }
                    const int isum = __mul24(sc0, s0) + __mul24(sc1, s1); // |s| < 2^13: 24-bit multiplies
                    acc[j] = fmaf(__low2float(yb->ds), d*(float) isum, acc[j]);
                };
                if (ncols == J) {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        col(j);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < J; ++j) {
                        if (j >= ncols) {
                            break;
                        }
                        col(j);
                    }
                }
            }
        }
    }

    if (valid) {
#pragma unroll
        for (int j = 0; j < J; ++j) {
            if (j < ncols) {
                float * o;
                if constexpr (DENSE) {
                    o = dst + (int64_t) (col0 + j)*s1_dst + row;
                } else {
                    const int dd = ids_dst[col0 + j];
                    o = dst + (dd / n_expert_used)*s2_dst + (dd % n_expert_used)*s1_dst + row;
                }
                if (glu_epi) {
                    const float g = *o;
                    *o = g/(1.0f + expf(-g))*acc[j];
                } else {
                    *o = acc[j];
                }
            }
        }
    }
}

// waves per block for the q2_K row-lane kernel at decode-sized batches: the smallest of 1, 2, 4, 8 giving >= 240 waves
// (one per SIMD) while every wave keeps >= 2 super-blocks
static int moe_rowlane_q2k_ks(const int64_t row_waves, const int64_t nsb) {
    static const int ks_env = moe_vec_env("GGML_CUDA_ROWLANE_KS", 0);
    if (ks_env > 0) {
        return ks_env;
    }
    int ks = 1;
    while (ks < 8 && row_waves*ks < 240 && nsb >= 4*ks) {
        ks *= 2;
    }
    return ks;
}

// Dense q2_K / q3_K matmuls with a few tokens (multi-sequence decode): the row-lane kernels above with one "expert"
// (DENSE). MMVQ took ~70 us per call on DeepSeek V4.1's projections at 8 tokens (short K-quant rows, decode work per
// column) against a few us of weight traffic. GGML_CUDA_ROWLANE_DENSE=0 off; token range _MIN (2) .. _MAX (16).
bool ggml_cuda_rowlane_dense_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int env   = moe_vec_env("GGML_CUDA_ROWLANE_DENSE", 1);
    static const int n_min    = moe_vec_env("GGML_CUDA_ROWLANE_DENSE_MIN", 1);
    static const int n_min_q3 = moe_vec_env("GGML_CUDA_ROWLANE_DENSE_MIN_Q3K", 1); // short-K q3_K: from 1 token
    static const int n_max = moe_vec_env("GGML_CUDA_ROWLANE_DENSE_MAX", 16);
    const int64_t N = src1->ne[1];
    const bool type_ok = src0->type == GGML_TYPE_Q3_K || (src0->type == GGML_TYPE_Q2_K && src0->nb[1] % 4 == 0);
    // the kernel parallelizes over rows only (a wave per 64 rows walks all of K): it wins with many rows or a short K
    // (8 tokens: q2_K 8192x1280 60 vs 106 us, q3_K 5120x512 22 vs 53) and loses on few rows x long K (576x5120: 210 vs
    // 34), which stay on MMVQ
    // q2_K at 1..4 tokens: the K-split waves could also cover few rows x long K (M >= 256), but MMVQ is faster there
    // (1 token: 512x5120 7.4 vs 15.6 us, 1280x5120 11.4 vs 16.8, 5120x2048 14.9 vs 16.3); GGML_CUDA_ROWLANE_DENSE_KS=1 on
    static const int ks_small = moe_vec_env("GGML_CUDA_ROWLANE_DENSE_KS", 0);
    const bool shape_ok = src0->ne[1] >= 8192 || src0->ne[0] <= 1024 ||
        (ks_small && src0->type == GGML_TYPE_Q2_K && N <= 4 && src0->ne[1] >= 256);
    const int nmin = src0->type == GGML_TYPE_Q3_K ? n_min_q3 : n_min;
    return env && GGML_CUDA_CC_IS_GCN(cc) && type_ok && shape_ok && src0->ne[0] % QK_K == 0 && N >= nmin && N <= n_max &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1 && ggml_is_contiguous(src0) &&
        src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->nb[1] % 2 == 0 &&
        ((uintptr_t) src0->data) % 4 == 0;
}

void ggml_cuda_rowlane_dense(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int64_t K_pad   = GGML_PAD(K, 256);
    const int64_t s_col_y = K_pad / QK8_1;
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), N*s_col_y);
    ggml_cuda_pool_alloc<int16_t>    y16(ctx.pool(), N*(K_pad/16));
    moe_vec_quantize_gather_s16<<<dim3((K_pad + 255)/256, N), 256, 0, stream>>>((const float *) src1->data, nullptr,
        y.get(), y16.get(), K, K_pad, src1->nb[1] / sizeof(float));
    constexpr int J = 8;
    const dim3 grid((M + 63)/64, (N + J - 1)/J);
    const int64_t s1 = dst->nb[1] / sizeof(float);
    if (src0->type == GGML_TYPE_Q2_K) {
        const int jn = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : 8;
        const int ks = N <= 4 ? moe_rowlane_q2k_ks((M + 63)/64, K/QK_K) : 1;
        const dim3 grid_n((M + 63)/64, (N + jn - 1)/jn);
#define RL_DENSE(jj, kk) moe_rowlane_q2k<jj, true, kk><<<grid_n, 64*kk, 0, stream>>>((const char *) src0->data, y.get(), \
            y16.get(), nullptr, nullptr, nullptr, nullptr, (float *) dst->data, (int) K, (int) M, src0->nb[1], 0, s_col_y, \
            (int) N, s1, 0, nullptr, 0)
#define RL_DENSE_KS(jj) switch (ks) { case 2: RL_DENSE(jj, 2); break; case 4: RL_DENSE(jj, 4); break; \
            case 8: RL_DENSE(jj, 8); break; default: RL_DENSE(jj, 1); break; }
        switch (jn) {
            case 1:  RL_DENSE_KS(1); break;
            case 2:  RL_DENSE_KS(2); break;
            case 4:  RL_DENSE_KS(4); break;
            default: RL_DENSE(8, 1); break;
        }
#undef RL_DENSE_KS
#undef RL_DENSE
    } else {
        moe_rowlane_q3k<J, true><<<grid, 64, 0, stream>>>((const char *) src0->data, y.get(), nullptr, nullptr,
            nullptr, nullptr, (float *) dst->data, (int) K, (int) M, src0->nb[1], 0, s_col_y, (int) N, s1, 0, nullptr, 0);
    }
    CUDA_CHECK(cudaGetLastError());
}

// q2_K prefill tiles on GCN (GGML_CUDA_MOE_Q2K=0: MMQ): more than 4 tokens (MMVQ keeps 1..4), whole super-blocks
// iq2_xxs likewise (GGML_CUDA_MOE_IQ2XXS=0: MMQ)
static bool moe_q2k_ok(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const int env    = moe_vec_env("GGML_CUDA_MOE_Q2K", 1);
    static const int env_xx = moe_vec_env("GGML_CUDA_MOE_IQ2XXS", 1);
    static const int env_q3 = moe_vec_env("GGML_CUDA_MOE_Q3K", 1);
    // above this many tokens (MMVQ takes 1..4 for these types on GCN; 5..8 fell to MMQ, 2.6 ms per call in DeepSeek V4.1
    // 8-sequence decode)
    static const int min_tokens = moe_vec_env("GGML_CUDA_MOE_Q2K_MIN_TOKENS", 4);
    // q3_K from 1 token: DeepSeek V4.1's expert down at TP4 has K = 512 (2 super-blocks per row), which leaves most
    // MMVQ lanes idle (88 us for 6 x 5120 x 512 at 1 token)
    static const int min_tokens_q3 = moe_vec_env("GGML_CUDA_MOE_Q3K_MIN_TOKENS", 0);
    // short-K q2_K (DeepSeek V4 Flash's expert down at TP4, 4096 x 512) from 1 token as well: MMVQ leaves most lanes idle
    // on 2 super-blocks per row (DS4 tg 35.9 -> 37.0, 4-token forward 94.9 -> 101.5 t/s); long-K q2_K (V4.1's gate/up)
    // keeps the threshold above. GGML_CUDA_MOE_Q2K_SHORT_MIN_TOKENS: the short-K threshold (default 0)
    static const int min_tokens_q2s = moe_vec_env("GGML_CUDA_MOE_Q2K_SHORT_MIN_TOKENS", 0);
    const int min_tok = src0->type == GGML_TYPE_Q3_K ? min_tokens_q3 :
                        src0->type == GGML_TYPE_Q2_K && src0->ne[0] <= 1024 ? min_tokens_q2s : min_tokens;
    const bool type_ok = (env && src0->type == GGML_TYPE_Q2_K) || (env_xx && src0->type == GGML_TYPE_IQ2_XXS) ||
                         (env_q3 && src0->type == GGML_TYPE_Q3_K);
    return type_ok && GGML_CUDA_CC_IS_GCN(cc) && src0->ne[0] % QK_K == 0 &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src0->ne[3] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        ids->nb[0] == sizeof(int32_t) && src0->nb[1] % 2 == 0 && src0->nb[2] % 2 == 0 && ((uintptr_t) src0->data) % 4 == 0 &&
        src1->ne[2] > min_tok;
}

bool ggml_cuda_moe_vec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                 const ggml_tensor * dst) {
    if (moe_q2k_ok(cc, src0, src1, ids, dst)) {
        return true;
    }
    // the row-lane kernel is the default on GCN for the types it covers; the grouped vec kernel (GGML_CUDA_MOE_VEC=1)
    // loses to MMQ on gfx906 (per-lane weight rows thrash L1, per-column y loads bound the T>1 variants)
    static const int env = moe_vec_env("GGML_CUDA_MOE_VEC", 0);
    const bool rowlane = GGML_CUDA_CC_IS_GCN(cc) && moe_rowlane_ok(src0);
    if (env != 1 && !rowlane) {
        return false;
    }
    if (!moe_vec_type_ok(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src0) || src1->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (src0->ne[0] % ggml_blck_size(src0->type) != 0) {
        return false;
    }
    // only for few tokens per expert; MMQ wins once the tiles fill up
    const int64_t n_slots = src1->ne[2]*ids->ne[0];
    const int64_t avg = (n_slots + src0->ne[2] - 1) / src0->ne[2];
    static const int max_avg = moe_vec_env("GGML_CUDA_MOE_VEC_MAX_AVG", 32);
    return avg <= max_avg;
}

// iq2_xs / iq3_xxs / iq4_xs / iq2_xxs experts at small batches (1..64 tokens, iq2_xxs 1..32) on GCN: moe_coal_iq (GGML_CUDA_MOE_IQ_DEC=0:
// MMVQ; GGML_CUDA_MOE_IQ4_DEC=0 / GGML_CUDA_MOE_IQ2XXS_DEC=0: the previous path for that type only; iq2_xxs = DeepSeek V4
// Flash IQ2XXS gate/up). iq4_xs (GLM-5.3 Q2 down, layers 11/12/44) only without a GLU partner.
static bool moe_iq_dec_ok(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    static const int env  = moe_vec_env("GGML_CUDA_MOE_IQ_DEC", 1);
    static const int env4 = moe_vec_env("GGML_CUDA_MOE_IQ4_DEC", 1);
    static const int envx = moe_vec_env("GGML_CUDA_MOE_IQ2XXS_DEC", 1);
    const bool bcast = src1->ne[1] == 1;
    const bool i4    = src0->type == GGML_TYPE_IQ4_XS;
    const int  al    = i4 ? 8 : 2; // iq4_xs loads qs as 8-byte pairs
    // tokens per call (GGML_CUDA_MOE_IQ_DEC_MAX; the expert-grouped GEMM in gcn-q8-gemm.cu starts above 64): the per-slot
    // coalesced kernel beats the GEMM well past decode sizes. GLM-5.3 Q2 TP4 forward ms, GEMM -> here: 5 tokens 62.9 -> 32.5,
    // 8: 80.3 -> 45.1, 16: 120.9 -> 78.8, 32: 178.0 -> 141.3, 64: 229.0 -> 199.0, 96: 274.3 -> 268.4. iq2_xxs (DeepSeek V4,
    // against the expert-sorted row-lane tiles): 6: 55.6 -> 49.0, 32: 186.6 -> 182.3, 64: 183.6 -> 194.0, so 32 there
    static const int env_max = moe_vec_env("GGML_CUDA_MOE_IQ_DEC_MAX", 0);
    const int max_tok = env_max > 0 ? std::min(env_max, 256) : src0->type == GGML_TYPE_IQ2_XXS ? 32 : 64;
    return env && GGML_CUDA_CC_IS_GCN(cc) && (src0->type == GGML_TYPE_IQ2_XS || src0->type == GGML_TYPE_IQ3_XXS || (i4 && env4) ||
         (src0->type == GGML_TYPE_IQ2_XXS && envx)) &&
        src0->ne[0] % QK_K == 0 && src0->ne[0] <= 8192 && src0->ne[3] == 1 && ggml_is_contiguous(src0) &&
        src0->nb[1] % al == 0 && src0->nb[2] % al == 0 && ((uintptr_t) src0->data) % al == 0 &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float) &&
        dst->nb[0] == sizeof(float) && ids->nb[0] == sizeof(int32_t) && src1->ne[2] <= max_tok &&
        (bcast || src1->nb[2] == src1->ne[1]*src1->nb[1]);
}

bool ggml_cuda_moe_vec_decode_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                        const ggml_tensor * dst) {
    if (moe_iq_dec_ok(cc, src0, src1, ids, dst)) {
        return true;
    }
    // default 8 tokens (2026-10-03; was 4): with the direct per-slot path the 5..8-token batches of multi-user decode
    // and MTP verify no longer fall onto the sorted tiles (GLM-5.3 Q4, 8 MI50s, S_TG 5 seqs 104.8 -> 126.9, 8: 140.2 -> 161.1)
    static const int dec_max = moe_vec_env("GGML_CUDA_MOE_DEC_MAX_TOKENS", 8);
    static const int dec_ks  = moe_vec_env("GGML_CUDA_MOE_DEC_KS", MOE_DEC_KS_DEFAULT);
    return dec_ks > 1 && src1->ne[2] <= std::min(dec_max, 16) && GGML_CUDA_CC_IS_GCN(cc) && moe_rowlane_ok(src0) &&
        ggml_cuda_moe_vec_supported(cc, src0, src1, ids, dst);
}

template <ggml_type type, int NC>
static void moe_vec_launch_t(int T, dim3 grid_x_rows, const char * x, const block_q8_1 * y, const int2 * tiles, const int32_t * n_tiles,
        const int32_t * bounds, const int32_t * ids_dst, float * dst, int ncols_x, int nrows_x, int64_t s01, int64_t s02,
        int64_t s_col_y, int n_used, int64_t s1, int64_t s2, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 block(warp_size, MOE_VEC_NWARPS);
    const int rpb_1 = MOE_VEC_NWARPS*warp_size;
    auto go = [&](auto kern, int t) {
        dim3 grid((nrows_x + rpb_1/t - 1)/(rpb_1/t), grid_x_rows.y);
        kern<<<grid, block, 0, stream>>>(x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2);
    };
    switch (T) {
        case 1:  go(moe_vec_q<type, NC, 1>,  1);  break;
        case 2:  go(moe_vec_q<type, NC, 2>,  2);  break;
        case 4:  go(moe_vec_q<type, NC, 4>,  4);  break;
        case 8:  go(moe_vec_q<type, NC, 8>,  8);  break;
        case 16: go(moe_vec_q<type, NC, 16>, 16); break;
        case 32: go(moe_vec_q<type, NC, 32>, 32); break;
        default: go(moe_vec_q<type, NC, 64>, 64); break;
    }
}

template <ggml_type type>
static void moe_vec_launch_type(int NC, int T, dim3 grid, const char * x, const block_q8_1 * y, const int2 * tiles, const int32_t * n_tiles,
        const int32_t * bounds, const int32_t * ids_dst, float * dst, int ncols_x, int nrows_x, int64_t s01, int64_t s02,
        int64_t s_col_y, int n_used, int64_t s1, int64_t s2, cudaStream_t stream) {
    switch (NC) {
        case 4:  moe_vec_launch_t<type, 4> (T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
        case 8:  moe_vec_launch_t<type, 8> (T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
        default: moe_vec_launch_t<type, 16>(T, grid, x, y, tiles, n_tiles, bounds, ids_dst, dst, ncols_x, nrows_x, s01, s02, s_col_y, n_used, s1, s2, stream); break;
    }
}

void ggml_cuda_moe_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up_src0, const float glu_limit) {
    cudaStream_t stream = ctx.stream();
    // the SwiGLU(-clamp) limit for the tile launches (no direct ids)
    moe_direct_t dl;
    dl.glu_limit = glu_limit;

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // rows
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2]; // tokens
    const int64_t n_used = ids->ne[0];
    const int64_t n_slots = ne12*n_used;

    if (moe_iq_dec_ok(ggml_cuda_info().devices[ggml_cuda_get_device()].cc, src0, src1, ids, dst) &&
            (!up_src0 || (up_src0->type == src0->type && src0->type != GGML_TYPE_IQ4_XS))) {
        const bool    bcast    = ne11 == 1;
        moe_direct_t dr;
        dr.ids       = (const int32_t *) ids->data;
        dr.s1        = ids->nb[1] / sizeof(int32_t);
        dr.y_div     = bcast ? (int) n_used : 1;
        dr.glu_limit = glu_limit;
        // the activations are quantized by a separate launch: quantizing them in each block (one launch fewer per
        // projection) measured slower in the model, 61.5 -> 58.2 t/s (every block's weight stream starts later)
        const int64_t ne10_pad = GGML_PAD(ne00, 256);
        const int64_t s_col    = ne10_pad / QK8_1;
        ggml_cuda_pool_alloc<block_q8_1> yd(ctx.pool(), (bcast ? ne12 : n_slots)*s_col);
        moe_vec_quantize_gather<<<dim3((ne10_pad + 255)/256, bcast ? ne12 : n_slots), 256, 0, stream>>>((const float *) src1->data,
            nullptr, yd.get(), ne00, ne10_pad, (bcast ? src1->nb[2] : src1->nb[1]) / sizeof(float));
        // GGML_CUDA_MOE_IQ_FLAT: 1 (default) the flat grid (1..2 tokens) for iq2_xs / iq3_xxs / iq4_xs (GLM-5.3 Q2 TP4 tg
        // 65.8 -> 67.6 t/s) but not iq2_xxs (DeepSeek V4: 43.1 -> 42.6-43.0), 2 every type, 0 off;
        // GGML_CUDA_MOE_IQ_FLAT_WHICH: 1 (default) the gate/up pair only (the down is neutral), 0 both, 2 the down only
        static const int flat_env   = moe_vec_env("GGML_CUDA_MOE_IQ_FLAT", 1);
        static const int flat_which = moe_vec_env("GGML_CUDA_MOE_IQ_FLAT_WHICH", 1);
        const bool flat_type  = flat_env == 2 || (flat_env == 1 && src0->type != GGML_TYPE_IQ2_XXS);
        const bool flat_which_ok = flat_which == 0 || (flat_which == 1 && up_src0) || (flat_which == 2 && !up_src0);
        static const int flat_max_tok = moe_vec_env("GGML_CUDA_MOE_IQ_FLAT_MAX_TOK", 2); // GLM pp2 105.6 -> 107.2, pp4 153.2 -> 152.3
        if (flat_type && flat_which_ok && ne12 <= flat_max_tok && bcast && n_used <= 64) {
            static const int flat_lpr = moe_vec_env("GGML_CUDA_MOE_IQ_FLAT_LPR", 16);
            static const int flat_bpc = moe_vec_env("GGML_CUDA_MOE_IQ_FLAT_BPC", 2);
            const ggml_tensor * u = up_src0 ? up_src0 : src0;
            const size_t shm = (size_t) ne12*(ne00 / QK8_1)*sizeof(block_q8_1);
            const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
            auto go_flat = [&](auto kern, int l) {
                const int64_t n_tasks = ((ne01 + 256/l - 1)/(256/l))*n_used*ne12;
                const int grid = (int) std::min<int64_t>(n_tasks, (int64_t) flat_bpc*nsm);
                kern<<<grid, 256, shm, stream>>>((const char *) src0->data, (const char *) u->data, yd.get(), (float *) dst->data,
                    (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], u->nb[1], u->nb[2], (int) n_used, dst->nb[1] / sizeof(float), dr,
                    (int) ne12, dst->nb[2] / sizeof(float), s_col);
            };
#define MOE_IQ_FLAT_L(t, glu) switch (flat_lpr) { case 8: go_flat(moe_coal_iq_flat<t, glu, 8>, 8); break; \
            case 32: go_flat(moe_coal_iq_flat<t, glu, 32>, 32); break; default: go_flat(moe_coal_iq_flat<t, glu, 16>, 16); break; }
            if (src0->type == GGML_TYPE_IQ2_XS) {
                if (up_src0) { MOE_IQ_FLAT_L(GGML_TYPE_IQ2_XS, true) } else { MOE_IQ_FLAT_L(GGML_TYPE_IQ2_XS, false) }
            } else if (src0->type == GGML_TYPE_IQ4_XS) {
                MOE_IQ_FLAT_L(GGML_TYPE_IQ4_XS, false)
            } else if (src0->type == GGML_TYPE_IQ2_XXS) {
                if (up_src0) { MOE_IQ_FLAT_L(GGML_TYPE_IQ2_XXS, true) } else { MOE_IQ_FLAT_L(GGML_TYPE_IQ2_XXS, false) }
            } else {
                if (up_src0) { MOE_IQ_FLAT_L(GGML_TYPE_IQ3_XXS, true) } else { MOE_IQ_FLAT_L(GGML_TYPE_IQ3_XXS, false) }
            }
#undef MOE_IQ_FLAT_L
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        // lanes per row (GGML_CUDA_MOE_IQ_LPR: 8 / 16 / 32, GGML_CUDA_MOE_IQ_LPR_GLU for the gate/up pair) and row groups
        // per block (GGML_CUDA_MOE_IQ_RG: 1 / 2 / 4), from
        // the TP4 sweep (us at 1/2/4 tokens): gate iq2_xs 512 x 4096, LPR 8 RG 1: 24/33/54 (RG 4: 47/54/69, too few
        // blocks for 512-row experts); down iq3_xxs 4096 x 512, LPR 8 RG 4: 23/34/60 (RG 1: 23/38/70; LPR 16 RG 1: 32/58/111)
        static const int lpr_env     = moe_vec_env("GGML_CUDA_MOE_IQ_LPR", 0);
        static const int lpr_glu_env = moe_vec_env("GGML_CUDA_MOE_IQ_LPR_GLU", 0);
        const int lpr = up_src0 && lpr_glu_env ? lpr_glu_env : lpr_env ? lpr_env : 8;
        const ggml_tensor * u = up_src0 ? up_src0 : src0;
        const size_t shm = (ne00 / QK8_1)*sizeof(block_q8_1);
        static const int rg_env = moe_vec_env("GGML_CUDA_MOE_IQ_RG", 0);
        const int rg = rg_env ? rg_env : (int) std::min<int64_t>(4, std::max<int64_t>(1, ne01/1024));
        auto go = [&](auto kern, int l, int r) {
            const int rows_pb = (256/l)*r;
            kern<<<dim3((ne01 + rows_pb - 1)/rows_pb, n_slots), 256, shm, stream>>>((const char *) src0->data, (const char *) u->data,
                yd.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], u->nb[1], u->nb[2],
                s_col, (int) n_used, dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), dr);
        };
#define MOE_IQ_RG(t, glu, l) switch (rg) { case 2: go(moe_coal_iq<t, glu, l, 2>, l, 2); break; case 4: go(moe_coal_iq<t, glu, l, 4>, l, 4); break; \
            default: go(moe_coal_iq<t, glu, l, 1>, l, 1); break; }
#define MOE_IQ_L(t, glu) switch (lpr) { case 16: MOE_IQ_RG(t, glu, 16) break; case 32: MOE_IQ_RG(t, glu, 32) break; \
            default: MOE_IQ_RG(t, glu, 8) break; }
        if (src0->type == GGML_TYPE_IQ2_XS) {
            if (up_src0) { MOE_IQ_L(GGML_TYPE_IQ2_XS, true) } else { MOE_IQ_L(GGML_TYPE_IQ2_XS, false) }
        } else if (src0->type == GGML_TYPE_IQ4_XS) {
            MOE_IQ_L(GGML_TYPE_IQ4_XS, false)
        } else if (src0->type == GGML_TYPE_IQ2_XXS) {
            if (up_src0) { MOE_IQ_L(GGML_TYPE_IQ2_XXS, true) } else { MOE_IQ_L(GGML_TYPE_IQ2_XXS, false) }
        } else {
            if (up_src0) { MOE_IQ_L(GGML_TYPE_IQ3_XXS, true) } else { MOE_IQ_L(GGML_TYPE_IQ3_XXS, false) }
        }
#undef MOE_IQ_L
#undef MOE_IQ_RG
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (moe_q2k_ok(ggml_cuda_info().devices[ggml_cuda_get_device()].cc, src0, src1, ids, dst)) {
        GGML_ASSERT(up_src0 == nullptr);
        ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
        ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
        ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), ne02 + 1);
        {
            const int si1  = ids->nb[1] / sizeof(int32_t);
            const int sis1 = src1->nb[2] / src1->nb[1];
            ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
                ne02, ne12, n_used, ne11, si1, sis1, /*write_inverse =*/ false, stream, &ctx.pool());
        }
        const int64_t ne10_padded = GGML_PAD(ne00, 256);
        const int64_t s_col_y = ne10_padded / QK8_1;
        // gate/up (one activation row per token): quantize each token once, look rows up through ids_src1
        const bool dedup = ne11 == 1 && src1->nb[2] == src1->nb[1];
        const int64_t n_yrows = dedup ? ne12 : n_slots;
        ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), n_yrows*s_col_y);
        ggml_cuda_pool_alloc<int16_t>    y16(ctx.pool(), n_yrows*(ne10_padded/16));
        moe_vec_quantize_gather_s16<<<dim3((ne10_padded + 255)/256, n_yrows), 256, 0, stream>>>((const float *) src1->data,
            dedup ? nullptr : ids_src1.get(), y.get(), y16.get(), ne00, ne10_padded, src1->nb[1] / sizeof(float));
        // 8-token tiles, 4 above 8 tokens per expert (more blocks for the latency-bound kernel; 16 spilled)
        static const int j_env = moe_vec_env("GGML_CUDA_MOE_Q2K_J", 0);
        const int64_t avg = (n_slots + ne02 - 1) / ne02;
        int J = j_env ? j_env : (avg <= 8 ? 8 : 4);
        // decode-sized batches (<= 4 tokens: about one token per expert tile): single-column tiles, and for q2_K the
        // K-split waves (6 experts x 576 rows at TP4 are only ~54 row waves)
        int KS = 1;
        if (!j_env && ne12 <= 4 && src0->type != GGML_TYPE_IQ2_XXS) {
            J = 1;
            if (src0->type == GGML_TYPE_Q2_K) {
                KS = moe_rowlane_q2k_ks(((ne01 + 63)/64)*std::min<int64_t>(n_slots, ne02), ne00/QK_K);
            }
        }
        const int max_tiles = (int) std::min<int64_t>(n_slots, n_slots / J + ne02);
        ggml_cuda_pool_alloc<int2>    tiles(ctx.pool(), max_tiles);
        ggml_cuda_pool_alloc<int32_t> n_tiles(ctx.pool(), 1);
        moe_vec_tile_map<<<1, 256, 256*sizeof(int), stream>>>(bounds.get(), tiles.get(), n_tiles.get(), (int) ne02, J);
        const int64_t s1 = dst->nb[1] / sizeof(float);
        const int64_t s2 = dst->nb[2] / sizeof(float);
        const dim3 grid((ne01 + 63)/64, max_tiles);
#define MOE_Q2K_KS(j, ks) moe_rowlane_q2k<j, false, ks><<<grid, 64*ks, 0, stream>>>((const char *) src0->data, y.get(), \
            y16.get(), tiles.get(), n_tiles.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, \
            src0->nb[1], src0->nb[2], s_col_y, (int) n_used, s1, s2, dedup ? ids_src1.get() : nullptr, 0)
#define MOE_Q2K(j) MOE_Q2K_KS(j, 1)
#define MOE_IQ2XXS(j) moe_rowlane_iq2xxs<j><<<grid, 64, 0, stream>>>((const char *) src0->data, y.get(), tiles.get(), \
            n_tiles.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], \
            s_col_y, (int) n_used, s1, s2, dedup ? ids_src1.get() : nullptr, 0)
#define MOE_Q3K(j) moe_rowlane_q3k<j><<<grid, 64, 0, stream>>>((const char *) src0->data, y.get(), tiles.get(), \
            n_tiles.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], \
            s_col_y, (int) n_used, s1, s2, dedup ? ids_src1.get() : nullptr, 0)
        if (src0->type == GGML_TYPE_IQ2_XXS) {
            switch (J) {
                case 4:  MOE_IQ2XXS(4);  break;
                case 16: MOE_IQ2XXS(16); break;
                default: MOE_IQ2XXS(8);  break;
            }
        } else if (src0->type == GGML_TYPE_Q3_K) {
            switch (J) {
                case 1:  MOE_Q3K(1);  break;
                case 4:  MOE_Q3K(4);  break;
                case 16: MOE_Q3K(16); break;
                default: MOE_Q3K(8);  break;
            }
        } else if (J == 1) {
            switch (KS) {
                case 2:  MOE_Q2K_KS(1, 2); break;
                case 4:  MOE_Q2K_KS(1, 4); break;
                case 8:  MOE_Q2K_KS(1, 8); break;
                default: MOE_Q2K_KS(1, 1); break;
            }
        } else {
            switch (J) {
                case 4:  MOE_Q2K(4);  break;
                case 16: MOE_Q2K(16); break;
                default: MOE_Q2K(8);  break;
            }
        }
#undef MOE_Q3K
#undef MOE_IQ2XXS
#undef MOE_Q2K
#undef MOE_Q2K_KS
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // decode: no expert sort / tile map, one block per slot reading ids directly, K split over up to dec_ks waves per
    // block (GGML_CUDA_MOE_DEC_DIRECT=0: through the sorted tile path)
    {
        const int cc_d = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        static const int dec_max    = moe_vec_env("GGML_CUDA_MOE_DEC_MAX_TOKENS", 8);
        static const int dec_ks     = std::min(MOE_DEC_KS, moe_vec_env("GGML_CUDA_MOE_DEC_KS", MOE_DEC_KS_DEFAULT));
        // (up to 4 tokens: with the long-span kernel the per-slot path beats the sorted tiles, whose tokens share an
        // expert's weight reads, even on real text: MTP verify 49.1/66.3 -> 51.2/67.5 t/s; with the older kernels the
        // tiles won at 4 tokens, so GGML_CUDA_MOE_LS_RG=0 should go with GGML_CUDA_MOE_DEC_DIRECT_MAX_TOKENS=1)
        static const int direct_env = moe_vec_env("GGML_CUDA_MOE_DEC_DIRECT", 1);
        static const int direct_max = moe_vec_env("GGML_CUDA_MOE_DEC_DIRECT_MAX_TOKENS", 8);
        static const bool q8k_env_d = moe_vec_env("GGML_CUDA_MOE_ROWLANE_Q8K", 0) != 0;
        const bool bcast   = ne11 == 1; // gate/up: every slot of a token reads the token's row
        const bool rows_ok = bcast || src1->nb[2] == ne11*src1->nb[1];
        if (direct_env && dec_ks > 1 && GGML_CUDA_CC_IS_GCN(cc_d) && moe_rowlane_ok(src0) && !q8k_env_d &&
                ne12 <= std::min(dec_max, 16) && ne12 <= direct_max && rows_ok && (!up_src0 || bcast) && ids->nb[0] == sizeof(int32_t)) {
            const int64_t ne10_pad = GGML_PAD(ne00, 256);
            const int64_t s_col    = ne10_pad / QK8_1;
            const int64_t n_rows_y = bcast ? ne12 : n_slots;
            moe_direct_t dr;
            dr.ids       = (const int32_t *) ids->data;
            dr.s1        = ids->nb[1] / sizeof(int32_t);
            dr.y_div     = bcast ? (int) n_used : 1;
            dr.glu_limit = glu_limit;
            const int64_t s1 = dst->nb[1] / sizeof(float);
            const int64_t s2 = dst->nb[2] / sizeof(float);

            ggml_cuda_pool_alloc<block_q8_1> yd(ctx.pool(), n_rows_y*s_col);
            moe_vec_quantize_gather<<<dim3((ne10_pad + 255)/256, n_rows_y), 256, 0, stream>>>((const float *) src1->data, nullptr,
                yd.get(), ne00, ne10_pad, (bcast ? src1->nb[2] : src1->nb[1]) / sizeof(float));
            const dim3 grid_d((ne01 + 63)/64, n_slots);
            auto nsl = [&](const ggml_tensor * w) {
                const int64_t steps = w->type == GGML_TYPE_Q5_1 || w->type == GGML_TYPE_Q8_0 ? ne00/(4*ggml_blck_size(w->type)) : ne00/QK_K;
                return (int) std::max<int64_t>(1, std::min<int64_t>(dec_ks, steps));
            };
            const bool kq = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K;
            // coalesced kernels (GGML_CUDA_MOE_DEC_COAL=0: K-split row-lane): per kind switchable, LPR lanes per row; the
            // gate/up pair stays on the row-lane kernel by default (the coalesced GLU variant measured slower in the model)
            static const int coal_env = moe_vec_env("GGML_CUDA_MOE_DEC_COAL", 1);
            static const int coal_glu = moe_vec_env("GGML_CUDA_MOE_COAL_GLU", 0);
            static const int coal_kq  = moe_vec_env("GGML_CUDA_MOE_COAL_KQ", 1);
            static const int coal_q51 = moe_vec_env("GGML_CUDA_MOE_COAL_Q51", 1);
            // lanes per row: 16 (Qwen, K 2560) unless set; GLM-5-Next (TP4 decode): gate/up K 4096 -> 32, q5_K down K 512 -> 8
            // (tg 48.9 -> 50.1 t/s)
            static const int lpr_glu_env = moe_vec_env("GGML_CUDA_MOE_COAL_GLU_LPR", 0);
            static const int lpr_kq_env  = moe_vec_env("GGML_CUDA_MOE_COAL_KQ_LPR", 0);
            const int lpr_glu = lpr_glu_env ? lpr_glu_env : (ne00 >= 4096 ? 32 : 16);
            const int lpr_kq  = lpr_kq_env  ? lpr_kq_env  : (ne00 <= 1024 ? 8 : 16);
            static const int lpr_q51  = moe_vec_env("GGML_CUDA_MOE_COAL_Q51_LPR", 8);
            const bool ukq = !up_src0 || up_src0->type == GGML_TYPE_Q4_K || up_src0->type == GGML_TYPE_Q5_K;
            // long-span coalesced kernel with the activation row in shared memory (GGML_CUDA_MOE_LS_RG=0: previous
            // kernels): gfx906 decode 39.7 -> 41.3 t/s with 1 row group of 16 rows (longer spans were slower here)
            static const int ls_rg = moe_vec_env("GGML_CUDA_MOE_LS_RG", 1);
            if (ls_rg > 0 && kq && ukq) {
                const ggml_tensor * u = up_src0 ? up_src0 : src0;
                const int lpr = up_src0 ? lpr_glu : lpr_kq;
                const size_t shm = (ne00 / QK8_1)*sizeof(block_q8_1);
                static const int dedup_env = moe_vec_env("GGML_CUDA_MOE_DEDUP", 0);
                const bool dd = dedup_env && ne12 >= 2 && ne12 <= MOE_DD_MAX && n_slots <= 256;
                auto go = [&](auto kern, auto kern_dd, int l, int rg) {
                    const int rows_pb = (256/l)*rg;
                    if (dd) {
                        kern_dd<<<dim3((ne01 + rows_pb - 1)/rows_pb, n_slots), 256, shm*ne12, stream>>>((const char *) src0->data, (const char *) u->data,
                            yd.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], u->nb[1], u->nb[2], s_col,
                            (int) n_used, s1, s2, dr);
                        return;
                    }
                    kern<<<dim3((ne01 + rows_pb - 1)/rows_pb, n_slots), 256, shm, stream>>>((const char *) src0->data, (const char *) u->data,
                        yd.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], u->nb[1], u->nb[2], s_col,
                        (int) n_used, s1, s2, dr);
                };
#define MOE_LS_RG(g5, u5, glu, l) switch (ls_rg) { case 1: go(moe_coal_kq_ls<g5, u5, glu, l, 1>, moe_coal_kq_ls_dd<g5, u5, glu, l, 1>, l, 1); break; \
                    default: go(moe_coal_kq_ls<g5, u5, glu, l, 4>, moe_coal_kq_ls_dd<g5, u5, glu, l, 4>, l, 4); break; }
#define MOE_LS(g5, u5, glu) switch (lpr) { case 8: MOE_LS_RG(g5, u5, glu, 8) break; case 32: MOE_LS_RG(g5, u5, glu, 32) break; \
                    default: MOE_LS_RG(g5, u5, glu, 16) break; }
                const bool g5 = src0->type == GGML_TYPE_Q5_K, u5 = u->type == GGML_TYPE_Q5_K;
                if (up_src0) {
                    if (g5 && u5) { MOE_LS(true, true, true) } else if (g5) { MOE_LS(true, false, true) }
                    else if (u5) { MOE_LS(false, true, true) } else { MOE_LS(false, false, true) }
                } else {
                    if (g5) { MOE_LS(true, true, false) } else { MOE_LS(false, false, false) }
                }
#undef MOE_LS
#undef MOE_LS_RG
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (coal_env && kq && ukq && (up_src0 ? coal_glu : coal_kq)) {
                const ggml_tensor * u = up_src0 ? up_src0 : src0;
                const int lpr = up_src0 ? lpr_glu : lpr_kq;
                auto go = [&](auto kern, int l) {
                    kern<<<dim3((ne01 + 256/l - 1)/(256/l), n_slots), 256, 0, stream>>>((const char *) src0->data, (const char *) u->data,
                        yd.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], u->nb[1], u->nb[2], s_col,
                        (int) n_used, s1, s2, dr);
                };
#define MOE_CO_L(g5, u5, glu) switch (lpr) { case 8: go(moe_coal_kq<g5, u5, glu, 8>, 8); break; \
                    case 32: go(moe_coal_kq<g5, u5, glu, 32>, 32); break; default: go(moe_coal_kq<g5, u5, glu, 16>, 16); break; }
                const bool g5 = src0->type == GGML_TYPE_Q5_K, u5 = u->type == GGML_TYPE_Q5_K;
                if (up_src0) {
                    if (g5 && u5) { MOE_CO_L(true, true, true) } else if (g5) { MOE_CO_L(true, false, true) }
                    else if (u5) { MOE_CO_L(false, true, true) } else { MOE_CO_L(false, false, true) }
                } else {
                    if (g5) { MOE_CO_L(true, true, false) } else { MOE_CO_L(false, false, false) }
                }
#undef MOE_CO_L
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (coal_env && coal_q51 && !up_src0 && src0->type == GGML_TYPE_Q5_1) {
                static const int dedup_env = moe_vec_env("GGML_CUDA_MOE_DEDUP", 0);
                const bool dd = dedup_env && ne12 >= 2 && ne12 <= MOE_DD_MAX && n_slots <= 256;
                auto go = [&](auto kern, auto kern_dd, int l) {
                    if (dd) {
                        kern_dd<<<dim3((ne01 + 256/l - 1)/(256/l), n_slots), 256, 0, stream>>>((const char *) src0->data, yd.get(),
                            (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], s_col, (int) n_used, s1, s2, dr);
                        return;
                    }
                    kern<<<dim3((ne01 + 256/l - 1)/(256/l), n_slots), 256, 0, stream>>>((const char *) src0->data, yd.get(),
                        (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], s_col, (int) n_used, s1, s2, dr);
                };
                switch (lpr_q51) {
                    case 16: go(moe_coal_q51<16>, moe_coal_q51_dd<16>, 16); break;
                    default: go(moe_coal_q51<8>, moe_coal_q51_dd<8>, 8);   break;
                }
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (up_src0 && kq) {
                const dim3 blk(64, nsl(src0));
#define MOE_DG(tg, tu) moe_rowlane_glu<tg, tu, 1, MOE_DEC_KS><<<grid_d, blk, 0, stream>>>((const char *) src0->data, \
                    (const char *) up_src0->data, yd.get(), nullptr, nullptr, nullptr, nullptr, (float *) dst->data, (int) ne00, (int) ne01, \
                    src0->nb[1], src0->nb[2], up_src0->nb[1], up_src0->nb[2], s_col, (int) n_used, s1, s2, nullptr, dr)
                const bool g5 = src0->type == GGML_TYPE_Q5_K, u5 = up_src0->type == GGML_TYPE_Q5_K;
                if (g5 && u5) { MOE_DG(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K); } else if (g5) { MOE_DG(GGML_TYPE_Q5_K, GGML_TYPE_Q4_K); }
                else if (u5) { MOE_DG(GGML_TYPE_Q4_K, GGML_TYPE_Q5_K); } else { MOE_DG(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K); }
#undef MOE_DG
            } else {
                auto launch_d = [&](const ggml_tensor * w, const int epi) {
                    const dim3 blk(64, nsl(w));
#define MOE_DR(t) moe_rowlane<t, 1, false, MOE_DEC_KS><<<grid_d, blk, 0, stream>>>((const char *) w->data, yd.get(), nullptr, nullptr, \
                        nullptr, nullptr, (float *) dst->data, (int) ne00, (int) ne01, w->nb[1], w->nb[2], s_col, (int) n_used, s1, s2, \
                        nullptr, epi, 1, dr)
                    switch (w->type) {
                        case GGML_TYPE_Q5_1: MOE_DR(GGML_TYPE_Q5_1); break;
                        case GGML_TYPE_Q8_0: MOE_DR(GGML_TYPE_Q8_0); break;
                        case GGML_TYPE_Q5_K: MOE_DR(GGML_TYPE_Q5_K); break;
                        default:             MOE_DR(GGML_TYPE_Q4_K); break;
                    }
#undef MOE_DR
                };
                launch_d(src0, 0);
                if (up_src0) {
                    launch_d(up_src0, 1);
                }
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
    }

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_slots);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), ne02 + 1);
    {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
            ne02, ne12, n_used, ne11, si1, sis1, /*write_inverse =*/ false, stream, &ctx.pool());
    }

    const int64_t ne10_padded = GGML_PAD(ne00, 256);
    const int64_t s_col_y = ne10_padded / QK8_1;
    // the K-quant row-lane path quantizes its own activations (block_q8_moek)
    const int cc_q = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool skip_q8_1 = GGML_CUDA_CC_IS_GCN(cc_q) && moe_rowlane_ok(src0) && moe_vec_env("GGML_CUDA_MOE_ROWLANE_Q8K", 0) != 0 &&
        (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K);
    // gate/up: every slot of a token reads the same activation row, so the K-quant row-lane path quantizes each token once
    // (ne12 rows instead of ne12*n_used) and looks the row up through ids_src1 (GGML_CUDA_MOE_DEDUP=0: per slot)
    static const bool dedup_env = moe_vec_env("GGML_CUDA_MOE_DEDUP", 1) != 0;
    const bool dedup = dedup_env && !skip_q8_1 && ne11 == 1 && src1->nb[2] == src1->nb[1] && GGML_CUDA_CC_IS_GCN(cc_q) &&
        moe_rowlane_ok(src0) && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K);
    const int64_t n_yrows = dedup ? ne12 : n_slots;
    ggml_cuda_pool_alloc<block_q8_1> y(ctx.pool(), skip_q8_1 ? 1 : n_yrows*s_col_y);
    if (!skip_q8_1) {
        const int64_t s_col = src1->nb[1] / sizeof(float);
        const dim3 grid((ne10_padded + 255)/256, n_yrows);
        moe_vec_quantize_gather<<<grid, 256, 0, stream>>>((const float *) src1->data, dedup ? nullptr : ids_src1.get(), y.get(),
            ne00, ne10_padded, s_col);
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (GGML_CUDA_CC_IS_GCN(cc) && moe_rowlane_ok(src0)) {
        static const bool q8k_env = moe_vec_env("GGML_CUDA_MOE_ROWLANE_Q8K", 0) != 0; // opt-in: no gain in the model (latency-bound)
        const bool q8k = q8k_env && (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K);
        static const int j_env = moe_vec_env("GGML_CUDA_MOE_ROWLANE_J", 0);
        static const int tile_mult = moe_vec_env("GGML_CUDA_MOE_TILE_CHUNKS", 1); // J-chunks per tile; >1 measured slower (load imbalance)
        // decode (few tokens): tiles as wide as the token count and K split over up to dec_ks waves per block, since
        // one wave per 64 rows of each active expert leaves the GPU mostly idle (GGML_CUDA_MOE_DEC_KS=1 off)
        static const int dec_max = moe_vec_env("GGML_CUDA_MOE_DEC_MAX_TOKENS", 8);
        static const int dec_ks  = std::min(MOE_DEC_KS, moe_vec_env("GGML_CUDA_MOE_DEC_KS", MOE_DEC_KS_DEFAULT));
        const bool dec = !j_env && !q8k && tile_mult == 1 && ne12 <= std::min(dec_max, 8) && dec_ks > 1;
        const int64_t avg_rl = (n_slots + ne02 - 1) / ne02;
        const int J = dec ? (ne12 <= 1 ? 1 : ne12 <= 2 ? 2 : ne12 <= 4 ? 4 : 8) : j_env ? j_env : (avg_rl <= 8 ? 8 : 16);
        const int tile_cols = J*std::max(1, tile_mult);
        // (every tile holds at least one slot)
        const int max_tiles_rl = (int) std::min<int64_t>(n_slots, n_slots / tile_cols + ne02);
        ggml_cuda_pool_alloc<int2>    tiles_rl(ctx.pool(), max_tiles_rl);
        ggml_cuda_pool_alloc<int32_t> n_tiles_rl(ctx.pool(), 1);
        moe_vec_tile_map<<<1, 256, 256*sizeof(int), stream>>>(bounds.get(), tiles_rl.get(), n_tiles_rl.get(), (int) ne02, tile_cols);
        ggml_cuda_pool_alloc<block_q8_moek> yk(ctx.pool());
        if (q8k) {
            yk.alloc(n_slots*(ne00/QK_K));
            const int64_t s_col = src1->nb[1] / sizeof(float);
            moe_vec_quantize_gather_q8k<<<dim3(ne00/QK_K, n_slots), QK_K, 0, stream>>>((const float *) src1->data, ids_src1.get(), yk.get(), ne00, s_col);
        }
        const int64_t s1 = dst->nb[1] / sizeof(float);
        const int64_t s2 = dst->nb[2] / sizeof(float);
        // up_src0 (fused gate/up + SwiGLU): the same preparation serves both projections; the second launch combines
        const dim3 grid_rl((ne01 + 63)/64, max_tiles_rl);
        if (dec) {
            // K steps of each kernel: 4 blocks for q5_1/q8_0, one super-block for the K-quants
            auto nsl = [&](const ggml_tensor * w) {
                const int64_t steps = w->type == GGML_TYPE_Q5_1 || w->type == GGML_TYPE_Q8_0 ? ne00/(4*ggml_blck_size(w->type)) : ne00/QK_K;
                return (int) std::max<int64_t>(1, std::min<int64_t>(dec_ks, steps));
            };
#define MOE_DJ(M, ...) if (J == 1) { M(1, __VA_ARGS__); } else if (J == 2) { M(2, __VA_ARGS__); } else if (J == 4) { M(4, __VA_ARGS__); } \
            else { M(8, __VA_ARGS__); }
            if (up_src0 && dedup) {
                const dim3 blk(64, nsl(src0));
#define MOE_DG(j, tg, tu) moe_rowlane_glu<tg, tu, j, MOE_DEC_KS><<<grid_rl, blk, 0, stream>>>((const char *) src0->data, \
                    (const char *) up_src0->data, y.get(), tiles_rl.get(), n_tiles_rl.get(), bounds.get(), ids_dst.get(), (float *) dst->data, \
                    (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], up_src0->nb[1], up_src0->nb[2], s_col_y, (int) n_used, s1, s2, ids_src1.get(), dl)
                const bool g5 = src0->type == GGML_TYPE_Q5_K, u5 = up_src0->type == GGML_TYPE_Q5_K;
                if (g5 && u5) { MOE_DJ(MOE_DG, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K) } else if (g5) { MOE_DJ(MOE_DG, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K) }
                else if (u5) { MOE_DJ(MOE_DG, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K) } else { MOE_DJ(MOE_DG, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K) }
#undef MOE_DG
            } else {
                auto launch_d = [&](const ggml_tensor * w, const int epi) {
                    const dim3 blk(64, nsl(w));
#define MOE_DR(j, t) moe_rowlane<t, j, false, MOE_DEC_KS><<<grid_rl, blk, 0, stream>>>((const char *) w->data, y.get(), tiles_rl.get(), \
                        n_tiles_rl.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, w->nb[1], w->nb[2], s_col_y, \
                        (int) n_used, s1, s2, dedup ? ids_src1.get() : nullptr, epi, J, dl)
                    switch (w->type) {
                        case GGML_TYPE_Q5_1: MOE_DJ(MOE_DR, GGML_TYPE_Q5_1) break;
                        case GGML_TYPE_Q8_0: MOE_DJ(MOE_DR, GGML_TYPE_Q8_0) break;
                        case GGML_TYPE_Q5_K: MOE_DJ(MOE_DR, GGML_TYPE_Q5_K) break;
                        default:             MOE_DJ(MOE_DR, GGML_TYPE_Q4_K) break;
                    }
#undef MOE_DR
                };
                launch_d(src0, 0);
                if (up_src0) {
                    launch_d(up_src0, 1);
                }
            }
#undef MOE_DJ
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        auto launch = [&](const ggml_tensor * w, const int epi) {
#define MOE_RL(t, j) moe_rowlane<t, j><<<grid_rl, 64, 0, stream>>>((const char *) w->data, y.get(), tiles_rl.get(), n_tiles_rl.get(), \
            bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, w->nb[1], w->nb[2], s_col_y, (int) n_used, s1, s2, \
            dedup ? ids_src1.get() : nullptr, epi, tile_cols, dl)
#define MOE_RL_J(t) if (J == 8) { MOE_RL(t, 8); } else if (J == 4) { MOE_RL(t, 4); } else { MOE_RL(t, 16); }
        if (w->type == GGML_TYPE_Q5_1) {
            MOE_RL_J(GGML_TYPE_Q5_1)
        } else if (w->type == GGML_TYPE_Q8_0) {
            MOE_RL_J(GGML_TYPE_Q8_0)
        } else if (q8k) {
#define MOE_RLK(t, j) moe_rowlane<t, j, true><<<grid_rl, 64, 0, stream>>>((const char *) w->data, (const block_q8_1 *) yk.get(), tiles_rl.get(), \
            n_tiles_rl.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, src0->nb[1], src0->nb[2], ne00/QK_K, \
            (int) n_used, s1, s2, nullptr, 0, tile_cols)
            if (w->type == GGML_TYPE_Q5_K) {
                if (J == 8) { MOE_RLK(GGML_TYPE_Q5_K, 8); } else if (J == 4) { MOE_RLK(GGML_TYPE_Q5_K, 4); } else { MOE_RLK(GGML_TYPE_Q5_K, 16); }
            } else {
                if (J == 8) { MOE_RLK(GGML_TYPE_Q4_K, 8); } else if (J == 4) { MOE_RLK(GGML_TYPE_Q4_K, 4); } else { MOE_RLK(GGML_TYPE_Q4_K, 16); }
            }
#undef MOE_RLK
        } else if (w->type == GGML_TYPE_Q5_K) {
            MOE_RL_J(GGML_TYPE_Q5_K)
        } else {
            MOE_RL_J(GGML_TYPE_Q4_K)
        }
#undef MOE_RL_J
#undef MOE_RL
        };
        // fused gate/up kernel: one pass, J=8 tiles only (GGML_CUDA_MOE_GLU_FUSED=0: two launches)
        static const bool glu_fused_env = moe_vec_env("GGML_CUDA_MOE_GLU_FUSED", 1) != 0;
        if (up_src0 && glu_fused_env && dedup && !q8k && J == 8 && tile_mult == 1) {
#define MOE_RLG(tg, tu) moe_rowlane_glu<tg, tu, 8><<<grid_rl, 64, 0, stream>>>((const char *) src0->data, (const char *) up_src0->data, \
                y.get(), tiles_rl.get(), n_tiles_rl.get(), bounds.get(), ids_dst.get(), (float *) dst->data, (int) ne00, (int) ne01, \
                src0->nb[1], src0->nb[2], up_src0->nb[1], up_src0->nb[2], s_col_y, (int) n_used, s1, s2, ids_src1.get(), dl)
            const bool g5 = src0->type == GGML_TYPE_Q5_K, u5 = up_src0->type == GGML_TYPE_Q5_K;
            if (g5 && u5) { MOE_RLG(GGML_TYPE_Q5_K, GGML_TYPE_Q5_K); } else if (g5) { MOE_RLG(GGML_TYPE_Q5_K, GGML_TYPE_Q4_K); }
            else if (u5) { MOE_RLG(GGML_TYPE_Q4_K, GGML_TYPE_Q5_K); } else { MOE_RLG(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K); }
#undef MOE_RLG
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        launch(src0, 0);
        if (up_src0) {
            launch(up_src0, 1);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(up_src0 == nullptr);

    static const int nc_env = moe_vec_env("GGML_CUDA_MOE_VEC_NC", 0);
    const int64_t avg = (n_slots + ne02 - 1) / ne02;
    const int NC = nc_env ? nc_env : (avg <= 4 ? 4 : avg <= 8 ? 8 : 16);

    const int max_tiles = (int) (n_slots / NC + ne02);
    ggml_cuda_pool_alloc<int2>    tiles(ctx.pool(), max_tiles);
    ggml_cuda_pool_alloc<int32_t> n_tiles(ctx.pool(), 1);
    moe_vec_tile_map<<<1, 256, 256*sizeof(int), stream>>>(bounds.get(), tiles.get(), n_tiles.get(), (int) ne02, NC);

    // T lanes per row: the largest power of 2 that leaves >= 4 work items per lane and divides the row evenly
    const int qk = ggml_blck_size(src0->type);
    const int qi = [&] {
        switch (src0->type) {
            case GGML_TYPE_Q4_0: return QI4_0; case GGML_TYPE_Q4_1: return QI4_1; case GGML_TYPE_Q5_0: return QI5_0;
            case GGML_TYPE_Q5_1: return QI5_1; case GGML_TYPE_Q8_0: return QI8_0; case GGML_TYPE_Q4_K: return QI4_K;
            case GGML_TYPE_Q5_K: return QI5_K; case GGML_TYPE_Q2_K: return QI2_K; case GGML_TYPE_IQ2_XXS: return QI2_XXS;
            default: return QI6_K;
        }
    }();
    const int vdr = src0->type == GGML_TYPE_Q6_K || src0->type == GGML_TYPE_Q2_K ? 1 : 2;
    const int items = (int) (ne00/qk) * (qi/vdr);
    static const int t_env = moe_vec_env("GGML_CUDA_MOE_VEC_T", 0);
    int T = 64;
    while (T > 4 && (items % T != 0 || items / T < 4)) {
        T /= 2;
    }
    if (t_env) {
        T = t_env;
    }

    const size_t ts0 = ggml_type_size(src0->type);
    const int64_t s01 = src0->nb[1] / ts0; // in blocks
    const int64_t s02 = src0->nb[2];       // in bytes
    const int64_t s1  = dst->nb[1] / sizeof(float);
    const int64_t s2  = dst->nb[2] / sizeof(float);

    const dim3 grid(1, max_tiles);
    const char * x = (const char *) src0->data;
    float * d = (float *) dst->data;
#define MOE_VEC_CASE(t) case t: moe_vec_launch_type<t>(NC, T, grid, x, y.get(), tiles.get(), n_tiles.get(), bounds.get(), ids_dst.get(), d, \
        (int) ne00, (int) ne01, s01, s02, s_col_y, (int) n_used, s1, s2, stream); break;
    switch (src0->type) {
        MOE_VEC_CASE(GGML_TYPE_Q4_0)
        MOE_VEC_CASE(GGML_TYPE_Q4_1)
        MOE_VEC_CASE(GGML_TYPE_Q5_0)
        MOE_VEC_CASE(GGML_TYPE_Q5_1)
        MOE_VEC_CASE(GGML_TYPE_Q8_0)
        MOE_VEC_CASE(GGML_TYPE_Q4_K)
        MOE_VEC_CASE(GGML_TYPE_Q5_K)
        MOE_VEC_CASE(GGML_TYPE_Q6_K)
        MOE_VEC_CASE(GGML_TYPE_Q2_K)
        MOE_VEC_CASE(GGML_TYPE_IQ2_XXS)
        default: GGML_ABORT("unsupported type");
    }
#undef MOE_VEC_CASE
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_vec_pair_supported(int cc, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu) {
    static const int env = moe_vec_env("GGML_CUDA_MOE_PAIR", 1);
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || moe_vec_env("GGML_CUDA_MOE_ROWLANE_Q8K", 0)) {
        return false;
    }
    const bool glu_ok = glu->op == GGML_OP_GLU && ggml_get_op_params_i32(glu, 1) == 0 &&
        (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP) &&
        glu->src[0] == gate && glu->src[1] == up &&
        gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID &&
        gate->src[1] == up->src[1] && gate->src[2] == up->src[2] && ggml_are_same_shape(gate->src[0], up->src[0]) &&
        ggml_are_same_shape(gate, glu) && ggml_are_same_stride(gate, glu) && ggml_is_contiguous(glu);
    // i-quant pairs of one type at decode sizes: moe_coal_iq with GLU
    if (glu_ok && gate->src[0]->type == up->src[0]->type && gate->src[0]->type != GGML_TYPE_IQ4_XS && gate->src[1]->ne[1] == 1 &&
            moe_iq_dec_ok(cc, gate->src[0], gate->src[1], gate->src[2], glu) && moe_iq_dec_ok(cc, up->src[0], up->src[1], up->src[2], glu)) {
        return true;
    }
    return glu->op == GGML_OP_GLU && ggml_get_op_params_i32(glu, 1) == 0 &&
        (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP) &&
        glu->src[0] == gate && glu->src[1] == up &&
        gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID &&
        gate->src[1] == up->src[1] && gate->src[2] == up->src[2] &&
        ggml_are_same_shape(gate->src[0], up->src[0]) && // types may differ (mixed quants): each launch uses its own
        ((gate->src[0]->type == GGML_TYPE_Q4_K || gate->src[0]->type == GGML_TYPE_Q5_K) ==
         (up->src[0]->type   == GGML_TYPE_Q4_K || up->src[0]->type   == GGML_TYPE_Q5_K)) &&
        ggml_are_same_shape(gate, glu) && ggml_are_same_stride(gate, glu) && ggml_is_contiguous(glu) &&
        moe_rowlane_ok(gate->src[0]) && moe_rowlane_ok(up->src[0]) &&
        ggml_cuda_moe_vec_supported(cc, gate->src[0], gate->src[1], gate->src[2], glu);
}

void ggml_cuda_moe_vec_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu) {
    const float limit = ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_CLAMP ? ggml_get_op_params_f32(glu, 3) : INFINITY;
    ggml_cuda_moe_vec(ctx, gate->src[0], gate->src[1], gate->src[2], glu, up->src[0], limit);
}


// ---- single-token MoE FFN (routed experts + shared expert), see ggml_cuda_moe1_args ----

// gate/up: workgroup = 32 rows of one expert (e = blockIdx.y; e == n_used: the shared expert), 8 waves: wave w takes
// rows 8*(w & 3) .. + 7 (two groups of 4) of the gate (w < 4) or up (w >= 4) matrix. Every workgroup quantizes the
// token into LDS: int8 in blocks of 32 values (as q8_1; 128-value blocks let a lane accumulate its 4 sub-blocks in
// integers but cost 0.5-0.8% decode PPL), dword-swizzled against bank conflicts, plus the per-16 sub-block sums of the dequantized
// values. Lane = (row r of 4, super-block b of 4, quarter c): each lane loads 16 qs bytes (dwordx4, 4 lanes per
// super-block's 64 bytes) = 64 values: 4 sub-blocks 8*n + 2*j + hh (n = c/2, hh = c%2, 2-bit shift j), elements
// 128*n + 32*j + 16*hh + 0..15. The two scale dwords 8*n .. 8*n + 7 hold all 4 sub-blocks' scale/min bytes. Output: f32
// SwiGLU-clamp(gate, up).
static constexpr int MOE1_ROWS_WG = 32;

static __device__ __forceinline__ int moe1_swz(const int w) {
    return w ^ (((w >> 5) & 3) << 3);
}
template <int ctrl> static __device__ __forceinline__ int moe1_dpp(const int v) {
    return __builtin_amdgcn_mov_dpp(v, ctrl, 0xF, 0xF, true);
}
// max over aligned groups of 8 lanes (one 32-value activation block of float4 lanes): quad_perm xor 1 / xor 2, then the
// other quad of the 8 (row_ror 4 pairs quad k with quad k+-1, whose max within an aligned 8 is the same pair)
static __device__ __forceinline__ float moe1_max8(float v) {
    v = fmaxf(v, __int_as_float(moe1_dpp<0xB1>(__float_as_int(v))));
    v = fmaxf(v, __int_as_float(moe1_dpp<0x4E>(__float_as_int(v))));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 4, WARP_SIZE));
    return v;
}
// max over aligned groups of 32 lanes: quad_perm xor 1 / xor 2, row_ror 4 / 8, then swap the 16-lane halves
static __device__ __forceinline__ float moe1_max32(float v) {
    v = fmaxf(v, __int_as_float(moe1_dpp<0xB1>(__float_as_int(v))));
    v = fmaxf(v, __int_as_float(moe1_dpp<0x4E>(__float_as_int(v))));
    v = fmaxf(v, __int_as_float(moe1_dpp<0x124>(__float_as_int(v))));
    v = fmaxf(v, __int_as_float(moe1_dpp<0x128>(__float_as_int(v))));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)));
    return v;
}

template <int XP, int G = 2> // passes of 512 float4 over the token: K <= 2048*XP; G row groups of 4 per wave (16*G rows)
static __global__ void __launch_bounds__(512, 4) moe1_gateup_q2k(
        const float * __restrict__ x, const int32_t * __restrict__ ids, const float * __restrict__ wts,
        const char * __restrict__ gate_exps, const char * __restrict__ up_exps, const int64_t nb_e,
        const char * __restrict__ gate_sh, const char * __restrict__ up_sh, const int64_t s_row,
        float * __restrict__ hout, int32_t * __restrict__ ids_copy, float * __restrict__ wts_copy,
        const int K, const int M, const int n_used, const float limit, const int exper, const int M_sh) {
    extern __shared__ int moe1_lds[];
    int   * xq = moe1_lds;                    // K/4 ints (swizzled)
    float * xd = (float *) (xq + K/4);        // K/32 activation scales
    float * sf = xd + K/32;                   // K/16 sub-block sums of the dequantized activations
    constexpr int ROWS_WG = 16*G;
    __shared__ float gu[2][ROWS_WG];

    const int tid = threadIdx.x;
    // the down kernel reads ids and router weights from these copies: the FFN output may reuse their memory
    if (blockIdx.x == 0 && blockIdx.y == 0 && tid < n_used) {
        ids_copy[tid] = ids[tid];
        wts_copy[tid] = wts[tid];
    }
    const int  e  = blockIdx.y;
    const bool sh = e == n_used;
    if ((int) blockIdx.x*ROWS_WG >= (sh ? M_sh : M)) {
        return; // the grid covers the wider of the routed and shared slices
    }

    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int mat  = wv >> 2;                  // 0: gate, 1: up
    const int r    = lane >> 4;                // row of the wave's group of 4
    const int b    = (lane >> 2) & 3;          // super-block of the 4 per step
    const int c    = lane & 3;
    const int n    = c >> 1;
    const int hh   = c & 1;
    const int nblk = K/QK_K;
    const char * wb = mat == 0 ? (sh ? gate_sh : gate_exps + (int64_t) ids[e]*nb_e)
                               : (sh ? up_sh   : up_exps   + (int64_t) ids[e]*nb_e);
    // rows past the slice (a last, partial workgroup when 16*G does not divide it) load the last row and store nothing
    const int M_e  = sh ? M_sh : M;
    const int row0 = blockIdx.x*ROWS_WG + 4*G*(wv & 3) + r;   // group g: row0 + 4*g
    wb += 16 + 16*c;

    // weight words of one step (4 super-blocks per wave; wb points at the lane's qs quarter): the first step's loads go
    // out before the token prologue and every step prefetches the next one
    struct wwords { uint4 v4; uint32_t ws0, ws1, wdm; };
    auto load_step = [&](const char * rowp, const int kb0) {
        wwords w;
        const char * blk = rowp + (int64_t) min(kb0 + b, nblk - 1)*sizeof(block_q2_K);
        w.v4  = *(const uint4 *) blk;
        w.ws0 = *(const uint32_t *) (blk - 16 - 16*c + 8*n);
        w.ws1 = *(const uint32_t *) (blk - 16 - 16*c + 8*n + 4);
        w.wdm = *(const uint32_t *) (blk - 16 - 16*c + 80);
        return w;
    };
    wwords cur = load_step(wb + (int64_t) min(row0, M_e - 1)*s_row, 0);

    // the token -> int8 in LDS: 8 lanes per 32-value block, float4 each; all loads first (one round trip)
    float4 xl[XP];
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int i4 = p*512 + tid;
        xl[p] = i4 < K/4 && exper != 1 ? ((const float4 *) x)[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int  i4 = p*512 + tid;
        const bool ok = i4 < K/4;
        const float4 v = xl[p];
        const float amax = moe1_max8(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float id   = amax > 0.0f ? 127.0f*__builtin_amdgcn_rcpf(amax) : 0.0f;
        const int q0 = (int) rintf(v.x*id), q1 = (int) rintf(v.y*id), q2 = (int) rintf(v.z*id), q3 = (int) rintf(v.w*id);
        int s4 = q0 + q1 + q2 + q3;
        s4 += moe1_dpp<0xB1>(s4);
        s4 += moe1_dpp<0x4E>(s4);
        if (ok) {
            xq[moe1_swz(i4)] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
            const float d = amax*(1.0f/127.0f);
            if ((i4 & 3) == 0) {
                sf[i4 >> 2] = d*(float) s4;
            }
            if ((i4 & 7) == 0) {
                xd[i4 >> 3] = d;
            }
        }
    }
    __syncthreads();

    // per-lane LDS offsets (the swizzle term (2*kb + n) & 3 does not change with the 4-aligned step base)
    int xo[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xo[j] = moe1_swz(64*b + 32*n + 8*j + 4*hh);
    }
    const int so = 16*b + 8*n + hh;
    const int dq = 8*b + 4*n; // 32-value activation block of j = 0

#pragma unroll 1
    for (int g = 0; g < G; ++g) {
        const char * rowp = wb + (int64_t) min(row0 + 4*g, M_e - 1)*s_row;
        if (g > 0) {
            cur = load_step(rowp, 0);
        }
        float acc = 0.0f;
        for (int kb0 = 0; kb0 < nblk; kb0 += 4) {
            const wwords nxt = kb0 + 4 < nblk ? load_step(rowp, kb0 + 4) : cur;
            if (kb0 + b < nblk) {
                if (exper == 2) {
                    acc += (float) (cur.v4.x ^ cur.v4.y ^ cur.v4.z ^ cur.v4.w ^ cur.ws0 ^ cur.ws1 ^ cur.wdm);
                } else {
                    const uint32_t sw0 = cur.ws0 >> (8*hh);
                    const uint32_t sw1 = cur.ws1 >> (8*hh);
                    float fs = 0.0f;
                    float fm = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int4 xv = *(const int4 *) (xq + 64*kb0 + xo[j]);
                        int s = 0;
                        s = ggml_cuda_dp4a((int) ((cur.v4.x >> (2*j)) & 0x03030303u), xv.x, s);
                        s = ggml_cuda_dp4a((int) ((cur.v4.y >> (2*j)) & 0x03030303u), xv.y, s);
                        s = ggml_cuda_dp4a((int) ((cur.v4.z >> (2*j)) & 0x03030303u), xv.z, s);
                        s = ggml_cuda_dp4a((int) ((cur.v4.w >> (2*j)) & 0x03030303u), xv.w, s);
                        const uint32_t sw = (j < 2 ? sw0 : sw1) >> (16*(j & 1));
                        fs = fmaf(xd[8*kb0 + dq + j], (float) __mul24((int) (sw & 0xF), s), fs); // |s| <= 16*3*127
                        fm = fmaf((float) ((sw >> 4) & 0xF), sf[16*kb0 + so + 2*j], fm);
                    }
                    const float d    = __half2float(__ushort_as_half((unsigned short) (cur.wdm & 0xFFFF)));
                    const float dmin = __half2float(__ushort_as_half((unsigned short) (cur.wdm >> 16)));
                    acc = fmaf(d, fs, fmaf(-dmin, fm, acc));
                }
            }
            cur = nxt;
        }
        acc = warp_reduce_sum<16>(acc); // the 16 lanes of a row
        if ((lane & 15) == 0) {
            gu[mat][4*G*(wv & 3) + 4*g + r] = acc;
        }
    }
    __syncthreads();
    if (tid < ROWS_WG && (int) blockIdx.x*ROWS_WG + tid < M_e) {
        hout[(int64_t) e*M + blockIdx.x*ROWS_WG + tid] = ggml_cuda_op_swiglu_clamp_single(gu[0][tid], gu[1][tid], limit);
    }
}

// down: workgroup = 64 output rows, wave e = expert e (row-lane q3_K, the moe_rowlane_q3k inner loop); each wave first
// quantizes its expert's h into LDS (q8, 32-value blocks); the weighted per-expert results are summed in expert order
// (routed 0..n_used-1, then the shared expert)
template <int NE>
static __global__ void __launch_bounds__(64*NE, 4) moe1_down_q3k(
        const float * __restrict__ hin, const int32_t * __restrict__ ids, const float * __restrict__ wts,
        const char * __restrict__ down_exps, const int64_t nb_e, const char * __restrict__ down_sh, const int64_t s_row,
        float * __restrict__ dst, const int M, const int N) {
    extern __shared__ int moe1_lds[];
    const int lane = threadIdx.x & 63;
    const int e    = threadIdx.x >> 6;
    const int n_used = NE - 1;
    const bool sh  = e == n_used;
    int   * yq = moe1_lds + e*(M/4);                       // this expert's h as int8 (M/4 ints)
    float * yd = (float *) (moe1_lds + NE*(M/4)) + e*(M/32);

    // h -> q8 (32-value blocks): 8 lanes per block, float4 each
    const float * he = hin + (int64_t) e*M;
    for (int i4 = lane; i4 < M/4; i4 += 64) {
        const float4 v = ((const float4 *) he)[i4];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = warp_reduce_max<8>(amax);
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        const int q0 = (int) roundf(v.x*id), q1 = (int) roundf(v.y*id), q2 = (int) roundf(v.z*id), q3 = (int) roundf(v.w*id);
        yq[i4] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
        if ((i4 & 7) == 0) {
            yd[i4 >> 3] = amax/127.0f;
        }
    }
    __syncthreads();

    const int  row = blockIdx.x*64 + lane;
    const bool valid = row < N;
    const char * base = sh ? down_sh : down_exps + (int64_t) ids[e]*nb_e;
    const char * xr   = base + (int64_t) (valid ? row : N - 1)*s_row;

    float acc = 0.0f;
    const int nsb = M / QK_K;
    for (int sb = 0; sb < nsb; ++sb) {
        const char * bp  = xr + sb*sizeof(block_q3_K);
        const int    mis = (int) ((uintptr_t) bp & 3); // 0 or 2
        const int  * qp  = (const int *) (bp - mis);
        int w[28];
#pragma unroll
        for (int k = 0; k < 28; ++k) {
            w[k] = qp[k];
        }
        auto dw = [&](const int k) {
            return (int) __builtin_amdgcn_alignbyte(w[k + 1], w[k], mis);
        };
        int hm[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            hm[i] = dw(i);
        }
        const unsigned a0 = dw(24), a1 = dw(25), tmp = dw(26);
        const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) w[27] >> (8*mis))));
        const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
        int scw[4];
        scw[0] = gkq_sub_bytes_mv((int) ((a0 & km2)        | (((tmp >> 0) & km1) << 4)), 0x20202020u);
        scw[1] = gkq_sub_bytes_mv((int) ((a1 & km2)        | (((tmp >> 2) & km1) << 4)), 0x20202020u);
        scw[2] = gkq_sub_bytes_mv((int) (((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)), 0x20202020u);
        scw[3] = gkq_sub_bytes_mv((int) (((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)), 0x20202020u);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int qd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qd[i] = dw(8 + 8*h + i);
            }
#pragma unroll
            for (int jb = 0; jb < 4; ++jb) {
                const int s  = 4*h + jb;
                const int is = 8*h + 2*jb;
                const int sc0 = (int8_t) ((scw[is/4] >> (8*(is % 4)))     & 0xFF);
                const int sc1 = (int8_t) ((scw[is/4] >> (8*(is % 4) + 8)) & 0xFF);
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int hb = s >= 2 ? hm[i] >> (s - 2) : hm[i] << (2 - s);
                    const int xx = (hb & 0x04040404) | ((qd[i] >> (2*jb)) & 0x03030303);
                    v[i] = (int) __builtin_amdgcn_perm(0x03020100u, 0xFFFEFDFCu, (unsigned) xx);
                }
                const int  yb = sb*8 + s;          // q8 block of these 32 values
                const int * yv = yq + 8*yb;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s0 = ggml_cuda_dp4a(v[i],     yv[i],     s0);
                    s1 = ggml_cuda_dp4a(v[4 + i], yv[4 + i], s1);
                }
                const int isum = __mul24(sc0, s0) + __mul24(sc1, s1);
                acc = fmaf(yd[yb], d*(float) isum, acc);
            }
        }
    }

    __shared__ float red[NE][64];
    red[e][lane] = sh ? acc : acc*wts[e];
    __syncthreads();
    if (e == 0 && valid) {
        float sum = red[0][lane];
#pragma unroll
        for (int k = 1; k < NE; ++k) {
            sum += red[k][lane];
        }
        dst[row] = sum;
    }
}

// down, coalesced: workgroup = 64 output rows x all NE experts (the shared expert last). An expert's 64 rows are one
// contiguous span (64*NSB*110 bytes): loaded with coalesced 16-byte lanes one expert ahead (registers), staged in LDS;
// thread = (row, super-block, half) unit(s) of the q3_K rows, 4 groups of 32 values each from LDS (the row-lane inner
// loop). Each thread sums its units' router-weighted results over the experts; the rows are summed through LDS. The
// row-lane kernel's 28 dword loads per block touched 64 rows per instruction (L1-thrashing, ~310 GB/s).
static constexpr int MOE1D2_ROWS = 64;

template <int NE, int NSB>
static __global__ void __launch_bounds__(256) moe1_down2_q3k(
        const float * __restrict__ hin, const int32_t * __restrict__ ids, const float * __restrict__ wts,
        const char * __restrict__ down_exps, const int64_t nb_e, const char * __restrict__ down_sh,
        float * __restrict__ dst) {
    constexpr int M     = 256*NSB;                  // inner size (the h vector of one expert)
    constexpr int S_ROW = 110*NSB;                  // bytes per weight row
    constexpr int SPAN4 = MOE1D2_ROWS*S_ROW/16;     // uint4 per (expert, 64 rows)
    constexpr int LPT   = (SPAN4 + 255)/256;        // uint4 loads per thread
    constexpr int UPR   = 2*NSB;                    // units per row
    constexpr int NU    = MOE1D2_ROWS*UPR;
    constexpr int UPT   = (NU + 255)/256;           // units per thread
    const int tid = threadIdx.x;
    const int r0  = blockIdx.x*MOE1D2_ROWS;

    __shared__ uint4 wsp[SPAN4 + 1];                // the expert's span (+1: the aligned reads may pass the end)
    __shared__ int   yq[NE][M/4];                   // h as int8 (32-value blocks), per expert
    __shared__ float yd[NE][M/32];
    __shared__ float red[MOE1D2_ROWS][UPR];

    auto span = [&](const int e) -> const uint4 * {
        const char * base = e == NE - 1 ? down_sh : down_exps + (int64_t) ids[e]*nb_e;
        return (const uint4 *) (base + (int64_t) r0*S_ROW);
    };

    // the first expert's span in flight, then h -> q8 (as moe1_down_q3k)
    uint4 pf[LPT];
    {
        const uint4 * sp = span(0);
#pragma unroll
        for (int j = 0; j < LPT; ++j) {
            const int k = tid + 256*j;
            pf[j] = k < SPAN4 ? sp[k] : make_uint4(0, 0, 0, 0);
        }
    }
    for (int i4 = tid; i4 < NE*M/4; i4 += 256) {
        const int e  = i4/(M/4);
        const int j4 = i4%(M/4);
        const float4 v = ((const float4 *) (hin + (int64_t) e*M))[j4];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = warp_reduce_max<8>(amax);
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        const int q0 = (int) roundf(v.x*id), q1 = (int) roundf(v.y*id), q2 = (int) roundf(v.z*id), q3 = (int) roundf(v.w*id);
        yq[e][j4] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
        if ((j4 & 7) == 0) {
            yd[e][j4 >> 3] = amax/127.0f;
        }
    }

    float acc[UPT];
#pragma unroll
    for (int uu = 0; uu < UPT; ++uu) {
        acc[uu] = 0.0f;
    }
    const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
#pragma unroll 1
    for (int e = 0; e < NE; ++e) {
        __syncthreads();                            // the previous span is consumed (e == 0: yq/yd are written)
#pragma unroll
        for (int j = 0; j < LPT; ++j) {
            const int k = tid + 256*j;
            if (k < SPAN4) {
                wsp[k] = pf[j];
            }
        }
        __syncthreads();
        if (e + 1 < NE) {                           // the next expert's span in flight during this one's math
            const uint4 * sp = span(e + 1);
#pragma unroll
            for (int j = 0; j < LPT; ++j) {
                const int k = tid + 256*j;
                pf[j] = k < SPAN4 ? sp[k] : make_uint4(0, 0, 0, 0);
            }
        }
        const float we = e == NE - 1 ? 1.0f : wts[e];
#pragma unroll
        for (int uu = 0; uu < UPT; ++uu) {
            const int u = tid + 256*uu;
            if (u >= NU) {
                continue;
            }
            const int r  = u/UPR;
            const int sb = (u%UPR)/2;
            const int h  = u%2;
            const int boff = r*S_ROW + sb*110;
            const int mis  = boff & 3;              // 0 or 2
            const int * qp = (const int *) ((const char *) wsp + (boff - mis));
            int wh[9], wq[9], ws[4];
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                wh[i] = qp[i];
                wq[i] = qp[8 + 8*h + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ws[i] = qp[24 + i];
            }
            auto al = [&](const int hi, const int lo) {
                return (int) __builtin_amdgcn_alignbyte(hi, lo, mis);
            };
            int hm[8], qd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hm[i] = al(wh[i + 1], wh[i]);
                qd[i] = al(wq[i + 1], wq[i]);
            }
            const unsigned a0 = al(ws[1], ws[0]), a1 = al(ws[2], ws[1]), tmp = al(ws[3], ws[2]);
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) ws[3] >> (8*mis))));
            int scw[4];
            scw[0] = gkq_sub_bytes_mv((int) ((a0 & km2)        | (((tmp >> 0) & km1) << 4)), 0x20202020u);
            scw[1] = gkq_sub_bytes_mv((int) ((a1 & km2)        | (((tmp >> 2) & km1) << 4)), 0x20202020u);
            scw[2] = gkq_sub_bytes_mv((int) (((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)), 0x20202020u);
            scw[3] = gkq_sub_bytes_mv((int) (((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)), 0x20202020u);
            float part = 0.0f;
#pragma unroll
            for (int jb = 0; jb < 4; ++jb) {
                const int s  = 4*h + jb;
                const int is = 8*h + 2*jb;
                const int sc0 = (int8_t) ((scw[is/4] >> (8*(is % 4)))     & 0xFF);
                const int sc1 = (int8_t) ((scw[is/4] >> (8*(is % 4) + 8)) & 0xFF);
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int hb = s >= 2 ? hm[i] >> (s - 2) : hm[i] << (2 - s);
                    const int xx = (hb & 0x04040404) | ((qd[i] >> (2*jb)) & 0x03030303);
                    v[i] = (int) __builtin_amdgcn_perm(0x03020100u, 0xFFFEFDFCu, (unsigned) xx);
                }
                const int   yb = sb*8 + s;
                const int * yv = yq[e] + 8*yb;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s0 = ggml_cuda_dp4a(v[i],     yv[i],     s0);
                    s1 = ggml_cuda_dp4a(v[4 + i], yv[4 + i], s1);
                }
                const int isum = __mul24(sc0, s0) + __mul24(sc1, s1);
                part = fmaf(yd[e][yb], d*(float) isum, part);
            }
            acc[uu] = fmaf(we, part, acc[uu]);
        }
    }
#pragma unroll
    for (int uu = 0; uu < UPT; ++uu) {
        const int u = tid + 256*uu;
        if (u < NU) {
            red[u/UPR][u%UPR] = acc[uu];
        }
    }
    __syncthreads();
    if (tid < MOE1D2_ROWS) {
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < UPR; ++k) {
            sum += red[tid][k];
        }
        dst[r0 + tid] = sum;
    }
}

// down, one workgroup per (64 rows, expert): its expert's 64-row span (64*NSB*110 contiguous bytes) is loaded with all
// coalesced 16-byte lanes in flight at once, staged in LDS, the expert's h quantized to q8 (as moe1_down_q3k), thread =
// (row, super-block, half) unit(s) of the q3_K rows. The router-weighted row sums go to part[expert][row]; the last of the
// NE workgroups of a row block (arrival counter) sums them in expert order (deterministic). Many small workgroups keep the
// load pipe full (moe1_down2_q3k's one span in flight per workgroup lost to the row-lane kernel).
template <int NE, int NSB_R, int NSB_S, int ROWS, int NT> // routed / shared expert super-blocks, ROWS rows per workgroup
static __global__ void __launch_bounds__(NT) moe1_down3_q3k(
        const float * __restrict__ hin, const int32_t * __restrict__ ids, const float * __restrict__ wts,
        const char * __restrict__ down_exps, const int64_t nb_e, const char * __restrict__ down_sh,
        float * __restrict__ part, int * __restrict__ counters, float * __restrict__ dst, const int N) {
    constexpr int NSBM   = NSB_R > NSB_S ? NSB_R : NSB_S;
    constexpr int MM     = 256*NSBM;
    constexpr int SPAN4M = ROWS*110*NSBM/16;
    constexpr int LPTM   = (SPAN4M + NT - 1)/NT;
    constexpr int UPTM   = (ROWS*2*NSBM + NT - 1)/NT;
    constexpr int XPM    = (MM/4 + NT - 1)/NT;       // float4 of h per thread
    static_assert(NT % 8 == 0 && NT >= ROWS, "8 lanes per 32-value block, one row sum per thread");
    const int tid = threadIdx.x;
    const int rb  = blockIdx.x;
    const int e   = blockIdx.y;
    const int r0  = rb*ROWS;
    const bool shx  = e == NE - 1;
    const int  nsb  = shx ? NSB_S : NSB_R;
    const int  M_e  = 256*nsb;
    const int  s_row = 110*nsb;
    const int  span4 = ROWS*s_row/16;
    const int  upr   = 2*nsb;
    const int  nu    = ROWS*upr;

    __shared__ uint4 wsp[SPAN4M + 1];
    __shared__ int   yq[MM/4];
    __shared__ float yd[MM/32];
    __shared__ float red[ROWS][2*NSBM];
    __shared__ int   s_last;

    // the span first (all in flight), then h of this expert (the gate/up kernel writes expert e's at e*M_routed)
    const char * base = shx ? down_sh : down_exps + (int64_t) ids[e]*nb_e;
    const uint4 * sp = (const uint4 *) (base + (int64_t) r0*s_row);
    uint4 pf[LPTM];
#pragma unroll
    for (int j = 0; j < LPTM; ++j) {
        const int k = tid + NT*j;
        pf[j] = k < span4 ? sp[k] : make_uint4(0, 0, 0, 0);
    }
    const float4 * he = (const float4 *) (hin + (int64_t) e*(256*NSB_R));
    const float we = shx ? 1.0f : wts[e];
#pragma unroll
    for (int xi = 0; xi < XPM; ++xi) {
        const int i4 = tid + NT*xi;
        const float4 hv = i4 < M_e/4 ? he[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float amax = fmaxf(fmaxf(fabsf(hv.x), fabsf(hv.y)), fmaxf(fabsf(hv.z), fabsf(hv.w)));
        amax = warp_reduce_max<8>(amax);
        const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
        const int q0 = (int) roundf(hv.x*id), q1 = (int) roundf(hv.y*id), q2 = (int) roundf(hv.z*id), q3 = (int) roundf(hv.w*id);
        if (i4 < M_e/4) {
            yq[i4] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
            if ((i4 & 7) == 0) {
                yd[i4 >> 3] = amax/127.0f;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < LPTM; ++j) {
        const int k = tid + NT*j;
        if (k < span4) {
            wsp[k] = pf[j];
        }
    }
    __syncthreads();

    const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
#pragma unroll
    for (int uu = 0; uu < UPTM; ++uu) {
        const int u = tid + NT*uu;
        if (u >= nu) {
            continue;
        }
        const int r  = shx ? u/(2*NSB_S) : u/(2*NSB_R); // constant divisors
        const int ur = u - r*upr;
        const int sb = ur >> 1;
        const int h  = ur & 1;
        const int boff = r*s_row + sb*110;
        const int mis  = boff & 3;
        const int * qp = (const int *) ((const char *) wsp + (boff - mis));
        int wh[9], wq[9], ws[4];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            wh[i] = qp[i];
            wq[i] = qp[8 + 8*h + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ws[i] = qp[24 + i];
        }
        auto al = [&](const int hi, const int lo) {
            return (int) __builtin_amdgcn_alignbyte(hi, lo, mis);
        };
        int hm[8], qd[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            hm[i] = al(wh[i + 1], wh[i]);
            qd[i] = al(wq[i + 1], wq[i]);
        }
        const unsigned a0 = al(ws[1], ws[0]), a1 = al(ws[2], ws[1]), tmp = al(ws[3], ws[2]);
        const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) ws[3] >> (8*mis))));
        int scw[4];
        scw[0] = gkq_sub_bytes_mv((int) ((a0 & km2)        | (((tmp >> 0) & km1) << 4)), 0x20202020u);
        scw[1] = gkq_sub_bytes_mv((int) ((a1 & km2)        | (((tmp >> 2) & km1) << 4)), 0x20202020u);
        scw[2] = gkq_sub_bytes_mv((int) (((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)), 0x20202020u);
        scw[3] = gkq_sub_bytes_mv((int) (((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)), 0x20202020u);
        float acc = 0.0f;
#pragma unroll
        for (int jb = 0; jb < 4; ++jb) {
            const int s  = 4*h + jb;
            const int is = 8*h + 2*jb;
            const int sc0 = (int8_t) ((scw[is/4] >> (8*(is % 4)))     & 0xFF);
            const int sc1 = (int8_t) ((scw[is/4] >> (8*(is % 4) + 8)) & 0xFF);
            int v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int hb = s >= 2 ? hm[i] >> (s - 2) : hm[i] << (2 - s);
                const int xx = (hb & 0x04040404) | ((qd[i] >> (2*jb)) & 0x03030303);
                v[i] = (int) __builtin_amdgcn_perm(0x03020100u, 0xFFFEFDFCu, (unsigned) xx);
            }
            const int   yb = sb*8 + s;
            const int * yv = yq + 8*yb;
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s0 = ggml_cuda_dp4a(v[i],     yv[i],     s0);
                s1 = ggml_cuda_dp4a(v[4 + i], yv[4 + i], s1);
            }
            const int isum = __mul24(sc0, s0) + __mul24(sc1, s1);
            acc = fmaf(yd[yb], d*(float) isum, acc);
        }
        red[r][ur] = acc;
    }
    __syncthreads();
    if (tid < ROWS) {
        float sum = 0.0f;
        for (int k = 0; k < upr; ++k) {
            sum += red[tid][k];
        }
        part[(int64_t) e*N + r0 + tid] = sum*we;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counters + rb, 1) == NE - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    if (tid < ROWS) {
        float pv[NE];
#pragma unroll
        for (int k = 0; k < NE; ++k) {
            pv[k] = part[(int64_t) k*N + r0 + tid];
        }
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < NE; ++k) {
            sum += pv[k];
        }
        dst[r0 + tid] = sum;
    }
    if (tid == 0) {
        counters[rb] = 0;
    }
}

// down, persistent and software-pipelined (opt-in GGML_CUDA_MOE1_DOWN5=1; LOSES: 100 VGPRs = 2 waves/SIMD vs down3's 4,
// and the conditional span loads defeat the wait counting, in-model 59 vs 31 us for M768): as many workgroups as are resident at once,
// each takes a contiguous range of (expert, ROWS-row block) units in expert-major order (mostly consecutive row blocks of
// one expert, so h is quantized once per expert change) and computes a unit from LDS while the next unit's span (and h)
// are in flight in registers (LDS-only barriers: nothing waits for them until they are staged). The unit math and the
// partial/merge layout are down3's (bit-identical results); the arrival counts and the last-arriver merges of a
// workgroup's row blocks happen once, after its last unit (one fence instead of one per unit).
template <int NE, int NSB_R, int NSB_S, int ROWS, int NT, int MAXU>
static __global__ void __launch_bounds__(NT) moe1_down5_q3k(
        const float * __restrict__ hin, const int32_t * __restrict__ ids, const float * __restrict__ wts,
        const char * __restrict__ down_exps, const int64_t nb_e, const char * __restrict__ down_sh,
        float * __restrict__ part, int * __restrict__ counters, float * __restrict__ dst, const int N) {
    constexpr int NSBM   = NSB_R > NSB_S ? NSB_R : NSB_S;
    constexpr int MM     = 256*NSBM;
    constexpr int SPAN4M = ROWS*110*NSBM/16;
    constexpr int LPTM   = (SPAN4M + NT - 1)/NT;
    constexpr int UPTM   = (ROWS*2*NSBM + NT - 1)/NT;
    constexpr int XPM    = (MM/4 + NT - 1)/NT;       // float4 of h per thread
    static_assert(NT % 8 == 0 && NT >= ROWS && NT >= MAXU, "8 lanes per 32-value block, one row sum per thread");
    const int tid  = threadIdx.x;
    const int n_rb = N/ROWS;
    const int U    = n_rb*NE;
    const int u0   = (int) (((int64_t) blockIdx.x*U)/gridDim.x);
    const int u1   = (int) (((int64_t) (blockIdx.x + 1)*U)/gridDim.x);

    __shared__ uint4 wsp[SPAN4M + 1];
    __shared__ int   yq[MM/4];
    __shared__ float yd[MM/32];
    __shared__ float red[ROWS*2*NSBM];
    __shared__ int   s_last[MAXU];

    // the routed experts and their router weights, once
    int   idv[NE - 1];
    float wtv[NE - 1];
#pragma unroll
    for (int k = 0; k < NE - 1; ++k) {
        idv[k] = ids[k];
        wtv[k] = wts[k];
    }

    uint4  pf[LPTM];
    float4 hf[XPM];
    auto issue = [&](const int u, const bool with_h) {
        const int  e     = u / n_rb;
        const int  rb    = u - e*n_rb;
        const bool shx   = e == NE - 1;
        const int  nsb   = shx ? NSB_S : NSB_R;
        const int  s_row = 110*nsb;
        const int  span4 = ROWS*s_row/16;
        int ide = 0;
#pragma unroll
        for (int k = 0; k < NE - 1; ++k) {
            ide = k == e ? idv[k] : ide;
        }
        const char  * base = shx ? down_sh : down_exps + (int64_t) ide*nb_e;
        const uint4 * sp   = (const uint4 *) (base + (int64_t) rb*ROWS*s_row);
#pragma unroll
        for (int j = 0; j < LPTM; ++j) {
            const int k = tid + NT*j;
            pf[j] = k < span4 ? sp[k] : make_uint4(0, 0, 0, 0);
        }
        if (with_h) {
            const float4 * he  = (const float4 *) (hin + (int64_t) e*(256*NSB_R));
            const int      M_e = 256*nsb;
#pragma unroll
            for (int xi = 0; xi < XPM; ++xi) {
                const int i4 = tid + NT*xi;
                hf[xi] = i4 < M_e/4 ? he[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    };

    issue(u0, true);
    int cur_e = -1;
    const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    for (int u = u0; u < u1; ++u) {
        const int  e     = u / n_rb;
        const int  rb    = u - e*n_rb;
        const int  r0    = rb*ROWS;
        const bool shx   = e == NE - 1;
        const int  nsb   = shx ? NSB_S : NSB_R;
        const int  M_e   = 256*nsb;
        const int  s_row = 110*nsb;
        const int  span4 = ROWS*s_row/16;
        const int  upr   = 2*nsb;
        const int  nu    = ROWS*upr;
        float we = 1.0f;
#pragma unroll
        for (int k = 0; k < NE - 1; ++k) {
            we = k == e ? wtv[k] : we;
        }

        // the previous unit's compute and row sums are done with wsp, yq, yd and red
        if (u != u0) {
            __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
        }
        if (e != cur_e) {
#pragma unroll
            for (int xi = 0; xi < XPM; ++xi) {
                const int i4 = tid + NT*xi;
                const float4 hv = i4 < M_e/4 ? hf[xi] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float amax = fmaxf(fmaxf(fabsf(hv.x), fabsf(hv.y)), fmaxf(fabsf(hv.z), fabsf(hv.w)));
                amax = warp_reduce_max<8>(amax);
                const float id = amax > 0.0f ? 127.0f/amax : 0.0f;
                const int q0 = (int) roundf(hv.x*id), q1 = (int) roundf(hv.y*id), q2 = (int) roundf(hv.z*id), q3 = (int) roundf(hv.w*id);
                if (i4 < M_e/4) {
                    yq[i4] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
                    if ((i4 & 7) == 0) {
                        yd[i4 >> 3] = amax/127.0f;
                    }
                }
            }
            cur_e = e;
        }
#pragma unroll
        for (int j = 0; j < LPTM; ++j) {
            const int k = tid + NT*j;
            if (k < span4) {
                wsp[k] = pf[j];
            }
        }
        // the next unit's loads go out now and stay in flight through this unit's compute
        if (u + 1 < u1) {
            issue(u + 1, (u + 1)/n_rb != e);
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");

        float accs[UPTM];
#pragma unroll
        for (int uu = 0; uu < UPTM; ++uu) {
            const int un = tid + NT*uu;
            accs[uu] = 0.0f;
            if (un >= nu) {
                continue;
            }
            const int r  = shx ? un/(2*NSB_S) : un/(2*NSB_R); // constant divisors
            const int ur = un - r*upr;
            const int sb = ur >> 1;
            const int h  = ur & 1;
            const int boff = r*s_row + sb*110;
            const int mis  = boff & 3;
            const int * qp = (const int *) ((const char *) wsp + (boff - mis));
            int wh[9], wq[9], ws[4];
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                wh[i] = qp[i];
                wq[i] = qp[8 + 8*h + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ws[i] = qp[24 + i];
            }
            auto al = [&](const int hi, const int lo) {
                return (int) __builtin_amdgcn_alignbyte(hi, lo, mis);
            };
            int hm[8], qd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                hm[i] = al(wh[i + 1], wh[i]);
                qd[i] = al(wq[i + 1], wq[i]);
            }
            const unsigned a0 = al(ws[1], ws[0]), a1 = al(ws[2], ws[1]), tmp = al(ws[3], ws[2]);
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) ws[3] >> (8*mis))));
            int scw[4];
            scw[0] = gkq_sub_bytes_mv((int) ((a0 & km2)        | (((tmp >> 0) & km1) << 4)), 0x20202020u);
            scw[1] = gkq_sub_bytes_mv((int) ((a1 & km2)        | (((tmp >> 2) & km1) << 4)), 0x20202020u);
            scw[2] = gkq_sub_bytes_mv((int) (((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)), 0x20202020u);
            scw[3] = gkq_sub_bytes_mv((int) (((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)), 0x20202020u);
            float acc = 0.0f;
#pragma unroll
            for (int jb = 0; jb < 4; ++jb) {
                const int s  = 4*h + jb;
                const int is = 8*h + 2*jb;
                const int sc0 = (int8_t) ((scw[is/4] >> (8*(is % 4)))     & 0xFF);
                const int sc1 = (int8_t) ((scw[is/4] >> (8*(is % 4) + 8)) & 0xFF);
                int v[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int hb = s >= 2 ? hm[i] >> (s - 2) : hm[i] << (2 - s);
                    const int xx = (hb & 0x04040404) | ((qd[i] >> (2*jb)) & 0x03030303);
                    v[i] = (int) __builtin_amdgcn_perm(0x03020100u, 0xFFFEFDFCu, (unsigned) xx);
                }
                const int   yb = sb*8 + s;
                const int * yv = yq + 8*yb;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s0 = ggml_cuda_dp4a(v[i],     yv[i],     s0);
                    s1 = ggml_cuda_dp4a(v[4 + i], yv[4 + i], s1);
                }
                const int isum = __mul24(sc0, s0) + __mul24(sc1, s1);
                acc = fmaf(yd[yb], d*(float) isum, acc);
            }
            accs[uu] = acc;
        }
#pragma unroll
        for (int uu = 0; uu < UPTM; ++uu) {
            const int un = tid + NT*uu;
            if (un < nu) {
                red[un] = accs[uu];
            }
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
        if (tid < ROWS) {
            float sum = 0.0f;
            for (int k = 0; k < upr; ++k) {
                sum += red[tid*upr + k];
            }
            part[(int64_t) e*N + r0 + tid] = sum*we;
        }
    }

    // all of this workgroup's partials released at once, then the arrivals of its row blocks
    __threadfence();
    __syncthreads();
    if (tid < u1 - u0) {
        const int u  = u0 + tid;
        const int rb = u - (u / n_rb)*n_rb;
        s_last[tid] = atomicAdd(counters + rb, 1) == NE - 1;
    }
    __syncthreads();
    for (int i = 0; i < u1 - u0; ++i) {
        if (!s_last[i]) {
            continue;
        }
        __threadfence();
        const int u  = u0 + i;
        const int rb = u - (u / n_rb)*n_rb;
        if (tid < ROWS) {
            const int rr = rb*ROWS + tid;
            float pv[NE];
#pragma unroll
            for (int k = 0; k < NE; ++k) {
                pv[k] = part[(int64_t) k*N + rr];
            }
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < NE; ++k) {
                sum += pv[k];
            }
            dst[rr] = sum;
        }
        if (tid == 0) {
            counters[rb] = 0;
        }
    }
}

// ---- single-token dense q2_K matvec (the moe1_gateup_q2k inner loop for one matrix) ----
// workgroup = 16 rows (4 waves x 4 rows); every workgroup quantizes the token into LDS (int8 in 32-value blocks +
// per-16 sub-block sums); lane = (row r of 4, super-block b of 4, 16-byte qs quarter c), steps prefetched one ahead.
// Batched over blockIdx.y (src0/src1/dst channel strides) for grouped projections.
static constexpr int GEMV1_ROWS_WG = 16;

// NORM: the token is RMS_NORM(x)*nw first (the rms_norm_f32 formula; fused RMS_NORM -> MUL -> MUL_MAT, single token)
// MULTI: several matrices (same K, same token) in one launch: workgroup -> (segment, row block) from mt (w, y, M, s_row
// and the batch arguments come from the segment)
// RG: row groups per wave (16*RG rows per workgroup): big matrices (q_b 8192 rows) need fewer workgroups (one round
// instead of ~3, and the token quantization amortized over more rows); every group's weight words are in flight at once
// the ROPE math of rope.cu (rope_yarn<true>, rope_yarn_ramp) for the epilogue
static __device__ __forceinline__ void gemv1_rope_yarn(const float theta_extrap, const float freq_scale, const float corr0,
        const float corr1, const int i0, const float ext_factor, float mscale, float & cos_theta, float & sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = (i0 / 2 - corr0) / max(0.001f, corr1 - corr0);
        const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

template <int XP, bool NORM, bool MULTI = false, int RG = 1> // passes of 256 float4 over the token: K <= 1024*XP (= the lane's steps of 4 super-blocks)
static __global__ void __launch_bounds__(256, 2) gemv1_q2k(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int K, int M, int64_t s_row, const int64_t s_w2, const int64_t s_x2, const int64_t s_y2,
        const float * __restrict__ nw, const float eps, const ggml_cuda_gemv1_multi mt, const ggml_cuda_gemv1_epi ep) {
    extern __shared__ int moe1_lds[];
    int   * xq = moe1_lds;                    // K/4 ints (swizzled)
    float * xd = (float *) (xq + K/4);        // K/32 activation scales
    float * sf = xd + K/32;                   // K/16 sub-block sums of the dequantized activations

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int r    = lane >> 4;
    const int b    = (lane >> 2) & 3;
    const int c    = lane & 3;
    const int n    = c >> 1;
    const int hh   = c & 1;
    const int nblk = K/QK_K;
    int wgx = blockIdx.x;
    int sg  = 0;
    if (MULTI) {
#pragma unroll
        for (int k = 1; k < GGML_CUDA_GEMV1_MULTI_MAX; ++k) {
            sg = k < mt.n && wgx >= mt.wg0[k] ? k : sg;
        }
        wgx  -= mt.wg0[sg];
        w     = mt.w[sg];
        y     = mt.y[sg];
        M     = mt.M[sg];
        s_row = mt.s_row[sg];
    }
    const int row0 = wgx*GEMV1_ROWS_WG*RG + 4*wv + r;   // group g: row0 + 16*g
    x += blockIdx.y*s_x2;
    y += blockIdx.y*s_y2;

    const bool epw = MULTI && ep.seg >= 0 && sg == ep.seg;
    float   e_w0 = 0.0f, e_w1 = 0.0f, e_c = 1.0f, e_s = 0.0f;
    int32_t e_pos = 0;

    // the token first (vector memory completes in issue order), then every weight word of the lane: one round trip
    float4 xl[XP], nl[XP];
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int i4 = p*256 + tid;
        xl[p] = i4 < K/4 ? ((const float4 *) x)[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (NORM) {
            nl[p] = i4 < K/4 ? ((const float4 *) nw)[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    struct wwords { uint4 v4; uint32_t ws0, ws1, wdm; };
    wwords ws[RG][XP];
#pragma unroll
    for (int g = 0; g < RG; ++g) {
        const int row = row0 + 16*g;
        const char * wb = w + blockIdx.y*s_w2 + (int64_t) (row < M ? row : M - 1)*s_row + 16 + 16*c;
#pragma unroll
        for (int st = 0; st < XP; ++st) {
            // clamped block index: a step past the matrix reloads its last block (unused)
            const char * blk = wb + (int64_t) min(4*st + b, nblk - 1)*sizeof(block_q2_K);
            ws[g][st].v4  = *(const uint4 *) blk;
            ws[g][st].ws0 = *(const uint32_t *) (blk - 16 - 16*c + 8*n);
            ws[g][st].ws1 = *(const uint32_t *) (blk - 16 - 16*c + 8*n + 4);
            ws[g][st].wdm = *(const uint32_t *) (blk - 16 - 16*c + 80);
            __asm__ volatile("" ::: "memory"); // issue the steps in order: the dot products consume them in order
        }
    }
    // epilogue inputs behind the weight words: the norm weight pair is used only at the end (vector memory returns in
    // issue order), the position is a scalar load (its own counter), so neither delays a step
    if (epw) {
        // scalar load (constant address space): a vector load of this uniform value would be moved to a scalar register
        // right away, which waits for every weight word issued before it
        e_pos = *(const __attribute__((address_space(4))) int32_t *) (uintptr_t) ep.pos;
        if (2*tid < M) {
            const float2 wv2 = *(const float2 *) (ep.nw + 2*tid);
            e_w0 = wv2.x;
            e_w1 = wv2.y;
        }
    }
    if (NORM) {
        // sum of squares: wave sums by DPP (lane 63), the 4 waves through LDS behind an LDS-only barrier (a
        // __syncthreads would also wait for every weight load in flight)
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < XP; ++p) {
            const float4 v = xl[p];
            ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0xB1,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x4E,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x141, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x140, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x142, 0xA, 0xF, false));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x143, 0xC, 0xF, false));
        __shared__ float gemv1_ss[4];
        if (lane == 63) {
            gemv1_ss[wv] = ss;
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
        const float scale = rsqrtf((gemv1_ss[0] + gemv1_ss[1] + gemv1_ss[2] + gemv1_ss[3])/K + eps);
#pragma unroll
        for (int p = 0; p < XP; ++p) {
            xl[p] = make_float4(scale*xl[p].x*nl[p].x, scale*xl[p].y*nl[p].y, scale*xl[p].z*nl[p].z, scale*xl[p].w*nl[p].w);
        }
    }
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int  i4 = p*256 + tid;
        const bool ok = i4 < K/4;
        const float4 v = xl[p];
        const float amax = moe1_max8(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float id   = amax > 0.0f ? 127.0f*__builtin_amdgcn_rcpf(amax) : 0.0f;
        const int q0 = (int) rintf(v.x*id), q1 = (int) rintf(v.y*id), q2 = (int) rintf(v.z*id), q3 = (int) rintf(v.w*id);
        int s4 = q0 + q1 + q2 + q3;
        s4 += moe1_dpp<0xB1>(s4);
        s4 += moe1_dpp<0x4E>(s4);
        if (ok) {
            xq[moe1_swz(i4)] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
            const float d = amax*(1.0f/127.0f);
            if ((i4 & 3) == 0) {
                sf[i4 >> 2] = d*(float) s4;
            }
            if ((i4 & 7) == 0) {
                xd[i4 >> 3] = d;
            }
        }
    }
    // LDS-only barrier (build with -DGGML_CUDA_GEMV1_FULLBAR for __syncthreads, which also waits for every weight word in
    // flight): each step's dot products start as soon as its own words arrive (vector memory returns in issue order)
#ifndef GGML_CUDA_GEMV1_FULLBAR
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
#else
    __syncthreads();
#endif

    int xo[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xo[j] = moe1_swz(64*b + 32*n + 8*j + 4*hh);
    }
    const int so = 16*b + 8*n + hh;
    const int dq = 8*b + 4*n; // 32-value activation block of j = 0

    if (epw) {
        // the rotation of this thread's pair (no frequency factors: the matcher requires none) while the weights stream
        const int i0 = 2*tid;
        if (i0 >= ep.n_offs && i0 < ep.n_offs + ep.n_dims) {
            const int   iw         = i0 - ep.n_offs;
            const float theta_base = e_pos*powf(ep.theta_scale, iw/2.0f);
            gemv1_rope_yarn(theta_base, ep.freq_scale, ep.corr0, ep.corr1, iw, ep.ext_factor, ep.attn_factor, e_c, e_s);
        }
    }

#pragma unroll
    for (int g = 0; g < RG; ++g) {
    float acc = 0.0f;
#pragma unroll
    for (int st = 0; st < XP; ++st) {
        const int kb0 = 4*st;
        __builtin_amdgcn_sched_barrier(0); // step order: each step waits only for its own words
        if (kb0 + b < nblk) {
            const wwords cur = ws[g][st];
            const uint32_t sw0 = cur.ws0 >> (8*hh);
            const uint32_t sw1 = cur.ws1 >> (8*hh);
            float fs = 0.0f;
            float fm = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int4 xv = *(const int4 *) (xq + 64*kb0 + xo[j]);
                int s = 0;
                s = ggml_cuda_dp4a((int) ((cur.v4.x >> (2*j)) & 0x03030303u), xv.x, s);
                s = ggml_cuda_dp4a((int) ((cur.v4.y >> (2*j)) & 0x03030303u), xv.y, s);
                s = ggml_cuda_dp4a((int) ((cur.v4.z >> (2*j)) & 0x03030303u), xv.z, s);
                s = ggml_cuda_dp4a((int) ((cur.v4.w >> (2*j)) & 0x03030303u), xv.w, s);
                const uint32_t sw = (j < 2 ? sw0 : sw1) >> (16*(j & 1));
                fs = fmaf(xd[8*kb0 + dq + j], (float) __mul24((int) (sw & 0xF), s), fs); // |s| <= 16*3*127
                fm = fmaf((float) ((sw >> 4) & 0xF), sf[16*kb0 + so + 2*j], fm);
            }
            const float d    = __half2float(__ushort_as_half((unsigned short) (cur.wdm & 0xFFFF)));
            const float dmin = __half2float(__ushort_as_half((unsigned short) (cur.wdm >> 16)));
            acc = fmaf(d, fs, fmaf(-dmin, fm, acc));
        }
    }
    acc = warp_reduce_sum<16>(acc);
    const int row = row0 + 16*g;
    if ((lane & 15) == 0 && row < M) {
        y[row] = acc;
    }
    }

    if (MULTI && ep.seg >= 0) {
        // the epilogue segment's last workgroup normalizes, rotates and stores the whole output row (M <= 512: a pair
        // per thread, the rms_norm_mul_rope_f32 float operations; only the sum order differs)
        if (!epw) {
            return;
        }
        __shared__ int   gemv1_last;
        __shared__ float gemv1_es[4];
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            gemv1_last = atomicAdd(ep.counter, 1) == (M + GEMV1_ROWS_WG - 1)/GEMV1_ROWS_WG - 1;
        }
        __syncthreads();
        if (!gemv1_last) {
            return;
        }
        __threadfence();
        const int i0 = 2*tid;
        const int64_t e_row = ep.row_idx[0];
        float x0 = 0.0f, x1 = 0.0f;
        if (i0 < M) {
            const float2 v = *(const float2 *) (y + i0);
            x0 = v.x;
            x1 = v.y;
        }
        float ss = x0*x0 + x1*x1;
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0xB1,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x4E,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x141, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x140, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x142, 0xA, 0xF, false));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x143, 0xC, 0xF, false));
        if (lane == 63) {
            gemv1_es[wv] = ss;
        }
        __syncthreads();
        const float scale = rsqrtf((gemv1_es[0] + gemv1_es[1] + gemv1_es[2] + gemv1_es[3])/M + ep.eps);
        if (i0 < M) {
            const float v0 = scale * x0 * e_w0;
            const float v1 = scale * x1 * e_w1;
            float o0 = v0, o1 = v1;
            if (i0 >= ep.n_offs && i0 < ep.n_offs + ep.n_dims) {
                o0 = v0*e_c - v1*e_s;
                o1 = v0*e_s + v1*e_c;
            }
            *(half2 *) (ep.dst + e_row*ep.stride + i0) = __floats2half2_rn(o0, o1);
        }
        if (tid == 0) {
            *ep.counter = 0;
        }
    }
}

// small K (NB <= 6 super-blocks): lane = (row of 16, 16-byte qs quarter), step = super-block, so every lane is busy in
// every step (gemv1_q2k's 4 super-blocks per step leave 3/4 of the lanes idle in K = 1280's second step) and a workgroup
// covers 64 rows (V4.1 q_b 8192 x 1280: 128 workgroups in one round instead of 512 in ~1.7). Same token layout and dot
// products as gemv1_q2k (bit-identical rows).
// MULTI: several matrices sharing the token, workgroup -> (segment, 64-row block) from mt.wg0 (counted in 64-row blocks)
template <int NB, bool NORM, bool MULTI = false>
static __global__ void __launch_bounds__(256, 2) gemv1r_q2k(
        const char * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int K, int M, int64_t s_row, const int64_t s_w2, const int64_t s_x2, const int64_t s_y2,
        const float * __restrict__ nw, const float eps, const ggml_cuda_gemv1_multi mt) {
    constexpr int XP = (NB*64 + 255)/256;     // float4 of the token per thread
    extern __shared__ int moe1_lds[];
    int   * xq = moe1_lds;                    // K/4 ints (swizzled)
    float * xd = (float *) (xq + K/4);        // K/32 activation scales
    float * sf = xd + K/32;                   // K/16 sub-block sums of the dequantized activations

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int r    = lane >> 2;
    const int c    = lane & 3;
    const int n    = c >> 1;
    const int hh   = c & 1;
    int wgx = blockIdx.x;
    if (MULTI) {
        int sg = 0;
#pragma unroll
        for (int k = 1; k < GGML_CUDA_GEMV1_MULTI_MAX; ++k) {
            sg = k < mt.n && wgx >= mt.wg0[k] ? k : sg;
        }
        wgx  -= mt.wg0[sg];
        w     = mt.w[sg];
        y     = mt.y[sg];
        M     = mt.M[sg];
        s_row = mt.s_row[sg];
    }
    const int row  = wgx*64 + 16*wv + r;
    x += blockIdx.y*s_x2;
    y += blockIdx.y*s_y2;

    // the token first (vector memory returns in issue order), then every weight word of the lane in step order
    float4 xl[XP], nl[XP];
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int i4 = p*256 + tid;
        xl[p] = i4 < K/4 ? ((const float4 *) x)[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (NORM) {
            nl[p] = i4 < K/4 ? ((const float4 *) nw)[i4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    struct wwords { uint4 v4; uint32_t ws0, ws1, wdm; };
    wwords ws[NB];
    const char * wb = w + blockIdx.y*s_w2 + (int64_t) (row < M ? row : M - 1)*s_row + 16 + 16*c;
#pragma unroll
    for (int st = 0; st < NB; ++st) {
        const char * blk = wb + (int64_t) st*sizeof(block_q2_K);
        ws[st].v4  = *(const uint4 *) blk;
        ws[st].ws0 = *(const uint32_t *) (blk - 16 - 16*c + 8*n);
        ws[st].ws1 = *(const uint32_t *) (blk - 16 - 16*c + 8*n + 4);
        ws[st].wdm = *(const uint32_t *) (blk - 16 - 16*c + 80);
        __asm__ volatile("" ::: "memory"); // issue the steps in order: the dot products consume them in order
    }
    if (NORM) {
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < XP; ++p) {
            const float4 v = xl[p];
            ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0xB1,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x4E,  0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x141, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x140, 0xF, 0xF, true));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x142, 0xA, 0xF, false));
        ss += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(ss), 0x143, 0xC, 0xF, false));
        __shared__ float gemv1r_ss[4];
        if (lane == 63) {
            gemv1r_ss[wv] = ss;
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");
        const float scale = rsqrtf((gemv1r_ss[0] + gemv1r_ss[1] + gemv1r_ss[2] + gemv1r_ss[3])/K + eps);
#pragma unroll
        for (int p = 0; p < XP; ++p) {
            xl[p] = make_float4(scale*xl[p].x*nl[p].x, scale*xl[p].y*nl[p].y, scale*xl[p].z*nl[p].z, scale*xl[p].w*nl[p].w);
        }
    }
#pragma unroll
    for (int p = 0; p < XP; ++p) {
        const int  i4 = p*256 + tid;
        const bool ok = i4 < K/4;
        const float4 v = xl[p];
        const float amax = moe1_max8(fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        const float id   = amax > 0.0f ? 127.0f*__builtin_amdgcn_rcpf(amax) : 0.0f;
        const int q0 = (int) rintf(v.x*id), q1 = (int) rintf(v.y*id), q2 = (int) rintf(v.z*id), q3 = (int) rintf(v.w*id);
        int s4 = q0 + q1 + q2 + q3;
        s4 += moe1_dpp<0xB1>(s4);
        s4 += moe1_dpp<0x4E>(s4);
        if (ok) {
            xq[moe1_swz(i4)] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | ((int) ((unsigned) q3 << 24));
            const float d = amax*(1.0f/127.0f);
            if ((i4 & 3) == 0) {
                sf[i4 >> 2] = d*(float) s4;
            }
            if ((i4 & 7) == 0) {
                xd[i4 >> 3] = d;
            }
        }
    }
    // LDS-only barrier: the weight words stay in flight, each step waits for its own
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory");

    float acc = 0.0f;
#pragma unroll
    for (int st = 0; st < NB; ++st) {
        __builtin_amdgcn_sched_barrier(0);
        const wwords cur = ws[st];
        const uint32_t sw0 = cur.ws0 >> (8*hh);
        const uint32_t sw1 = cur.ws1 >> (8*hh);
        float fs = 0.0f;
        float fm = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int4 xv = *(const int4 *) (xq + moe1_swz(64*st + 32*n + 8*j + 4*hh));
            int s = 0;
            s = ggml_cuda_dp4a((int) ((cur.v4.x >> (2*j)) & 0x03030303u), xv.x, s);
            s = ggml_cuda_dp4a((int) ((cur.v4.y >> (2*j)) & 0x03030303u), xv.y, s);
            s = ggml_cuda_dp4a((int) ((cur.v4.z >> (2*j)) & 0x03030303u), xv.z, s);
            s = ggml_cuda_dp4a((int) ((cur.v4.w >> (2*j)) & 0x03030303u), xv.w, s);
            const uint32_t sw = (j < 2 ? sw0 : sw1) >> (16*(j & 1));
            fs = fmaf(xd[8*st + 4*n + j], (float) __mul24((int) (sw & 0xF), s), fs); // |s| <= 16*3*127
            fm = fmaf((float) ((sw >> 4) & 0xF), sf[16*st + 8*n + hh + 2*j], fm);
        }
        const float d    = __half2float(__ushort_as_half((unsigned short) (cur.wdm & 0xFFFF)));
        const float dmin = __half2float(__ushort_as_half((unsigned short) (cur.wdm >> 16)));
        acc = fmaf(d, fs, fmaf(-dmin, fm, acc));
    }
    acc = warp_reduce_sum<4>(acc);            // the row's 4 quarters
    if (c == 0 && row < M) {
        y[row] = acc;
    }
}

bool ggml_cuda_gemv1_q2k_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const int env = moe_vec_env("GGML_CUDA_GEMV1", 1);
    static const int64_t gemv1_max_rows = moe_vec_env("GGML_CUDA_GEMV1_MAX_ROWS", 32768);
    const int64_t K = src0->ne[0];
    return env && GGML_CUDA_CC_IS_GCN(cc) && src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && src1->ne[1] == 1 && src1->ne[3] == 1 && src0->ne[3] == 1 &&
        src0->ne[2] == src1->ne[2] && dst->ne[2] == src1->ne[2] && src1->ne[2] <= 65535 &&
        // every workgroup quantizes the token, so the row count is capped (GGML_CUDA_GEMV1_MAX_ROWS): V4.1's engram
        // projection (25600 x 6144) is still faster here in-model than on the row-lane kernel (tg 94.5 vs 93.9; isolated
        // test-backend-ops timings had favored the row-lane kernel)
        src0->ne[1]*src0->ne[2] <= gemv1_max_rows &&
        K % QK_K == 0 && K <= 8192 && src0->nb[1] % 4 == 0 && src0->nb[2] % 4 == 0 && ((uintptr_t) src0->data) % 4 == 0 &&
        src1->nb[0] == sizeof(float) && src1->nb[2] % 16 == 0 && ((uintptr_t) src1->data) % 16 == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[2] % sizeof(float) == 0 &&
        (K/4 + K/32 + K/16)*sizeof(int) <= 48*1024;
}

template <int XP, int RG>
static void gemv1_q2k_launch(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const float * norm_w,
                             const float norm_eps, cudaStream_t stream) {
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    const dim3 grid((M + GEMV1_ROWS_WG*RG - 1)/(GEMV1_ROWS_WG*RG), src1->ne[2]);
    const size_t lds = (size_t) (K/4 + K/32 + K/16)*sizeof(int);
    const ggml_cuda_gemv1_multi mt0 = {};
    ggml_cuda_gemv1_epi ep0 = {};
    ep0.seg = -1;
    if (norm_w) {
        gemv1_q2k<XP, true, false, RG><<<grid, 256, lds, stream>>>((const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, K, M, src0->nb[1], src0->nb[2], src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float), norm_w, norm_eps, mt0, ep0);
    } else {
        gemv1_q2k<XP, false, false, RG><<<grid, 256, lds, stream>>>((const char *) src0->data, (const float *) src1->data,
            (float *) dst->data, K, M, src0->nb[1], src0->nb[2], src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float), nullptr, 0.0f, mt0, ep0);
    }
}

void ggml_cuda_gemv1_q2k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         const float * norm_w, const float norm_eps) {
    const int K = (int) src0->ne[0], M = (int) src0->ne[1];
    cudaStream_t stream = ctx.stream();
    // K <= 6 super-blocks: the row-major variant (GGML_CUDA_GEMV1_ROWMAJOR=0 off)
    static const bool rowmajor = [] { const char * e = getenv("GGML_CUDA_GEMV1_ROWMAJOR"); return !e || atoi(e) != 0; }();
    const int nb = K/QK_K;
    if (rowmajor && nb <= 6) {
        const dim3 grid((M + 63)/64, src1->ne[2]);
        const size_t lds = (size_t) (K/4 + K/32 + K/16)*sizeof(int);
        const ggml_cuda_gemv1_multi mt0 = {};
#define GEMV1R(NB) do { if (norm_w) { \
            gemv1r_q2k<NB, true><<<grid, 256, lds, stream>>>((const char *) src0->data, (const float *) src1->data, (float *) dst->data, \
                K, M, src0->nb[1], src0->nb[2], src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float), norm_w, norm_eps, mt0); \
        } else { \
            gemv1r_q2k<NB, false><<<grid, 256, lds, stream>>>((const char *) src0->data, (const float *) src1->data, (float *) dst->data, \
                K, M, src0->nb[1], src0->nb[2], src1->nb[2]/sizeof(float), dst->nb[2]/sizeof(float), nullptr, 0.0f, mt0); \
        } } while (0)
        switch (nb) {
            case 1:  GEMV1R(1); break;
            case 2:  GEMV1R(2); break;
            case 3:  GEMV1R(3); break;
            case 4:  GEMV1R(4); break;
            case 5:  GEMV1R(5); break;
            default: GEMV1R(6); break;
        }
#undef GEMV1R
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    // row groups per wave for big matrices (opt-in GGML_CUDA_GEMV1_RG=1): in-model q_b 19.0 vs 15.5 us, wo_b 14.2 vs
    // 9.8 us - fewer, fatter workgroups (2 waves/SIMD at 82-108 VGPRs) lose to 16-row workgroups at 6 waves/SIMD
    static const bool rg_on = [] { const char * e = getenv("GGML_CUDA_GEMV1_RG"); return e && atoi(e) != 0; }();
    const int xp = (K + 1023)/1024;
    const int rg = !rg_on || M < 4096 || src1->ne[2] != 1 ? 1 : (xp <= 2 ? 4 : (xp <= 5 ? 2 : 1));
#define GEMV1(xp) do { if (rg == 4) { gemv1_q2k_launch<xp, xp <= 2 ? 4 : 1>(src0, src1, dst, norm_w, norm_eps, stream); } \
        else if (rg == 2) { gemv1_q2k_launch<xp, xp <= 5 ? 2 : 1>(src0, src1, dst, norm_w, norm_eps, stream); } \
        else { gemv1_q2k_launch<xp, 1>(src0, src1, dst, norm_w, norm_eps, stream); } } while (0)
    switch ((K + 1023)/1024) {
        case 1:  GEMV1(1); break;
        case 2:  GEMV1(2); break;
        case 3:  GEMV1(3); break;
        case 4:  GEMV1(4); break;
        case 5:  GEMV1(5); break;
        case 6:  GEMV1(6); break;
        case 7:  GEMV1(7); break;
        default: GEMV1(8); break;
    }
#undef GEMV1
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gemv1_q2k_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mms, const int n,
                               const ggml_cuda_gemv1_epi * epi) {
    GGML_ASSERT(n >= 1 && n <= GGML_CUDA_GEMV1_MULTI_MAX);
    ggml_cuda_gemv1_epi ep = {};
    ep.seg = -1;
    if (epi != nullptr) {
        ep = *epi;
        GGML_ASSERT(ep.seg >= 0 && ep.seg < n && mms[ep.seg]->src[0]->ne[1] <= 512 && mms[ep.seg]->src[0]->ne[1] % 2 == 0);
        if (ctx.gemv1_epi_counter == nullptr) {
            CUDA_CHECK(cudaMalloc((void **) &ctx.gemv1_epi_counter, sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(ctx.gemv1_epi_counter, 0, sizeof(int), ctx.stream()));
        }
        ep.counter = ctx.gemv1_epi_counter;
    }
    const ggml_tensor * src1 = mms[0]->src[1];
    const int K = (int) mms[0]->src[0]->ne[0];
    // K <= 6 super-blocks without an epilogue: the row-major variant (64-row workgroups; GGML_CUDA_GEMV1_ROWMAJOR=0 off)
    static const bool rowmajor = [] { const char * e = getenv("GGML_CUDA_GEMV1_ROWMAJOR"); return !e || atoi(e) != 0; }();
    const bool rm = rowmajor && ep.seg < 0 && K/QK_K <= 6;
    const int rows_wg = rm ? 64 : GEMV1_ROWS_WG;
    ggml_cuda_gemv1_multi mt = {};
    mt.n = n;
    int nwg = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * w = mms[k]->src[0];
        mt.w[k]     = (const char *) w->data;
        mt.y[k]     = (float *) mms[k]->data;
        mt.M[k]     = (int) w->ne[1];
        mt.s_row[k] = w->nb[1];
        mt.wg0[k]   = nwg;
        nwg += (int) ((w->ne[1] + rows_wg - 1)/rows_wg);
    }
    if (rm) {
        const size_t lds = (size_t) (K/4 + K/32 + K/16)*sizeof(int);
        cudaStream_t stream = ctx.stream();
#define GEMV1RM(NB) gemv1r_q2k<NB, false, true><<<dim3(nwg, 1), 256, lds, stream>>>(mt.w[0], (const float *) src1->data, \
            mt.y[0], K, mt.M[0], mt.s_row[0], 0, 0, 0, nullptr, 0.0f, mt)
        switch (K/QK_K) {
            case 1:  GEMV1RM(1); break;
            case 2:  GEMV1RM(2); break;
            case 3:  GEMV1RM(3); break;
            case 4:  GEMV1RM(4); break;
            case 5:  GEMV1RM(5); break;
            default: GEMV1RM(6); break;
        }
#undef GEMV1RM
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const size_t lds = (size_t) (K/4 + K/32 + K/16)*sizeof(int);
    cudaStream_t stream = ctx.stream();
#define GEMV1M(xp) gemv1_q2k<xp, false, true><<<dim3(nwg, 1), 256, lds, stream>>>(mt.w[0], (const float *) src1->data, mt.y[0], \
        K, mt.M[0], mt.s_row[0], 0, 0, 0, nullptr, 0.0f, mt, ep)
    switch ((K + 1023)/1024) {
        case 1:  GEMV1M(1); break;
        case 2:  GEMV1M(2); break;
        case 3:  GEMV1M(3); break;
        case 4:  GEMV1M(4); break;
        case 5:  GEMV1M(5); break;
        case 6:  GEMV1M(6); break;
        case 7:  GEMV1M(7); break;
        default: GEMV1M(8); break;
    }
#undef GEMV1M
    CUDA_CHECK(cudaGetLastError());
}

// shapes moe1_down3_q3k instantiates: routed slice 512/768, shared slice 256/512/768, 7 or 9 experts
static bool ggml_cuda_moe1_down3_ok(const int64_t M, const int64_t M_sh, const int64_t N, const int NE) {
    static const bool down3 = [] { const char * e = getenv("GGML_CUDA_MOE1_DOWN3"); return !e || atoi(e) != 0; }();
    return down3 && N % 32 == 0 && N/32 <= 1024 && (M == 512 || M == 768) && (M_sh == 256 || M_sh == 512 || M_sh == 768) &&
        (NE == 7 || NE == 9);
}

bool ggml_cuda_moe1_supported(int cc, const ggml_cuda_moe1_args & a) {
    static const int env = moe_vec_env("GGML_CUDA_MOE1", 1);
    if (!env || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const int64_t K = a.gate_exps->ne[0], M = a.gate_exps->ne[1], N = a.down_exps->ne[1];
    const int64_t M_sh = a.gate_sh->ne[1]; // the shared expert's slice may differ (V4.1 MoE balance), down3 handles it
    const int n_used = (int) a.ids->ne[0];
    auto q2_ok = [&](const ggml_tensor * t, bool exps) {
        return t->type == GGML_TYPE_Q2_K && t->ne[0] == K && t->ne[1] == (exps ? M : M_sh) && (exps || (t->ne[2] == 1 && t->ne[3] == 1)) &&
            t->nb[1] == a.gate_exps->nb[1] && t->nb[1] % 4 == 0 && ((uintptr_t) t->data) % 4 == 0 &&
            (!exps || t->nb[2] == a.gate_exps->nb[2]);
    };
    auto q3_ok = [&](const ggml_tensor * t, bool exps) {
        const int64_t Mt = exps ? M : M_sh;
        return t->type == GGML_TYPE_Q3_K && t->ne[0] == Mt && t->ne[1] == N && (exps || (t->ne[2] == 1 && t->ne[3] == 1)) &&
            t->nb[1] == (size_t) (Mt/QK_K)*sizeof(block_q3_K) && t->nb[1] % 2 == 0 && ((uintptr_t) t->data) % 4 == 0;
    };
    if (M_sh != M && !ggml_cuda_moe1_down3_ok(M, M_sh, N, n_used + 1)) {
        return false;
    }
    return (n_used == 6 || n_used == 8) && K % QK_K == 0 && K % 4 == 0 && M % QK_K == 0 && M % MOE1_ROWS_WG == 0 &&
        M_sh % QK_K == 0 && M_sh > 0 &&
        q2_ok(a.gate_exps, true) && q2_ok(a.up_exps, true) && q2_ok(a.gate_sh, false) && q2_ok(a.up_sh, false) &&
        q3_ok(a.down_exps, true) && q3_ok(a.down_sh, false) &&
        a.x->type == GGML_TYPE_F32 && ggml_nelements(a.x) == K && ggml_is_contiguous(a.x) && ((uintptr_t) a.x->data) % 16 == 0 &&
        a.ids->type == GGML_TYPE_I32 && ggml_nelements(a.ids) == n_used && a.ids->nb[0] == sizeof(int32_t) &&
        a.weights->type == GGML_TYPE_F32 && ggml_nelements(a.weights) == n_used && ggml_is_contiguous(a.weights) &&
        a.dst->type == GGML_TYPE_F32 && ggml_nelements(a.dst) == N && ggml_is_contiguous(a.dst) &&
        (K/4 + K/32 + K/16)*sizeof(int) <= 48*1024 && K % 128 == 0 && K <= 8192 && M % 128 == 0 &&
        (size_t) (n_used + 1)*(M/4 + M/32)*sizeof(int) <= 48*1024;
}

void ggml_cuda_moe1_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_moe1_args & a) {
    cudaStream_t stream = ctx.stream();
    const int K = (int) a.gate_exps->ne[0], M = (int) a.gate_exps->ne[1], N = (int) a.down_exps->ne[1];
    const int M_sh = (int) a.gate_sh->ne[1];
    const int n_used = (int) a.ids->ne[0];
    const int NE = n_used + 1;
    ggml_cuda_pool_alloc<float>   h(ctx.pool(), (size_t) n_used*M + M_sh); // routed e at e*M, the shared expert after
    ggml_cuda_pool_alloc<int32_t> idw(ctx.pool(), 2*16);
    int32_t * ids_copy = idw.get();
    float   * wts_copy = (float *) (idw.get() + 16);

    static const int moe1_exper = moe_vec_env("GGML_CUDA_MOE1_EXPER", 0); // debug: 1 = no token quantization, 2 = loads only
    const size_t lds1 = (size_t) (K/4 + K/32 + K/16)*sizeof(int);
    // 32-row workgroups (GGML_CUDA_MOE1_GU_G=1: 16-row, one group of 4 rows per wave: tg 88.1 vs 88.7 - every
    // workgroup quantizes the 5120-value token, so fewer workgroups win here)
    static const int gu_g = [] { const char * e = getenv("GGML_CUDA_MOE1_GU_G"); return e ? atoi(e) : 2; }();
    // two 512-thread workgroups fit a CU (63 VGPRs): with more 32-row units than that (a 768 routed slice: 6*24 + shared),
    // the remainder runs as a second, mostly idle round (in-model 33 vs 24 us), so such a launch uses 48-row workgroups
    // (G = 3) and fits one round (GGML_CUDA_MOE1_GU_FIT=0 off)
    static const bool gu_fit = [] { const char * e = getenv("GGML_CUDA_MOE1_GU_FIT"); return !e || atoi(e) != 0; }();
    const int gsel = gu_g != 2 ? gu_g :
        (gu_fit && M % 48 == 0 && n_used*(M/32) + (M_sh + 31)/32 > 2*ggml_cuda_info().devices[ctx.device].nsm ? 3 : 2);
#define MOE1_GATEUP(xp) do { if (gsel == 3) { MOE1_GATEUP_G(xp, 3); } else if (gsel == 2) { MOE1_GATEUP_G(xp, 2); } \
        else { MOE1_GATEUP_G(xp, 1); } } while (0)
#define MOE1_GATEUP_G(xp, g) moe1_gateup_q2k<xp, g><<<dim3((std::max(M, M_sh) + 16*(g) - 1)/(16*(g)), NE), 512, lds1, stream>>>( \
        (const float *) a.x->data, (const int32_t *) a.ids->data, (const float *) a.weights->data, (const char *) a.gate_exps->data, \
        (const char *) a.up_exps->data, a.gate_exps->nb[2], (const char *) a.gate_sh->data, (const char *) a.up_sh->data, \
        a.gate_exps->nb[1], h.get(), ids_copy, wts_copy, K, M, n_used, a.limit, moe1_exper, M_sh)
    switch ((K + 2047)/2048) {
        case 1:  MOE1_GATEUP(1); break;
        case 2:  MOE1_GATEUP(2); break;
        case 3:  MOE1_GATEUP(3); break;
        default: MOE1_GATEUP(4); break;
    }
#undef MOE1_GATEUP
#undef MOE1_GATEUP_G
    CUDA_CHECK(cudaGetLastError());

    // one workgroup per (32 rows, expert) + last-arriver expert sum (GGML_CUDA_MOE1_DOWN3=0 off; required when the shared
    // expert's slice differs from the routed one)
    if (ggml_cuda_moe1_down3_ok(M, M_sh, N, NE) &&
            a.down_exps->nb[1] == (size_t) (M/256)*110 && a.down_sh->nb[1] == (size_t) (M_sh/256)*110 &&
            a.down_exps->nb[2] % 16 == 0 && ((uintptr_t) a.down_exps->data) % 16 == 0 && ((uintptr_t) a.down_sh->data) % 16 == 0) {
        if (ctx.moe1_counters == nullptr) {
            CUDA_CHECK(cudaMalloc((void **) &ctx.moe1_counters, 1024*sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(ctx.moe1_counters, 0, 1024*sizeof(int), stream));
        }
        ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) NE*N);
        // persistent pipelined variant (opt-in GGML_CUDA_MOE1_DOWN5=1, slower): one workgroup per resident slot
        static const bool down5 = [] { const char * e = getenv("GGML_CUDA_MOE1_DOWN5"); return e && atoi(e) != 0; }();
        if (down5) {
            const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
            const int U   = (N/32)*NE;
            bool launched = false;
#define MOE1_DOWN5(ne, nr, ns) do { \
            constexpr int NT5 = ((nr) == 3 || (ns) == 3) ? 192 : 128; \
            auto kern = moe1_down5_q3k<ne, nr, ns, 32, NT5, 8>; \
            static int occ = 0; \
            if (occ == 0) { CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, NT5, 0)); occ = std::max(occ, 1); } \
            const int G = std::max(std::min(U, occ*nsm), (U + 7)/8); \
            kern<<<G, NT5, 0, stream>>>(h.get(), ids_copy, wts_copy, (const char *) a.down_exps->data, a.down_exps->nb[2], \
                (const char *) a.down_sh->data, part.get(), ctx.moe1_counters, (float *) a.dst->data, N); \
            launched = true; \
        } while (0)
#define MOE1_DOWN5_NE(ne) do { \
            const int nr = M/256, ns = M_sh/256; \
            if (nr == 2) { if (ns == 1) { MOE1_DOWN5(ne, 2, 1); } else if (ns == 2) { MOE1_DOWN5(ne, 2, 2); } else { MOE1_DOWN5(ne, 2, 3); } } \
            else         { if (ns == 1) { MOE1_DOWN5(ne, 3, 1); } else if (ns == 2) { MOE1_DOWN5(ne, 3, 2); } else { MOE1_DOWN5(ne, 3, 3); } } \
        } while (0)
            if (NE == 7) {
                MOE1_DOWN5_NE(7);
            } else {
                MOE1_DOWN5_NE(9);
            }
#undef MOE1_DOWN5_NE
#undef MOE1_DOWN5
            if (launched) {
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        // 32 rows per workgroup, 128 threads (both slices 512) or 192 (a 768 slice: one unit per thread)
#define MOE1_DOWN3(ne, nr, ns) moe1_down3_q3k<ne, nr, ns, 32, ((nr) == 3 || (ns) == 3) ? 192 : 128> \
            <<<dim3(N/32, ne), ((nr) == 3 || (ns) == 3) ? 192 : 128, 0, stream>>>(h.get(), \
            ids_copy, wts_copy, (const char *) a.down_exps->data, a.down_exps->nb[2], (const char *) a.down_sh->data, \
            part.get(), ctx.moe1_counters, (float *) a.dst->data, N)
#define MOE1_DOWN3_NE(ne) do { \
            const int nr = M/256, ns = M_sh/256; \
            if (nr == 2) { if (ns == 1) { MOE1_DOWN3(ne, 2, 1); } else if (ns == 2) { MOE1_DOWN3(ne, 2, 2); } else { MOE1_DOWN3(ne, 2, 3); } } \
            else         { if (ns == 1) { MOE1_DOWN3(ne, 3, 1); } else if (ns == 2) { MOE1_DOWN3(ne, 3, 2); } else { MOE1_DOWN3(ne, 3, 3); } } \
        } while (0)
        if (NE == 7) {
            MOE1_DOWN3_NE(7);
        } else {
            MOE1_DOWN3_NE(9);
        }
#undef MOE1_DOWN3_NE
#undef MOE1_DOWN3
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // coalesced down (opt-in GGML_CUDA_MOE1_DOWN2=1): bandwidth-better in isolation, but its 7 serial expert steps per
    // workgroup (one span in flight) lose in-model to the row-lane kernel (tg 83.9 vs 86.0)
    static const bool down2 = [] { const char * e = getenv("GGML_CUDA_MOE1_DOWN2"); return e && atoi(e) != 0; }();
    if (down2 && N % MOE1D2_ROWS == 0 && (M == 512 || M == 768) && (NE == 7 || NE == 9) &&
            a.down_exps->nb[1] == (size_t) (M/256)*110 && a.down_sh->nb[1] == a.down_exps->nb[1] &&
            a.down_exps->nb[2] % 16 == 0 && ((uintptr_t) a.down_exps->data) % 16 == 0 && ((uintptr_t) a.down_sh->data) % 16 == 0) {
#define MOE1_DOWN2(ne, nsb) moe1_down2_q3k<ne, nsb><<<N/MOE1D2_ROWS, 256, 0, stream>>>(h.get(), ids_copy, wts_copy, \
            (const char *) a.down_exps->data, a.down_exps->nb[2], (const char *) a.down_sh->data, (float *) a.dst->data)
        if (NE == 7) {
            if (M == 512) { MOE1_DOWN2(7, 2); } else { MOE1_DOWN2(7, 3); }
        } else {
            if (M == 512) { MOE1_DOWN2(9, 2); } else { MOE1_DOWN2(9, 3); }
        }
#undef MOE1_DOWN2
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(M_sh == M); // the kernels below index the shared expert like a routed one

    const size_t lds2 = (size_t) NE*(M/4 + M/32)*sizeof(int);
    const dim3 grid((N + 63)/64);
#define MOE1_DOWN(ne) moe1_down_q3k<ne><<<grid, 64*(ne), lds2, stream>>>(h.get(), ids_copy, wts_copy, \
        (const char *) a.down_exps->data, a.down_exps->nb[2], (const char *) a.down_sh->data, \
        a.down_exps->nb[1], (float *) a.dst->data, M, N)
    switch (NE) {
        case 7: MOE1_DOWN(7); break;
        case 9: MOE1_DOWN(9); break;
        default: GGML_ABORT("moe1: unsupported expert count");
    }
#undef MOE1_DOWN
    CUDA_CHECK(cudaGetLastError());
}
