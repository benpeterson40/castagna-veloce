#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

// Ties at the cut. When more values equal the cut than slots remain, the gather's shared/global atomic slots kept
// whichever equal values the waves (or blocks) reached first, so the selection changed from run to run (the
// indexer's per-head ReLU makes many blocks score exactly 0 at long context). Ties now keep the lowest columns:
// TOPK_TIE_BUF equal values of a block's range are collected during the gather and ranked by column afterwards;
// more than that (rare) take an ordered walk over the range. `before` = equal values of the row before the range.
static constexpr int TOPK_TIE_BUF = 256;

template <int BLOCK_SIZE>
static __device__ __forceinline__ void top_k_ties_rank(
        const int * __restrict__ buf, const int n, int * __restrict__ row_dst, const int k, const int rank,
        const int before) {
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const int c = buf[i];
        int r = before;
        for (int j = 0; j < n; ++j) {
            r += buf[j] < c;
        }
        if (r < rank) {
            row_dst[k - rank + r] = c;
        }
    }
}

template <int BLOCK_SIZE>
static __device__ __forceinline__ void top_k_ties_ordered(
        const float * __restrict__ row_src, int * __restrict__ row_dst, const int c_beg, const int c_end,
        const uint32_t prefix, const int k, const int rank, int taken) {
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    constexpr int NW = BLOCK_SIZE / WS;
    __shared__ int s_wcnt[NW];
    const int tid  = threadIdx.x;
    const int lane = tid % WS;
    const int w    = tid / WS;
    const uint64_t below = lane == 63 ? ~0ull >> 1 : ((1ull << lane) - 1);
    for (int col0 = c_beg; col0 < c_end && taken < rank; col0 += BLOCK_SIZE) { // taken is block-uniform
        const int  col = col0 + tid;
        const bool eq  = col < c_end && top_k_float_to_ordered(row_src[col]) == prefix;
        const uint64_t m_eq = __ballot(eq);
        if (lane == 0) {
            s_wcnt[w] = __popcll(m_eq);
        }
        __syncthreads();
        int before = taken, total = 0;
#pragma unroll
        for (int j = 0; j < NW; ++j) {
            const int c = s_wcnt[j];
            before += j < w ? c : 0;
            total  += c;
        }
        const int pos = before + __popcll(m_eq & below);
        if (eq && pos < rank) {
            row_dst[k - rank + pos] = col;
        }
        taken += total;
        __syncthreads();
    }
}

// one block per row: the four 8-bit radix passes and the gather in a single launch, histograms in shared memory.
// The multi-block version below needs 11 launches and global histograms, so a one-row decode top-k was
// latency-bound (32768 -> 2051: 106 us, 8192: 92 us on gfx906); the row is re-read from L2 each pass.
template<int BLOCK_SIZE>
static __global__ void __launch_bounds__(BLOCK_SIZE) top_k_radix_row(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int NBINS = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;

    __shared__ int      histogram[NBINS];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int      s_rank, s_gt, s_eq, s_eqn;
    __shared__ int      s_tie[TOPK_TIE_BUF];

    if (tid == 0) {
        s_prefix = 0;
        s_mask   = 0;
        s_rank   = k;
        s_gt     = 0;
        s_eq     = 0;
        s_eqn    = 0;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < NBINS; b += BLOCK_SIZE) {
            histogram[b] = 0;
        }
        __syncthreads();
        const uint32_t prefix = s_prefix;
        const uint32_t mask   = s_mask;
        // scores share their high bits, so the first passes put almost every value in one or two bins: count runs
        // of the same bin per thread and add them once (one shared atomic per run instead of per value)
        int run_bin = -1;
        int run_cnt = 0;
        auto count = [&](const uint32_t key) {
            if ((key & mask) == prefix) {
                const int b = (key >> shift) & (NBINS - 1);
                if (b == run_bin) {
                    run_cnt++;
                } else {
                    if (run_cnt > 0) {
                        atomicAdd(&histogram[run_bin], run_cnt);
                    }
                    run_bin = b;
                    run_cnt = 1;
                }
            }
        };
        // 8 loads in flight per thread: one block per row is latency-bound on the row reads otherwise
        constexpr int U = 8;
        int col = tid;
        for (; col + (U - 1)*BLOCK_SIZE < ncols; col += U*BLOCK_SIZE) {
            float v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                v[u] = row_src[col + u*BLOCK_SIZE];
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                count(top_k_float_to_ordered(v[u]));
            }
        }
        for (; col < ncols; col += BLOCK_SIZE) {
            count(top_k_float_to_ordered(row_src[col]));
        }
        if (run_cnt > 0) {
            atomicAdd(&histogram[run_bin], run_cnt);
        }
        __syncthreads();
        // pick the largest bin b with suffix(b) = sum_{j >= b} hist[j] >= rank (bin 0 if none) and continue with
        // rank - suffix(b + 1): the suffix sums are a parallel scan (a serial walk over 256 shared-memory bins
        // by one thread was most of the kernel's time)
        const int rank_in = s_rank;
        for (int step = 1; step < NBINS; step *= 2) {
            int v = 0;
            if (tid < NBINS) {
                v = histogram[tid] + (tid + step < NBINS ? histogram[tid + step] : 0);
            }
            __syncthreads();
            if (tid < NBINS) {
                histogram[tid] = v;
            }
            __syncthreads();
        }
        if (tid < NBINS) {
            const int suf      = histogram[tid];
            const int suf_next = tid + 1 < NBINS ? histogram[tid + 1] : 0;
            if ((suf >= rank_in && suf_next < rank_in) || (tid == 0 && suf < rank_in)) {
                s_rank    = rank_in - suf_next;
                s_prefix |= (uint32_t) tid << shift;
                s_mask   |= (uint32_t) (NBINS - 1) << shift;
                s_eqn     = suf - suf_next; // values in the chosen bin; after the last pass, the values equal to the cut
            }
        }
        __syncthreads();
    }
    const uint32_t prefix = s_prefix;
    const int      rank   = s_rank;
    const int      eqn    = s_eqn;
    const bool     ties   = eqn > rank;                   // see top_k_ties_rank
    const bool     tbuf   = ties && eqn <= TOPK_TIE_BUF;  // equal values go to s_tie (slot order: any)
    constexpr int  WS     = ggml_cuda_get_physical_warp_size();
    const int      lane   = tid % WS;
    // wave-aggregated slots: one shared atomic per wave and outcome, lanes take consecutive slots by ballot rank
    // (the order of the kept columns may vary between runs, their set may not: consumers only test membership)
    for (int col0 = 0; col0 < ncols; col0 += BLOCK_SIZE) {
        const int col = col0 + tid;
        uint32_t key = 0;
        if (col < ncols) {
            key = top_k_float_to_ordered(row_src[col]);
        }
        const bool gt = col < ncols && key > prefix;
        const bool eq = col < ncols && key == prefix;
        const uint64_t m_gt = __ballot(gt);
        const uint64_t m_eq = __ballot(eq);
        const uint64_t below = lane == 63 ? ~0ull >> 1 : ((1ull << lane) - 1);
        int base_gt = 0, base_eq = 0;
        if (lane == 0) {
            if (m_gt) base_gt = atomicAdd(&s_gt, __popcll(m_gt));
            if (m_eq) base_eq = atomicAdd(&s_eq, __popcll(m_eq));
        }
        base_gt = __shfl(base_gt, 0, WS);
        base_eq = __shfl(base_eq, 0, WS);
        if (gt) {
            row_dst[base_gt + __popcll(m_gt & below)] = col;
        } else if (eq) {
            const int pos = base_eq + __popcll(m_eq & below);
            if (!ties) {
                if (pos < rank) {
                    row_dst[k - rank + pos] = col;
                }
            } else if (tbuf) {
                s_tie[pos] = col;
            }
        }
    }
    if (tbuf) {
        __syncthreads();
        top_k_ties_rank<BLOCK_SIZE>(s_tie, eqn, row_dst, k, rank, 0);
    } else if (ties) {
        __syncthreads();
        top_k_ties_ordered<BLOCK_SIZE>(row_src, row_dst, 0, ncols, prefix, k, rank, 0);
    }
}

// Decode / MTP verify (<= 8 rows): each row split over several blocks (one block per row walks 33K scores four times:
// 58 us at 32K on gfx906). Per pass: block-local histogram of its segment, non-zero bins added to a global histogram,
// grid barrier (all blocks are co-resident: <= 64 blocks), then every block derives the same bin from the global
// histogram. Scratch lives in __device__ globals per (virtual device, stream) and is left zeroed for the next call.
static constexpr int TOPK_MB_MAX_ROWS = 8;
struct topk_mb_scratch {
    int          hist[TOPK_MB_MAX_ROWS][4][256];
    int          gt[TOPK_MB_MAX_ROWS];
    int          eq[TOPK_MB_MAX_ROWS];
    int          eqcnt[TOPK_MB_MAX_ROWS][64];  // per part: values of its segment equal to the cut (bpr <= 64)
    unsigned int bar_count, bar_gen, done;
};
static __device__ topk_mb_scratch g_topk_mb[GGML_CUDA_MAX_DEVICES*GGML_CUDA_MAX_STREAMS];

static __device__ __forceinline__ void topk_mb_barrier(topk_mb_scratch * sc, const unsigned int nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned int * gen = &sc->bar_gen;
        const unsigned int g = *gen;
        __threadfence();
        if (atomicAdd(&sc->bar_count, 1u) == nblocks - 1) {
            atomicExch(&sc->bar_count, 0u);
            __threadfence();
            atomicAdd(&sc->bar_gen, 1u);
        } else {
            while (*gen == g) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence();
    }
    __syncthreads();
}

template <int BLOCK_SIZE>
static __global__ void __launch_bounds__(BLOCK_SIZE) top_k_radix_mb(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k, const int bpr,
        topk_mb_scratch * __restrict__ sc) {
    constexpr int NBINS = 256;
    const int row  = blockIdx.x / bpr;
    const int part = blockIdx.x % bpr;
    const int tid  = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const int seg   = (ncols + bpr - 1) / bpr;
    const int c_beg = part*seg, c_end = min(ncols, c_beg + seg);

    __shared__ int histogram[NBINS];
    __shared__ int suffix[NBINS];
    uint32_t prefix = 0, mask = 0;
    int rank = k;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8*pass;
        for (int b = tid; b < NBINS; b += BLOCK_SIZE) {
            histogram[b] = 0;
        }
        __syncthreads();
        int run_bin = -1, run_cnt = 0;
        auto count = [&](const uint32_t key) {
            if ((key & mask) == prefix) {
                const int b = (key >> shift) & (NBINS - 1);
                if (b == run_bin) {
                    run_cnt++;
                } else {
                    if (run_cnt > 0) {
                        atomicAdd(&histogram[run_bin], run_cnt);
                    }
                    run_bin = b;
                    run_cnt = 1;
                }
            }
        };
        constexpr int U = 4;
        int col = c_beg + tid;
        for (; col + (U - 1)*BLOCK_SIZE < c_end; col += U*BLOCK_SIZE) {
            float v[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                v[u] = row_src[col + u*BLOCK_SIZE];
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                count(top_k_float_to_ordered(v[u]));
            }
        }
        for (; col < c_end; col += BLOCK_SIZE) {
            count(top_k_float_to_ordered(row_src[col]));
        }
        if (run_cnt > 0) {
            atomicAdd(&histogram[run_bin], run_cnt);
        }
        __syncthreads();
        int * gh = sc->hist[row][pass];
        for (int b = tid; b < NBINS; b += BLOCK_SIZE) {
            if (histogram[b]) {
                atomicAdd(&gh[b], histogram[b]);
            }
        }
        topk_mb_barrier(sc, gridDim.x);
        // suffix sums of the global histogram (every block the same)
        for (int b = tid; b < NBINS; b += BLOCK_SIZE) {
            suffix[b] = __builtin_nontemporal_load(&gh[b]);
        }
        __syncthreads();
        for (int step = 1; step < NBINS; step *= 2) {
            int v = 0;
            if (tid < NBINS) {
                v = suffix[tid] + (tid + step < NBINS ? suffix[tid + step] : 0);
            }
            __syncthreads();
            if (tid < NBINS) {
                suffix[tid] = v;
            }
            __syncthreads();
        }
        __shared__ int s_sel, s_rank;
        if (tid < NBINS) {
            const int suf      = suffix[tid];
            const int suf_next = tid + 1 < NBINS ? suffix[tid + 1] : 0;
            if ((suf >= rank && suf_next < rank) || (tid == 0 && suf < rank)) {
                s_sel  = tid;
                s_rank = rank - suf_next;
            }
        }
        __syncthreads();
        prefix |= (uint32_t) s_sel << shift;
        mask   |= (uint32_t) (NBINS - 1) << shift;
        rank    = s_rank;
        __syncthreads();
    }
    // this block's values equal to the cut: its own (shared) histogram of the last pass at the cut's low byte;
    // published before the barrier, so that ties can be ranked by column across the blocks of the row
    const int n_loc = histogram[prefix & (NBINS - 1)];
    if (tid == 0) {
        sc->eqcnt[row][part] = n_loc;
    }
    // every block has read all four global histograms: block 0 clears them for the next call after this barrier
    topk_mb_barrier(sc, gridDim.x);
    // suffix[] still holds the last pass's suffix sums over the whole row
    const int  sel_last = (int) (prefix & (NBINS - 1));
    const int  eq_row   = suffix[sel_last] - (sel_last + 1 < NBINS ? suffix[sel_last + 1] : 0);
    const bool ties     = eq_row > rank;                   // see top_k_ties_rank
    const bool tbuf     = ties && n_loc <= TOPK_TIE_BUF;
    __shared__ int s_tie[TOPK_TIE_BUF];
    __shared__ int s_ntie;
    if (tid == 0) {
        s_ntie = 0;
    }
    __syncthreads();
    if (blockIdx.x == 0) {
        for (int i = tid; i < TOPK_MB_MAX_ROWS*4*NBINS; i += BLOCK_SIZE) {
            (&sc->hist[0][0][0])[i] = 0;
        }
    }
    constexpr int WS = ggml_cuda_get_physical_warp_size();
    const int lane = tid % WS;
    const uint64_t below = lane == 63 ? ~0ull >> 1 : ((1ull << lane) - 1);
    for (int col0 = c_beg; col0 < c_end; col0 += BLOCK_SIZE) {
        const int col = col0 + tid;
        uint32_t key = 0;
        if (col < c_end) {
            key = top_k_float_to_ordered(row_src[col]);
        }
        const bool gt = col < c_end && key > prefix;
        const bool eq = col < c_end && key == prefix;
        const uint64_t m_gt = __ballot(gt);
        const uint64_t m_eq = __ballot(eq);
        int base_gt = 0, base_eq = 0;
        if (lane == 0) {
            if (m_gt) base_gt = atomicAdd(&sc->gt[row], __popcll(m_gt));
            if (m_eq) base_eq = ties ? atomicAdd(&s_ntie, __popcll(m_eq)) : atomicAdd(&sc->eq[row], __popcll(m_eq));
        }
        base_gt = __shfl(base_gt, 0, WS);
        base_eq = __shfl(base_eq, 0, WS);
        if (gt) {
            row_dst[base_gt + __popcll(m_gt & below)] = col;
        } else if (eq) {
            const int pos = base_eq + __popcll(m_eq & below);
            if (!ties) {
                if (pos < rank) {
                    row_dst[k - rank + pos] = col;
                }
            } else if (tbuf) {
                s_tie[pos] = col;
            }
        }
    }
    if (ties) {
        // equal values of the lower parts of the row rank first
        int before = 0;
        for (int p = 0; p < part; ++p) {
            before += __builtin_nontemporal_load(&sc->eqcnt[row][p]);
        }
        __syncthreads();
        if (tbuf) {
            top_k_ties_rank<BLOCK_SIZE>(s_tie, n_loc, row_dst, k, rank, before);
        } else {
            top_k_ties_ordered<BLOCK_SIZE>(row_src, row_dst, c_beg, c_end, prefix, k, rank, before);
        }
    }
    // the last block to finish clears the slot counters
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sc->done, 1u) == gridDim.x - 1) {
            for (int r = 0; r < TOPK_MB_MAX_ROWS; ++r) {
                sc->gt[r] = 0;
                sc->eq[r] = 0;
            }
            sc->done = 0;
            __threadfence();
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

// small rows (<= 1024): the argsort bitonic network (same comparisons, so the same order and ties) with the values in LDS
// next to the indices (the argsort kernel re-reads x[idx] from global memory at every step) and the first k written
// straight to dst (no 2D copy)
static int top_k_next_pow2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template <int NPAD>
static __global__ void __launch_bounds__(NPAD) top_k_bitonic_lds(const float * __restrict__ x, int * __restrict__ dst,
        const int ncols, const int k) {
    const int col = threadIdx.x;
    const int row = blockIdx.x;
    __shared__ int   si[NPAD];
    __shared__ float sv[NPAD];
    si[col] = col;
    sv[col] = col < ncols ? x[(int64_t) row*ncols + col] : 0.0f;
    __syncthreads();
    for (int kk = 2; kk <= NPAD; kk *= 2) {
        for (int j = kk/2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                const int   a  = si[col], b = si[ixj];
                const float va = sv[col], vb = sv[ixj];
                // descending: as k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
                const bool sw = (col & kk) == 0 ? (a >= ncols || (b < ncols && va < vb))
                                                : (b >= ncols || (a < ncols && va > vb));
                if (sw) {
                    si[col] = b; si[ixj] = a;
                    sv[col] = vb; sv[ixj] = va;
                }
            }
            __syncthreads();
        }
    }
    if (col < k) {
        dst[(int64_t) row*k + col] = si[col];
    }
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    static const bool row_radix = [] { const char * e = getenv("GGML_CUDA_TOPK_ROW"); return !e || atoi(e) != 0; }();
    if (ncols > 1024 && row_radix && ncols <= INT_MAX && nrows <= INT_MAX) {
        // few rows: a wide block per row keeps the passes short; many rows (prefill) fill the GPU with smaller blocks
        static const int mb_seg = [] { const char * e = getenv("GGML_CUDA_TOPK_MB_SEG"); return e ? atoi(e) : 1024; }();
        // gfx906, width 2051: 33024 scores 56 -> 28 us (32 blocks); at 8448 the single block is faster (22 vs 30 us)
        const int bpr = mb_seg > 0 ? (int) std::min<int64_t>(64 / std::max<int64_t>(nrows, 1), (ncols + mb_seg - 1) / mb_seg) : 1;
        if (nrows <= TOPK_MB_MAX_ROWS && bpr >= 2 && ncols >= 16384) {
            static topk_mb_scratch * base[GGML_CUDA_MAX_DEVICES] = {};
            const int phys = ggml_cuda_info().devices[ctx.device].physical_device;
            if (base[phys] == nullptr) {
                CUDA_CHECK(cudaGetSymbolAddress((void **) &base[phys], (const void *) &g_topk_mb));
            }
            topk_mb_scratch * sc = base[phys] + (ctx.device*GGML_CUDA_MAX_STREAMS + ctx.curr_stream_no);
            top_k_radix_mb<256><<<(int) nrows*bpr, 256, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k, bpr, sc);
        } else if (nrows < 64) {
            top_k_radix_row<1024><<<(int) nrows, 1024, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
        } else {
            top_k_radix_row<256><<<(int) nrows, 256, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k);
        }
    } else if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else if (ncols <= 1024 && nrows <= INT_MAX && [] { const char * e = getenv("GGML_CUDA_TOPK_BITONIC_LDS"); return !e || atoi(e) != 0; }()) {
        const int npad = top_k_next_pow2((int) ncols);
#define TOPK_BITONIC(N) top_k_bitonic_lds<N><<<(int) nrows, N, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k)
        switch (npad) {
            case 1: case 2: case 4: case 8: case 16: case 32: TOPK_BITONIC(32); break;
            case 64:   TOPK_BITONIC(64);   break;
            case 128:  TOPK_BITONIC(128);  break;
            case 256:  TOPK_BITONIC(256);  break;
            case 512:  TOPK_BITONIC(512);  break;
            default:   TOPK_BITONIC(1024); break;
        }
#undef TOPK_BITONIC
        CUDA_CHECK(cudaGetLastError());
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
