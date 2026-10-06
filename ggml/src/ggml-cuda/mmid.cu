#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
    constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// Stable counting sort: tokens are split into chunks of MM_IDS_CHUNK. Pass 1 counts slots per (chunk, expert),
// pass 2 turns the counts into per-(chunk, expert) write offsets and the expert bounds, pass 3 walks each chunk's
// tokens in order and scatters. The top-k experts of one token are distinct, so within one token no two lanes
// bump the same shared counter.
static constexpr int MM_IDS_CHUNK = 64;

static __global__ void mm_ids_count(const int32_t * __restrict__ ids, int32_t * __restrict__ counts,
        const int n_experts, const int n_tokens, const int n_expert_used, const int si1) {
    extern __shared__ int cnt[];
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        cnt[e] = 0;
    }
    __syncthreads();
    const int it0 = blockIdx.x*MM_IDS_CHUNK;
    const int it1 = min(it0 + MM_IDS_CHUNK, n_tokens);
    for (int s = threadIdx.x; s < (it1 - it0)*n_expert_used; s += blockDim.x) {
        const int it  = it0 + s / n_expert_used;
        const int iex = s % n_expert_used;
        atomicAdd(&cnt[ids[it*si1 + iex]], 1);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        counts[blockIdx.x*n_experts + e] = cnt[e];
    }
}

// one block, one thread per expert (n_experts <= 1024)
static __global__ void mm_ids_offsets(int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_chunks) {
    extern __shared__ int scan[];
    const int e = threadIdx.x;
    int total = 0;
    if (e < n_experts) {
        for (int c = 0; c < n_chunks; ++c) {
            total += counts[c*n_experts + e];
        }
    }
    scan[e] = total;
    __syncthreads();
    for (int off = 1; off < (int) blockDim.x; off *= 2) {
        const int v = e >= off ? scan[e - off] : 0;
        __syncthreads();
        scan[e] += v;
        __syncthreads();
    }
    if (e >= n_experts) {
        return;
    }
    int pos = scan[e] - total; // exclusive prefix over experts
    expert_bounds[e] = pos;
    if (e == n_experts - 1) {
        expert_bounds[n_experts] = scan[e];
    }
    for (int c = 0; c < n_chunks; ++c) { // counts -> write offsets, in place
        const int n = counts[c*n_experts + e];
        counts[c*n_experts + e] = pos;
        pos += n;
    }
}

static __global__ void mm_ids_scatter(const int32_t * __restrict__ ids, const int32_t * __restrict__ offsets,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1,
        const bool write_inverse) {
    extern __shared__ int off[];
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        off[e] = offsets[blockIdx.x*n_experts + e];
    }
    __syncthreads();
    const int it0 = blockIdx.x*MM_IDS_CHUNK;
    const int it1 = min(it0 + MM_IDS_CHUNK, n_tokens);
    for (int it = it0; it < it1; ++it) {
        for (int iex = threadIdx.x; iex < n_expert_used; iex += blockDim.x) {
            const int pos = off[ids[it*si1 + iex]]++;
            ids_dst[pos] = it*n_expert_used + iex;
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
        }
        __syncthreads(); // the next token may use the same experts
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream,
        ggml_cuda_pool * pool) {
    static const bool no_sort = [] { const char * e = getenv("GGML_MM_IDS_NO_SORT"); return e && atoi(e) != 0; }();
    if (pool && !no_sort && n_tokens >= 2*MM_IDS_CHUNK && n_experts <= 1024 && n_expert_used <= 1024) {
        const int n_chunks = (n_tokens + MM_IDS_CHUNK - 1) / MM_IDS_CHUNK;
        ggml_cuda_pool_alloc<int32_t> counts(*pool, (size_t) n_chunks*n_experts);
        const size_t smem = n_experts*sizeof(int);
        mm_ids_count<<<n_chunks, 256, smem, stream>>>(ids, counts.get(), n_experts, n_tokens, n_expert_used, si1);
        const int threads = std::max(32, 1 << (32 - __builtin_clz(std::max(n_experts - 1, 1))));
        mm_ids_offsets<<<1, threads, threads*sizeof(int), stream>>>(counts.get(), expert_bounds, n_experts, n_chunks);
        const int scatter_threads = n_expert_used <= 32 ? 32 : 64;
        mm_ids_scatter<<<n_chunks, scatter_threads, smem, stream>>>(ids, counts.get(), ids_src1, ids_dst,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
        return;
    }
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
