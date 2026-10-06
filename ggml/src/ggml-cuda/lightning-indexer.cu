#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// ---- decode (few query rows per stream) indexer for f16 K and 128-dim heads, GCN ----
// score[c] = sum_h w[h]*relu(q[h] . k[c]) + mask[c]. Lane = head (NH = 32: lanes 32..63 take a second cell), q[h] held in
// registers as f16 pairs (v_dot2_f32_f16, f32 accumulation), K rows staged through LDS in chunks of 64 cells (coalesced
// 16-byte loads, broadcast reads), the head sum of a cell by DPP. The generic kernel spent most of its time in per-head
// warp shuffles and LDS round trips (~65 us at 256 cells, 4 workgroups). Query row = blockIdx.y: small verify ubatches
// (DeepSeek V4 DSpark, 64 heads, 6 rows at 256 cells: lidp 33 us, generic 118 us) run their rows side by side.
static constexpr int LID1_CHUNK = 64;

template <int ctrl, int row_mask = 0xF, bool bound = true>
static __device__ __forceinline__ float lid1_dpp(const float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, row_mask, 0xF, bound));
}

template <int NH, bool QLDS>
static __global__ void __launch_bounds__(256) lid1_decode_f16(
        const float * __restrict__ Q, const char * __restrict__ K, const float * __restrict__ W, const half * __restrict__ M,
        float * __restrict__ dst, const int n_kv,
        const size_t nb3, const size_t nbq1, const size_t nbq3, const size_t nbk2, const size_t nbk3, const size_t nbw3,
        const size_t nbm3, const int64_t nem3, const size_t nb1, const size_t nbq2, const size_t nbw1, const size_t nbm1) {
#if defined(GGML_USE_HIP)
    typedef _Float16 h2v __attribute__((ext_vector_type(2)));
    constexpr int CPP = 64/NH;               // cells per wave pass
    constexpr int CPW = LID1_CHUNK/4;        // cells per wave
    const int i_stream = blockIdx.z;
    const int i_row    = blockIdx.y;
    Q   = (const float *) ((const char *) Q + i_row*nbq2);
    W   = (const float *) ((const char *) W + i_row*nbw1);
    M   = (const half  *) ((const char *) M + i_row*nbm1);
    dst = (float       *) ((char       *) dst + i_row*nb1);
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = lane % NH;
    const int hf   = lane / NH;

    __shared__ uint4 ks[LID1_CHUNK][16 + 1]; // 256 B rows (+16 B pad)

    // the chunk's K rows first (vector memory completes in issue order), then q and w
    const int c_beg = blockIdx.x*LID1_CHUNK;
    uint4 kl[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = tid + 256*j;
        const int c = c_beg + e/16;
        kl[j] = c < n_kv ? ((const uint4 *) (K + i_stream*nbk3 + (size_t) c*nbk2))[e % 16] : make_uint4(0, 0, 0, 0);
    }
    // the query once per workgroup (all NH heads x 128 dims -> f16 pairs in LDS), then each lane's head into registers:
    // loading it per lane from L2 read NH*512 B per wave (128 KB per workgroup, ~17 MB per call at 16K context)
    // (GGML_CUDA_LID1_QLDS=0: per-lane loads)
    __shared__ h2v qsh[QLDS ? NH : 1][QLDS ? 64 + 1 : 1];
    h2v qv[64];
    if (QLDS) {
        constexpr int NQ4 = NH*32/256;       // float4 of the query per thread
        float4 ql[NQ4];
#pragma unroll
        for (int j = 0; j < NQ4; ++j) {
            const int e = tid + 256*j;
            ql[j] = ((const float4 *) ((const char *) Q + i_stream*nbq3 + (e/32)*nbq1))[e % 32];
        }
#pragma unroll
        for (int j = 0; j < NQ4; ++j) {
            const int e = tid + 256*j;
            qsh[e/32][2*(e % 32) + 0] = h2v{(_Float16) ql[j].x, (_Float16) ql[j].y};
            qsh[e/32][2*(e % 32) + 1] = h2v{(_Float16) ql[j].z, (_Float16) ql[j].w};
        }
    } else {
        const float4 * qp = (const float4 *) ((const char *) Q + i_stream*nbq3 + h*nbq1);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float4 v = qp[i];
            qv[2*i + 0] = h2v{(_Float16) v.x, (_Float16) v.y};
            qv[2*i + 1] = h2v{(_Float16) v.z, (_Float16) v.w};
        }
    }
    const float wh = ((const float *) ((const char *) W + i_stream*nbw3))[h];
    const half * mrow = (const half *) ((const char *) M + (i_stream % nem3)*nbm3);
    float * drow = (float *) ((char *) dst + i_stream*nb3);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = tid + 256*j;
        ks[e/16][e % 16] = kl[j];
    }
    __syncthreads();
    if (QLDS) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            qv[i] = qsh[h][i];
        }
    }

#pragma unroll 1
    for (int p = 0; p < CPW; p += CPP) {
        const int cell = wv*CPW + p + hf;
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint4 kk = ks[cell][i];
            acc = __builtin_amdgcn_fdot2(qv[4*i + 0], __builtin_bit_cast(h2v, kk.x), acc, false);
            acc = __builtin_amdgcn_fdot2(qv[4*i + 1], __builtin_bit_cast(h2v, kk.y), acc, false);
            acc = __builtin_amdgcn_fdot2(qv[4*i + 2], __builtin_bit_cast(h2v, kk.z), acc, false);
            acc = __builtin_amdgcn_fdot2(qv[4*i + 3], __builtin_bit_cast(h2v, kk.w), acc, false);
        }
        float sc = fmaxf(acc, 0.0f)*wh;
        // sum over the cell's NH lanes: rows of 16, then row 1 += row 0 (and row 3 += row 2), [NH 64: row 3 += rows 0-1]
        sc += lid1_dpp<0xB1>(sc);
        sc += lid1_dpp<0x4E>(sc);
        sc += lid1_dpp<0x141>(sc);
        sc += lid1_dpp<0x140>(sc);
        sc += lid1_dpp<0x142, 0xA, false>(sc);   // row_bcast:15
        if (NH == 64) {
            sc += lid1_dpp<0x143, 0xC, false>(sc);   // row_bcast:31
        }
        const int c = c_beg + cell;
        if (h == NH - 1 && c < n_kv) {
            drow[c] = sc + __half2float(mrow[c]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_kv, nb3, nbq1, nbq3, nbk2, nbk3, nbw3, nbm3, nem3, nb1, nbq2, nbw1, nbm1);
#endif
}

// ---- prefill indexer (many query rows per stream) for f16 K, 128-dim heads, 32 or 64 heads, GCN ----
// out[q][c] = sum_h w[q][h]*relu(q[q][h] . k[c]) + mask[q][c]. Workgroup = 4 query rows x 64 cells, wave = query row,
// thread = 8 heads (h = hg + 4i) x 4 cells (c = cg + 16j): the (row, head) x cell products run as a register-tiled f16
// GEMM (v_dot2_f32_f16, f32 accumulation like lid1_decode_f16) over two LDS chunks of 64 dims, the query converted to f16
// on the way in. The weighted relu sum runs over the thread's 8 heads in order, then over the 4 head groups (xor 16, 32).
// Tiles whose mask is all -inf (causal) only write the mask. The generic vec kernel took one cell per warp with a 5-step
// shuffle per (cell, head): ~3 TFLOPS, 25 ms per call at 16K context for a 1024-row ubatch.
// NH = 64 (DeepSeek V4's indexer): two passes of 32 heads (the K tile is loaded per pass), their weighted relu sums
// added in pass order; before it, 64-head verify / prefill ubatches took the generic vec kernel (DeepSeek V4 Flash
// 6-token verify: ~2.5 ms of 41 ms in it).
static constexpr int LIDP_TQ = 4;
static constexpr int LIDP_TC = 64;
static constexpr int LIDP_DC = 64;
static constexpr int LIDP_NH = 32; // heads per pass

template <int NH>
static __global__ void __launch_bounds__(256) lidp_prefill_f16(
        const float * __restrict__ Q, const char * __restrict__ K, const float * __restrict__ W, const half * __restrict__ M,
        float * __restrict__ dst, const int n_kv, const int n_batch,
        const size_t nb1, const size_t nb3, const size_t nbq1, const size_t nbq2, const size_t nbq3,
        const size_t nbk2, const size_t nbk3, const size_t nbw1, const size_t nbw3, const size_t nbm1, const size_t nbm3,
        const int64_t nem3) {
#if defined(GGML_USE_HIP)
    typedef _Float16 h2v __attribute__((ext_vector_type(2)));
    constexpr int RW = LIDP_DC/8 + 1; // uint4 per LDS row (+16 B pad)
    __shared__ uint4 qs[LIDP_TQ][LIDP_NH][RW];
    __shared__ uint4 ks[LIDP_TC][RW];

    const int tid      = threadIdx.x;
    const int lane     = tid & 63;
    const int qi       = tid >> 6;
    const int hg       = lane >> 4;
    const int cg       = lane & 15;
    const int c0       = blockIdx.x*LIDP_TC;
    const int q0       = blockIdx.y*LIDP_TQ;
    const int i_stream = blockIdx.z;
    const half * mbase = (const half *) ((const char *) M + (i_stream % nem3)*nbm3);

    // causal tiles: nothing visible -> the output is the mask
    {
        const int r = tid / LIDP_TC;
        const int c = c0 + tid % LIDP_TC;
        bool vis = false;
        if (q0 + r < n_batch && c < n_kv) {
            vis = __half2float(mbase[(size_t) (q0 + r)*(nbm1/sizeof(half)) + c]) != -INFINITY;
        }
        if (!__syncthreads_or(vis)) {
            if (q0 + r < n_batch && c < n_kv) {
                float * drow = (float *) ((char *) dst + (size_t) (q0 + r)*nb1 + i_stream*nb3);
                drow[c] = -INFINITY;
            }
            return;
        }
    }

    const int  q   = q0 + qi;
    const bool qok = q < n_batch;
    const float * wrow = (const float *) ((const char *) W + (size_t) min(q, n_batch - 1)*nbw1 + i_stream*nbw3);
    float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 1
    for (int hp = 0; hp < NH; hp += LIDP_NH) {
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = 0.0f;
            }
        }

#pragma unroll 1
        for (int d0 = 0; d0 < 128; d0 += LIDP_DC) {
            // q: 4 rows x 32 heads x 64 dims (f32) -> f16, 8 dims per op (4 per thread); k: 64 cells x 64 dims (2 per thread)
            float4 ql[4][2];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e  = tid + 256*u;          // 0 .. 1023: (row, head, 8-dim group)
                const int r  = e / (LIDP_NH*(LIDP_DC/8));
                const int h  = (e / (LIDP_DC/8)) % LIDP_NH;
                const int g  = e % (LIDP_DC/8);
                const int qr = min(q0 + r, n_batch - 1);
                const float4 * src = (const float4 *) ((const char *) Q + (size_t) qr*nbq2 + i_stream*nbq3 + (hp + h)*nbq1 +
                    (d0 + 8*g)*sizeof(float));
                ql[u][0] = src[0];
                ql[u][1] = src[1];
            }
            uint4 kl[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int e = tid + 256*u;           // 0 .. 511: (cell, 8-dim group)
                const int c = c0 + e / (LIDP_DC/8);
                const int g = e % (LIDP_DC/8);
                kl[u] = c < n_kv ? *(const uint4 *) (K + i_stream*nbk3 + (size_t) c*nbk2 + (d0 + 8*g)*sizeof(half)) :
                    make_uint4(0, 0, 0, 0);
            }
            if (hp > 0 || d0 > 0) {
                __syncthreads(); // the previous chunk's reads are done
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e = tid + 256*u;
                const int r = e / (LIDP_NH*(LIDP_DC/8));
                const int h = (e / (LIDP_DC/8)) % LIDP_NH;
                const int g = e % (LIDP_DC/8);
                const h2v a = h2v{(_Float16) ql[u][0].x, (_Float16) ql[u][0].y};
                const h2v b = h2v{(_Float16) ql[u][0].z, (_Float16) ql[u][0].w};
                const h2v c = h2v{(_Float16) ql[u][1].x, (_Float16) ql[u][1].y};
                const h2v d = h2v{(_Float16) ql[u][1].z, (_Float16) ql[u][1].w};
                qs[r][h][g] = make_uint4(__builtin_bit_cast(uint32_t, a), __builtin_bit_cast(uint32_t, b),
                                         __builtin_bit_cast(uint32_t, c), __builtin_bit_cast(uint32_t, d));
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int e = tid + 256*u;
                ks[e / (LIDP_DC/8)][e % (LIDP_DC/8)] = kl[u];
            }
            __syncthreads();

#pragma unroll
            for (int g = 0; g < LIDP_DC/8; ++g) {
                uint4 kk[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    kk[j] = ks[cg + 16*j][g];
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint4 qq = qs[qi][hg + 4*i][g];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float a = acc[i][j];
                        a = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2v, qq.x), __builtin_bit_cast(h2v, kk[j].x), a, false);
                        a = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2v, qq.y), __builtin_bit_cast(h2v, kk[j].y), a, false);
                        a = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2v, qq.z), __builtin_bit_cast(h2v, kk[j].z), a, false);
                        a = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2v, qq.w), __builtin_bit_cast(h2v, kk[j].w), a, false);
                        acc[i][j] = a;
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float wh = wrow[hp + hg + 4*i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[j] += fmaxf(acc[i][j], 0.0f)*wh;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j] += __shfl_xor(sc[j], 16, 64);
        sc[j] += __shfl_xor(sc[j], 32, 64);
    }
    if (hg == 0 && qok) {
        float * drow = (float *) ((char *) dst + (size_t) q*nb1 + i_stream*nb3);
        const half * mrow = mbase + (size_t) q*(nbm1/sizeof(half));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + cg + 16*j;
            if (c < n_kv) {
                drow[c] = sc[j] + __half2float(mrow[c]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_kv, n_batch, nb1, nb3, nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
#endif
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

static void ggml_cuda_lightning_indexer_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    // decode on GCN: lid1_decode_f16 (GGML_CUDA_LID1=0 off), also for verify ubatches of up to GGML_CUDA_LID1_MAX_ROWS
    // query rows (default 8) while rows x cells <= 8192 (64 heads, 6 rows: 256 cells 14.8 us vs lidp 32.7, 1024: 28.4 vs
    // 33.1, 4096: 95.8 vs 69.2)
    static const bool lid1 = [] { const char * e = getenv("GGML_CUDA_LID1"); return !e || atoi(e) != 0; }();
    static const int lid1_max_rows = [] { const char * e = getenv("GGML_CUDA_LID1_MAX_ROWS"); return e ? atoi(e) : 8; }();
    if (lid1 && GGML_CUDA_CC_IS_GCN(cc) && (n_batch == 1 || (n_batch <= lid1_max_rows && (int64_t) n_batch*n_kv <= 8192)) && n_embd == 128 &&
            (n_head == 32 || n_head == 64) && k->type == GGML_TYPE_F16 && nbq1 % 16 == 0 && nbq2 % 16 == 0 &&
            nbk2 % 16 == 0 && ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && nbq3 % 16 == 0 &&
            nbk3 % 16 == 0 && nbm1 % sizeof(half) == 0) {
        const dim3 grid((n_kv + LID1_CHUNK - 1)/LID1_CHUNK, n_batch, n_stream);
        static const bool qlds = [] { const char * e = getenv("GGML_CUDA_LID1_QLDS"); return !e || atoi(e) != 0; }();
#define LID1_LAUNCH(NH, QL) lid1_decode_f16<NH, QL><<<grid, 256, 0, ctx.stream()>>>(q_d, k_d, w_d, m_d, dst_d, n_kv, nb3, \
            nbq1, nbq3, nbk2, nbk3, nbw3, nbm3, nem3, nb1, nbq2, nbw1, nbm1)
        if (n_head == 32) {
            if (qlds) { LID1_LAUNCH(32, true); } else { LID1_LAUNCH(32, false); }
        } else {
            if (qlds) { LID1_LAUNCH(64, true); } else { LID1_LAUNCH(64, false); }
        }
#undef LID1_LAUNCH
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // prefill on GCN: lidp_prefill_f16 (GGML_CUDA_LIDP=0 off)
    static const bool lidp = [] { const char * e = getenv("GGML_CUDA_LIDP"); return !e || atoi(e) != 0; }();
    if (lidp && GGML_CUDA_CC_IS_GCN(cc) && n_batch > 1 && n_embd == 128 && (n_head == 32 || n_head == 64) && k->type == GGML_TYPE_F16 &&
            nbq1 % 16 == 0 && nbq2 % 16 == 0 && nbq3 % 16 == 0 && nbk2 % 16 == 0 && nbk3 % 16 == 0 &&
            ((uintptr_t) q_d) % 16 == 0 && ((uintptr_t) k_d) % 16 == 0 && nbm1 % sizeof(half) == 0 && n_batch <= 65535*LIDP_TQ) {
        const dim3 grid((n_kv + LIDP_TC - 1)/LIDP_TC, (n_batch + LIDP_TQ - 1)/LIDP_TQ, n_stream);
        if (n_head == 64) {
            lidp_prefill_f16<64><<<grid, 256, 0, ctx.stream()>>>(q_d, k_d, w_d, m_d, dst_d, n_kv, n_batch, nb1, nb3, nbq1, nbq2, nbq3,
                nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
        } else {
            lidp_prefill_f16<32><<<grid, 256, 0, ctx.stream()>>>(q_d, k_d, w_d, m_d, dst_d, n_kv, n_batch, nb1, nb3, nbq1, nbq2, nbq3,
                nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

// f32 keys on GCN (GLM-5-Next's pooled index keys: f16 cache values widened to f32 by get_rows, so the round trip is
// exact): converted to f16 so the decode (lid1) and prefill (lidp) kernels apply instead of the f32 path of the generic
// vector kernel. GGML_CUDA_LID_K16=0 off
void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[1];
    static const bool k16 = [] { const char * e = getenv("GGML_CUDA_LID_K16"); return !e || atoi(e) != 0; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (k16 && k->type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_GCN(cc) && k->ne[0] == 128 && ggml_is_contiguous(k)) {
        const int64_t ne = ggml_nelements(k);
        ggml_cuda_pool_alloc<half> kh(ctx.pool(), ne);
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(k->data, kh.get(), ne, ctx.stream());
        ggml_tensor k2 = *k;
        k2.type  = GGML_TYPE_F16;
        k2.data  = kh.get();
        k2.nb[0] = sizeof(half);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            k2.nb[i] = k2.nb[i - 1]*k2.ne[i - 1];
        }
        k2.view_src  = nullptr;
        k2.view_offs = 0;
        ggml_tensor d2 = *dst;
        d2.src[1] = &k2;
        ggml_cuda_lightning_indexer_impl(ctx, &d2);
        return;
    }
    ggml_cuda_lightning_indexer_impl(ctx, dst);
}
