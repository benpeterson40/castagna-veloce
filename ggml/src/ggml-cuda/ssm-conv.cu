#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Deferred CONCAT(conv_state, transpose(x)) for the gated-DeltaNet short convolution: instead of materializing the
// [n_t + 3, d_inner] concat, the convolution (+SiLU) and the conv-state tail read the state and x directly.
// v(tau, c) = tau < 0 ? state[c*3 + tau + 3] : x[tau*sx_t + c*sx_c]
static __device__ __forceinline__ float ssm_conv_src(const float * __restrict__ state, const float * __restrict__ x,
                                                     const int64_t tau, const int c, const int64_t sx_t, const int64_t sx_c) {
    return tau < 0 ? state[c*3 + (tau + 3)] : x[tau*sx_t + c*sx_c];
}

// One thread per channel walks the tokens in order: dst may alias x (the allocator may reuse x's memory for dst,
// same [n_t, d_inner] layout), and a thread only overwrites positions of its own channel it has already read.
static constexpr int SSM_CONV_DEFER_PF = 8; // tokens loaded ahead

static __global__ void ssm_conv_deferred_silu_f32(const float * state, const float * x,
        const float * __restrict__ w, float * dst, const int n_c, const int64_t n_t,
        const int64_t sx_t, const int64_t sx_c, const int64_t sw, const int64_t sd) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_c) {
        return;
    }
    const float w0 = w[c*sw + 0], w1 = w[c*sw + 1], w2 = w[c*sw + 2], w3 = w[c*sw + 3];
    float a = ssm_conv_src(state, x, -3, c, sx_t, sx_c);
    float b = ssm_conv_src(state, x, -2, c, sx_t, sx_c);
    float d = ssm_conv_src(state, x, -1, c, sx_t, sx_c);
    for (int64_t t0 = 0; t0 < n_t; t0 += SSM_CONV_DEFER_PF) {
        float e[SSM_CONV_DEFER_PF];
#pragma unroll
        for (int k = 0; k < SSM_CONV_DEFER_PF; ++k) {
            e[k] = t0 + k < n_t ? x[(t0 + k)*sx_t + c*sx_c] : 0.0f;
        }
#pragma unroll
        for (int k = 0; k < SSM_CONV_DEFER_PF; ++k) {
            if (t0 + k < n_t) {
                dst[(t0 + k)*sd + c] = ggml_cuda_op_silu_single(a*w0 + b*w1 + d*w2 + e[k]*w3);
            }
            a = b; b = d; d = e[k];
        }
    }
}

static __global__ void ssm_conv_deferred_tail_f32(const float * __restrict__ state, const float * __restrict__ x,
        float * __restrict__ dst, const int n_c, const int64_t n_t, const int64_t sx_t, const int64_t sx_c) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_c) {
        return;
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        dst[c*3 + j] = ssm_conv_src(state, x, n_t - 3 + j, c, sx_t, sx_c);
    }
}

// concat = CONCAT(state [3, d_inner], xt [n_t, d_inner] (a transposed view)); conv = SSM_CONV(concat, w); silu = SILU(conv)
bool ggml_cuda_ssm_conv_deferred_ok(const ggml_tensor * concat, const ggml_tensor * conv, const ggml_tensor * silu) {
    static const bool disabled = [] { const char * e = getenv("GGML_SSM_CONV_NO_DEFER"); return e && atoi(e) != 0; }();
    if (disabled) {
        return false;
    }
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    const ggml_tensor * w     = conv->src[1];
    return ggml_get_op_params_i32(concat, 0) == 0 && concat->type == GGML_TYPE_F32 && state->type == GGML_TYPE_F32 &&
        xt->type == GGML_TYPE_F32 && state->ne[0] == 3 && ggml_is_contiguous(state) && state->ne[1] == xt->ne[1] &&
        state->ne[2] == 1 && xt->ne[2] == 1 && w->type == GGML_TYPE_F32 && w->ne[0] == 4 && w->nb[0] == sizeof(float) &&
        conv->src[0] == concat && conv->ne[2] == 1 && silu->type == GGML_TYPE_F32 && silu->nb[0] == sizeof(float) &&
        ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == conv;
}

// Runs at the CONCAT node while x and the state are still live: the tail goes to a temporary first (the conv may
// overwrite x in place), then the convolution into silu, then the tail into the CONT node's buffer.
void ggml_cuda_ssm_conv_deferred(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * conv,
                                 ggml_tensor * silu, ggml_tensor * tail) {
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    const ggml_tensor * w     = conv->src[1];
    const int     n_c = state->ne[1];
    const int64_t n_t = xt->ne[0];
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), (size_t) 3*n_c);
    ssm_conv_deferred_tail_f32<<<(n_c + 255)/256, 256, 0, stream>>>((const float *) state->data, (const float *) xt->data,
        tmp.get(), n_c, n_t, xt->nb[0]/sizeof(float), xt->nb[1]/sizeof(float));
    ssm_conv_deferred_silu_f32<<<(n_c + 63)/64, 64, 0, stream>>>((const float *) state->data, (const float *) xt->data,
        (const float *) w->data, (float *) silu->data, n_c, n_t, xt->nb[0]/sizeof(float), xt->nb[1]/sizeof(float),
        w->nb[1]/sizeof(float), silu->nb[1]/sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(tail->data, tmp.get(), (size_t) 3*n_c*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaGetLastError());
}

// Decode / verify gated-DeltaNet conv block in one kernel (replaces CONCAT, the conv-state tail CONT + CPY per rollback
// slot, SSM_CONV + SILU and the q/k l2 norms RMS_NORM + SCALE): one block per 128-channel head, one thread per
// channel. The thread reads its 3 state values and the n_t new inputs, writes the SILU outputs, the conv-state tails
// straight into the cache rows, and for q/k heads the block reduces the per-token sum of squares and writes the
// normalized head. 5 launches -> 1 per GDN layer at decode.
static constexpr int GDN_CONV_MAX_T     = 16;
static constexpr int GDN_CONV_MAX_SLOTS = 8;

struct gdn_conv_block_args {
    const float * state; const float * x; const float * w;
    float * silu; float * qn; float * kn;
    float * tail[GDN_CONV_MAX_SLOTS]; int tail_start[GDN_CONV_MAX_SLOTS]; int n_slots;
    int n_c; int n_t;
    int64_t sx_t, sx_c, sw, sd;          // x (via the transposed view), weight row, silu token stride (floats)
    int q_c0, k_c0, n_qh, n_kh;           // first channel and heads of the q / k views
    int64_t q_st, k_st;                   // normalized q / k token strides (floats); heads are 128 apart
    float q_eps, k_eps, q_scale, k_scale;
    unsigned int * bar;                   // non-null: outputs overlap inputs, grid barrier between all reads and all writes
    const int32_t * state_ids;            // non-null: read the state row state_ids[0] of the cache (state = its base) directly,
    int64_t state_row;                    // the GET_ROWS gather is skipped (row stride in floats)
};

// all blocks of the conv-block kernel are co-resident (n_c/128 blocks of 128 threads), so a spinning barrier is safe;
// per (virtual device, stream) slot: [0] arrivals, [1] generation
static __device__ unsigned int g_gdn_conv_bar[GGML_CUDA_MAX_DEVICES*GGML_CUDA_MAX_STREAMS][2];

// only ordering matters here (every read before any write): each block's loads complete, then the arrival counter
// bar[0] and generation bar[1] on the scalar memory path (vector atomics, fences and polls queue behind other memory ops)
static __device__ __forceinline__ void gdn_conv_grid_barrier(unsigned int * bar) {
#if defined(GGML_USE_HIP) && !defined(__GFX9__)
    // other targets (RDNA: no scalar atomics / scalar stores): the same arrival + generation protocol with vector
    // atomics at device scope
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int * arrive = bar, * gen = bar + 1;
        const unsigned int g = __hip_atomic_load(gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned int old = atomicAdd(arrive, 1u);
        if (old == gridDim.x - 1) {
            atomicExch(arrive, 0u);
            __threadfence();
            atomicAdd(gen, 1u);
        } else {
            while (__hip_atomic_load(gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == g) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence();
    }
    __syncthreads();
#else
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x < 64) {
        const uint64_t pa = (uint64_t) (uintptr_t) bar;
        const uint64_t pu = ((uint64_t) (uint32_t) __builtin_amdgcn_readfirstlane((int) (pa >> 32)) << 32) |
                            (uint32_t) __builtin_amdgcn_readfirstlane((int) (uint32_t) pa);
        unsigned int * arrive = (unsigned int *) (uintptr_t) pu, * gen = arrive + 1;
        unsigned int g, old;
        __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(g) : "s"(gen) : "memory");
        __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(old) : "s"(arrive), "0"(1u) : "memory");
        if (old == gridDim.x - 1) {
            unsigned int z = 0, one = 1;
            __asm__ volatile("s_atomic_swap %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "+s"(z) : "s"(arrive) : "memory");
            __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "+s"(one) : "s"(gen) : "memory");
        } else {
            while (true) {
                unsigned int cur;
                __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(cur) : "s"(gen) : "memory");
                if (cur != g) { break; }
                __builtin_amdgcn_s_sleep(0);
            }
        }
    }
    __syncthreads();
#endif // defined(GGML_USE_HIP) && !defined(__GFX9__)
}

static __global__ void __launch_bounds__(128) gdn_conv_block_dec(const gdn_conv_block_args a) {
    const int c = blockIdx.x*128 + threadIdx.x;
    const int n_t = a.n_t;
    float seq[3 + GDN_CONV_MAX_T];
    const float * state = a.state_ids ? a.state + (int64_t) a.state_ids[0]*a.state_row : a.state;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        seq[j] = state[c*3 + j];
    }
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            seq[3 + t] = a.x[t*a.sx_t + c*a.sx_c];
        }
    }
    const float w0 = a.w[c*a.sw + 0], w1 = a.w[c*a.sw + 1], w2 = a.w[c*a.sw + 2], w3 = a.w[c*a.sw + 3];
    if (a.bar) {
        gdn_conv_grid_barrier(a.bar);
    }
    float y[GDN_CONV_MAX_T];
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            y[t] = ggml_cuda_op_silu_single(seq[t]*w0 + seq[t + 1]*w1 + seq[t + 2]*w2 + seq[t + 3]*w3);
            a.silu[t*a.sd + c] = y[t];
        }
    }
    for (int s = 0; s < a.n_slots; ++s) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            // seq index start+j (start <= n_t, so the index is < 3 + n_t)
            const int k = a.tail_start[s] + j;
            float v = seq[0];
#pragma unroll
            for (int q = 1; q < 3 + GDN_CONV_MAX_T; ++q) {
                v = q == k ? seq[q] : v;
            }
            a.tail[s][c*3 + j] = v;
        }
    }
    const int c0 = blockIdx.x*128;
    const bool is_q = c0 >= a.q_c0 && c0 < a.q_c0 + a.n_qh*128;
    const bool is_k = c0 >= a.k_c0 && c0 < a.k_c0 + a.n_kh*128;
    if (!is_q && !is_k) {
        return;
    }
    __shared__ float red[2][GDN_CONV_MAX_T];
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    constexpr int NW = 128/WARP_SIZE;
    static_assert(NW <= 2*GDN_CONV_MAX_T, "");
    __shared__ float part[NW][GDN_CONV_MAX_T];
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            const float ss = warp_reduce_sum(y[t]*y[t]);
            if (lane == 0) {
                part[wid][t] = ss;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < n_t) {
        float tot = 0.0f;
#pragma unroll
        for (int i = 0; i < NW; ++i) {
            tot += part[i][threadIdx.x];
        }
        red[0][threadIdx.x] = tot;
    }
    __syncthreads();
    const float eps = is_q ? a.q_eps : a.k_eps, scale = is_q ? a.q_scale : a.k_scale;
    float * out = is_q ? a.qn : a.kn;
    const int h = (c0 - (is_q ? a.q_c0 : a.k_c0))/128;
    const int64_t st = is_q ? a.q_st : a.k_st;
    for (int t = 0; t < n_t; ++t) {
        const float r = rsqrtf(red[0][t]/128.0f + eps);
        out[t*st + h*128 + threadIdx.x] = (y[t]*r)*scale;
    }
}

bool ggml_cuda_gdn_conv_block_dec_ok(const ggml_tensor * concat, const ggml_tensor * conv, const ggml_tensor * silu,
                                     const ggml_tensor * const * tails_cpy, int n_slots,
                                     const ggml_tensor * q_rms, const ggml_tensor * q_scale,
                                     const ggml_tensor * k_rms, const ggml_tensor * k_scale, bool * need_barrier) {
    static const bool disabled = [] { const char * e = getenv("GGML_GDN_CONV_BLOCK_DEC"); return e && atoi(e) == 0; }();
    *need_barrier = false;
    static const bool dbg = getenv("GGML_GDN_CONV_BLOCK_DBG") != nullptr;
#define CB_FAIL(n) do { if (dbg) { fprintf(stderr, "convblock ok fail %d\n", n); } return false; } while (0)
    if (disabled || n_slots < 1 || n_slots > GDN_CONV_MAX_SLOTS || !ggml_cuda_ssm_conv_deferred_ok(concat, conv, silu)) {
        CB_FAIL(1);
    }
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    const int64_t n_c = state->ne[1], n_t = xt->ne[0];
    if (n_t < 1 || n_t > GDN_CONV_MAX_T || n_c % 128 != 0 || silu->ne[0] != n_c || silu->ne[1] != n_t || silu->ne[2] != 1) {
        CB_FAIL(2);
    }
    for (const ggml_tensor * qk : { q_rms, k_rms }) {
        const ggml_tensor * v = qk->src[0]; // view of silu
        if (v->op != GGML_OP_VIEW || v->src[0] != silu || v->ne[0] != 128 || v->ne[2] != n_t || v->ne[3] != 1 ||
                v->nb[0] != sizeof(float) || v->nb[1] != 128*sizeof(float) || v->nb[2] != silu->nb[1] ||
                v->view_offs % (128*sizeof(float)) != 0 || v->view_offs/sizeof(float) + v->ne[1]*128 > (size_t) n_c) {
            CB_FAIL(3);
        }
    }
    for (const ggml_tensor * sc : { q_scale, k_scale }) {
        if (sc->type != GGML_TYPE_F32 || ggml_get_op_params_f32(sc, 1) != 0.0f || !ggml_is_contiguous(sc) ||
                !ggml_are_same_shape(sc, sc->src[0])) {
            CB_FAIL(4);
        }
    }
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    const ggml_tensor * w = conv->src[1];
    // outputs over the (dead) inputs state / x need the grid barrier (silu exactly replacing x does not: each thread
    // reads its channel before writing it); outputs may not overlap each other or the weights
    if ((overlaps(silu, xt) && !(silu->data == xt->data && silu->nb[1] == xt->nb[0] && xt->nb[1] == sizeof(float))) ||
            overlaps(silu, state)) {
        *need_barrier = true;
    }
    for (const ggml_tensor * o : { q_scale, k_scale }) {
        if (overlaps(o, state) || overlaps(o, xt)) {
            *need_barrier = true;
        }
    }
    if (overlaps(silu, w) || overlaps(q_scale, w) || overlaps(k_scale, w) || overlaps(q_scale, silu) ||
            overlaps(k_scale, silu) || overlaps(q_scale, k_scale)) {
        CB_FAIL(5);
    }
    for (int s = 0; s < n_slots; ++s) {
        const ggml_tensor * cpy = tails_cpy[s];
        const ggml_tensor * d   = cpy->src[1];
        if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != 3*n_c) {
            CB_FAIL(8);
        }
        for (const ggml_tensor * o : { state, xt, silu, w, q_scale, k_scale }) {
            if (overlaps(d, o)) {
                CB_FAIL(9);
            }
        }
    }
    if (dbg) {
        fprintf(stderr, "convblock barrier %d n_t %d\n", (int) *need_barrier, (int) n_t);
    }
    return true;
#undef CB_FAIL
}

void ggml_cuda_gdn_conv_block_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * conv,
                                  ggml_tensor * silu, const ggml_tensor * const * tails_view, const ggml_tensor * const * tails_cpy,
                                  int n_slots, const ggml_tensor * q_rms, ggml_tensor * q_scale,
                                  const ggml_tensor * k_rms, ggml_tensor * k_scale, bool need_barrier, const ggml_tensor * state_rows) {
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    const ggml_tensor * w     = conv->src[1];
    gdn_conv_block_args a = {};
    a.state = (const float *) state->data;
    if (state_rows) {
        // the deferred GET_ROWS (cache rows, ids): each thread reads its channel before writing its tails, so a tail slot
        // that is the source row itself is safe
        a.state     = (const float *) state_rows->src[0]->data;
        a.state_ids = (const int32_t *) state_rows->src[1]->data;
        a.state_row = state_rows->src[0]->nb[1]/sizeof(float);
    }
    a.x     = (const float *) xt->data;
    a.w     = (const float *) w->data;
    a.silu  = (float *) silu->data;
    a.qn    = (float *) q_scale->data;
    a.kn    = (float *) k_scale->data;
    a.n_slots = n_slots;
    for (int s = 0; s < n_slots; ++s) {
        a.tail[s]       = (float *) tails_cpy[s]->src[1]->data;
        a.tail_start[s] = (int) (tails_view[s]->view_offs / sizeof(float)); // column offset of the tail in the concat
    }
    a.n_c  = (int) state->ne[1];
    a.n_t  = (int) xt->ne[0];
    a.sx_t = xt->nb[0]/sizeof(float);
    a.sx_c = xt->nb[1]/sizeof(float);
    a.sw   = w->nb[1]/sizeof(float);
    a.sd   = silu->nb[1]/sizeof(float);
    a.q_c0 = (int) (q_rms->src[0]->view_offs/sizeof(float)); a.n_qh = (int) q_rms->src[0]->ne[1];
    a.k_c0 = (int) (k_rms->src[0]->view_offs/sizeof(float)); a.n_kh = (int) k_rms->src[0]->ne[1];
    a.q_st = q_scale->nb[2]/sizeof(float);
    a.k_st = k_scale->nb[2]/sizeof(float);
    a.q_eps = ggml_get_op_params_f32(q_rms, 0);   a.k_eps = ggml_get_op_params_f32(k_rms, 0);
    a.q_scale = ggml_get_op_params_f32(q_scale, 0); a.k_scale = ggml_get_op_params_f32(k_scale, 0);
    if (need_barrier) {
        static unsigned int * bar_base[GGML_CUDA_MAX_DEVICES] = {};
        const int phys = ggml_cuda_info().devices[ctx.device].physical_device;
        if (bar_base[phys] == nullptr) {
            CUDA_CHECK(cudaGetSymbolAddress((void **) &bar_base[phys], (const void *) &g_gdn_conv_bar));
        }
        a.bar = bar_base[phys] + 2*(ctx.device*GGML_CUDA_MAX_STREAMS + ctx.curr_stream_no);
    }
    gdn_conv_block_dec<<<a.n_c/128, 128, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

// GLM-5-Next (KDA) decode / verify conv block in one kernel. The model concatenates the q, k and v projections, and
// their conv weights, every token and convolves them as one [3*d_inner] block; this kernel reads the three projections
// and the three weight tensors directly and replaces the q|k and q|k|v CONCATs, the two weight CONCATs, the conv-state
// CONCAT, the conv-state tail CPY per rollback slot, SSM_CONV + SILU and the q / k L2_NORMs (10 launches -> 1 per KDA
// layer). One block per 128-channel head (the q heads, then k, then v), one thread per channel; the arithmetic of each
// output is that of the unfused ops (the L2 sums reduce in another order).
struct kda_conv_block_args {
    const float * x0; const float * x1; const float * x2;    // q, k, v projections [d_inner, n_t]
    int64_t sx0, sx1, sx2;                                    // their token strides (floats)
    const float * w0; const float * w1; const float * w2;    // their conv weights [4, d_inner] (contiguous)
    const float * state;                                      // conv state [3, 3*d_inner] (contiguous), or with
    const int32_t * state_ids; int64_t state_row;             // state_ids the cache base (row state_ids[0], row stride)
    float * silu; int64_t sd;                                 // SILU(conv) [3*d_inner, n_t] (token stride, floats)
    float * qn; float * kn; int64_t q_st, k_st;               // l2-normalized q / k [128, n_head, n_t] (token strides)
    float q_eps, k_eps;
    float * tail[GDN_CONV_MAX_SLOTS]; int tail_start[GDN_CONV_MAX_SLOTS]; int n_slots;
    int d_inner, n_t;
    unsigned int * bar;                                       // outputs over dead inputs: grid barrier (reads, writes)
};

static __global__ void __launch_bounds__(128) kda_conv_block_dec(const kda_conv_block_args a) {
    const int c    = blockIdx.x*128 + threadIdx.x;   // channel of the q|k|v block
    const int part = (blockIdx.x*128)/a.d_inner;     // 0 q, 1 k, 2 v (uniform per block: d_inner % 128 == 0)
    const int cl   = c - part*a.d_inner;             // channel within its projection
    const int n_t  = a.n_t;
    const float * x  = part == 0 ? a.x0  : part == 1 ? a.x1  : a.x2;
    const float * w  = part == 0 ? a.w0  : part == 1 ? a.w1  : a.w2;
    const int64_t sx = part == 0 ? a.sx0 : part == 1 ? a.sx1 : a.sx2;
    float seq[3 + GDN_CONV_MAX_T];
    const float * state = a.state_ids ? a.state + (int64_t) a.state_ids[0]*a.state_row : a.state;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        seq[j] = state[c*3 + j];
    }
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            seq[3 + t] = x[t*sx + cl];
        }
    }
    const float4 wc = *(const float4 *) (w + 4*cl);
    if (a.bar) {
        gdn_conv_grid_barrier(a.bar);
    }
    float y[GDN_CONV_MAX_T];
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            y[t] = ggml_cuda_op_silu_single(seq[t]*wc.x + seq[t + 1]*wc.y + seq[t + 2]*wc.z + seq[t + 3]*wc.w);
            a.silu[t*a.sd + c] = y[t];
        }
    }
    for (int s = 0; s < a.n_slots; ++s) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            // seq index start+j (start <= n_t, so the index is < 3 + n_t)
            const int k = a.tail_start[s] + j;
            float v = seq[0];
#pragma unroll
            for (int q = 1; q < 3 + GDN_CONV_MAX_T; ++q) {
                v = q == k ? seq[q] : v;
            }
            a.tail[s][c*3 + j] = v;
        }
    }
    if (part == 2) {
        return;
    }
    // q / k: l2-normalize the head (this block) per token, scale = rsqrt(max(sum, eps^2)) as in L2_NORM
    constexpr int NW = 128/WARP_SIZE;
    __shared__ float ps[NW][GDN_CONV_MAX_T];
    const int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            const float ss = warp_reduce_sum(y[t]*y[t]);
            if (lane == 0) {
                ps[wid][t] = ss;
            }
        }
    }
    __syncthreads();
    const float eps = part == 0 ? a.q_eps : a.k_eps;
    float * out = part == 0 ? a.qn : a.kn;
    const int64_t st = part == 0 ? a.q_st : a.k_st;
#pragma unroll
    for (int t = 0; t < GDN_CONV_MAX_T; ++t) {
        if (t < n_t) {
            float tot = 0.0f;
#pragma unroll
            for (int i = 0; i < NW; ++i) {
                tot += ps[i][t];
            }
            out[t*st + cl] = rsqrtf(fmaxf(tot, eps*eps))*y[t];
        }
    }
}

// qk = CONCAT(q, k), qkv = CONCAT(qk, v), concat = CONCAT(conv state, transposed view of qkv); w1 = CONCAT(wq, wk) and
// w2 = CONCAT(w1, wv) (dim 1, through RESHAPEs), conv = SSM_CONV(concat, w2), silu = SILU(conv); q_l2 / k_l2 = L2_NORM
// of the q / k views of silu; tails_cpy: the conv-state tail CPYs (views of concat into the cache rows)
bool ggml_cuda_kda_conv_block_ok(const ggml_tensor * qk, const ggml_tensor * qkv, const ggml_tensor * concat,
                                 const ggml_tensor * conv, const ggml_tensor * silu, const ggml_tensor * const * tails_cpy,
                                 int n_slots, const ggml_tensor * q_l2, const ggml_tensor * k_l2, bool * need_barrier) {
    static const bool disabled = [] { const char * e = getenv("GGML_KDA_CONV_BLOCK"); return e && atoi(e) == 0; }();
    static const bool dbg = getenv("GGML_GDN_CONV_BLOCK_DBG") != nullptr;
#define KCB_FAIL(n) do { if (dbg) { fprintf(stderr, "kda convblock ok fail %d\n", n); } return false; } while (0)
    *need_barrier = false;
    if (disabled || n_slots < 1 || n_slots > GDN_CONV_MAX_SLOTS || !ggml_cuda_ssm_conv_deferred_ok(concat, conv, silu)) {
        KCB_FAIL(1);
    }
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    const ggml_tensor * x[3]  = { qk->src[0], qk->src[1], qkv->src[1] };
    const ggml_tensor * w2    = conv->src[1];
    const ggml_tensor * w1    = w2->src[0];
    if (w2->op != GGML_OP_CONCAT || w1 == nullptr || w1->op != GGML_OP_CONCAT || ggml_get_op_params_i32(w2, 0) != 1 ||
            ggml_get_op_params_i32(w1, 0) != 1) {
        KCB_FAIL(2);
    }
    const ggml_tensor * w[3] = { w1->src[0], w1->src[1], w2->src[1] };
    const int64_t n_c = state->ne[1], n_t = xt->ne[0], d = x[0]->ne[0];
    if (ggml_get_op_params_i32(qk, 0) != 0 || ggml_get_op_params_i32(qkv, 0) != 0 || qkv->src[0] != qk ||
            xt->view_src != qkv || n_c != 3*d || d % 128 != 0 || n_t < 1 || n_t > GDN_CONV_MAX_T ||
            silu->ne[0] != n_c || silu->ne[1] != n_t || silu->ne[2] != 1) {
        KCB_FAIL(3);
    }
    for (int p = 0; p < 3; ++p) {
        if (x[p]->type != GGML_TYPE_F32 || x[p]->ne[0] != d || x[p]->ne[1] != n_t || x[p]->ne[2] != 1 || x[p]->ne[3] != 1 ||
                x[p]->nb[0] != sizeof(float) || w[p]->type != GGML_TYPE_F32 || w[p]->ne[0] != 4 || w[p]->ne[1] != d ||
                !ggml_is_contiguous(w[p]) || ((uintptr_t) w[p]->data) % 16 != 0) {
            KCB_FAIL(4);
        }
    }
    const ggml_tensor * l2[2] = { q_l2, k_l2 };
    for (int p = 0; p < 2; ++p) {
        const ggml_tensor * v = l2[p]->src[0]; // view of silu
        if (l2[p]->op != GGML_OP_L2_NORM || l2[p]->type != GGML_TYPE_F32 || !ggml_is_contiguous(l2[p]) ||
                v->op != GGML_OP_VIEW || v->src[0] != silu || v->ne[0] != 128 || v->ne[1]*128 != d || v->ne[2] != n_t ||
                v->ne[3] != 1 || v->nb[0] != sizeof(float) || v->nb[1] != 128*sizeof(float) || v->nb[2] != silu->nb[1] ||
                v->view_offs != (size_t) p*d*sizeof(float)) {
            KCB_FAIL(5);
        }
    }
    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    // outputs over (dead) inputs need the grid barrier; outputs may not overlap each other or the weights
    for (const ggml_tensor * o : { silu, q_l2, k_l2 }) {
        if (overlaps(o, state) || overlaps(o, x[0]) || overlaps(o, x[1]) || overlaps(o, x[2])) {
            *need_barrier = true;
        }
        for (int p = 0; p < 3; ++p) {
            if (overlaps(o, w[p])) {
                KCB_FAIL(6);
            }
        }
    }
    if (overlaps(silu, q_l2) || overlaps(silu, k_l2) || overlaps(q_l2, k_l2)) {
        KCB_FAIL(7);
    }
    for (int s = 0; s < n_slots; ++s) {
        const ggml_tensor * dt = tails_cpy[s]->src[1];
        if (dt->type != GGML_TYPE_F32 || !ggml_is_contiguous(dt) || ggml_nelements(dt) != 3*n_c) {
            KCB_FAIL(8);
        }
        for (const ggml_tensor * o : { state, silu, q_l2, k_l2, x[0], x[1], x[2], w[0], w[1], w[2] }) {
            if (overlaps(dt, o)) {
                KCB_FAIL(9);
            }
        }
    }
    return true;
#undef KCB_FAIL
}

void ggml_cuda_kda_conv_block(ggml_backend_cuda_context & ctx, const ggml_tensor * qk, const ggml_tensor * qkv,
                              const ggml_tensor * concat, const ggml_tensor * conv, ggml_tensor * silu,
                              const ggml_tensor * const * tails_view, const ggml_tensor * const * tails_cpy, int n_slots,
                              ggml_tensor * q_l2, ggml_tensor * k_l2, bool need_barrier, const ggml_tensor * state_rows) {
    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * w2    = conv->src[1];
    const ggml_tensor * w1    = w2->src[0];
    kda_conv_block_args a = {};
    a.x0 = (const float *) qk->src[0]->data;  a.sx0 = qk->src[0]->nb[1]/sizeof(float);
    a.x1 = (const float *) qk->src[1]->data;  a.sx1 = qk->src[1]->nb[1]/sizeof(float);
    a.x2 = (const float *) qkv->src[1]->data; a.sx2 = qkv->src[1]->nb[1]/sizeof(float);
    a.w0 = (const float *) w1->src[0]->data;
    a.w1 = (const float *) w1->src[1]->data;
    a.w2 = (const float *) w2->src[1]->data;
    a.state = (const float *) state->data;
    if (state_rows) {
        // the deferred GET_ROWS (cache rows, ids): each thread reads its channel before writing its tails, so a tail slot
        // that is the source row itself is safe
        a.state     = (const float *) state_rows->src[0]->data;
        a.state_ids = (const int32_t *) state_rows->src[1]->data;
        a.state_row = state_rows->src[0]->nb[1]/sizeof(float);
    }
    a.silu = (float *) silu->data;
    a.sd   = silu->nb[1]/sizeof(float);
    a.qn   = (float *) q_l2->data;  a.q_st = q_l2->nb[2]/sizeof(float);
    a.kn   = (float *) k_l2->data;  a.k_st = k_l2->nb[2]/sizeof(float);
    a.q_eps = ggml_get_op_params_f32(q_l2, 0);
    a.k_eps = ggml_get_op_params_f32(k_l2, 0);
    a.n_slots = n_slots;
    for (int s = 0; s < n_slots; ++s) {
        a.tail[s]       = (float *) tails_cpy[s]->src[1]->data;
        a.tail_start[s] = (int) (tails_view[s]->view_offs / sizeof(float)); // column offset of the tail in the concat
    }
    a.d_inner = (int) qk->src[0]->ne[0];
    a.n_t     = (int) concat->src[1]->ne[0];
    if (need_barrier) {
        static unsigned int * bar_base[GGML_CUDA_MAX_DEVICES] = {};
        const int phys = ggml_cuda_info().devices[ctx.device].physical_device;
        if (bar_base[phys] == nullptr) {
            CUDA_CHECK(cudaGetSymbolAddress((void **) &bar_base[phys], (const void *) &g_gdn_conv_bar));
        }
        a.bar = bar_base[phys] + 2*(ctx.device*GGML_CUDA_MAX_STREAMS + ctx.curr_stream_no);
    }
    kda_conv_block_dec<<<3*a.d_inner/128, 128, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
