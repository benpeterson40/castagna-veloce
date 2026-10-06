#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.y;
    const int64_t col   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

// float4 variant with the expert loop unrolled at compile time (N = n_expert_used), so each thread
// issues all N row loads before summing; the scalar kernel above reaches ~half of memory bandwidth.
// experts stored as F16 in place by the F16 MoE down projection (same layout); same association as the F32 kernels
template <int N>
static __global__ void moe_weighted_reduction_f16in(const half * __restrict__ experts,
                                                    const float * __restrict__ expert_scale,
                                                    const float * __restrict__ weights,
                                                    float * __restrict__ dst,
                                                    const int64_t n_embd) {
    const int64_t token = blockIdx.y;
    const int64_t c4    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * c4 >= n_embd) {
        return;
    }
    const uint64_t first_row = (uint64_t) token * N;
    float4 v[N];
#pragma unroll
    for (int e = 0; e < N; ++e) {
        const half2 * h = (const half2 *) (experts + (first_row + e) * n_embd) + 2*c4;
        const float2 lo = __half22float2(h[0]), hi = __half22float2(h[1]);
        v[e] = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int e = 0; e < N; ++e) {
        const float sc = expert_scale != nullptr ? expert_scale[first_row + e] : 1.0f;
        const float wt = weights[first_row + e];
        sum.x += (v[e].x * sc) * wt;
        sum.y += (v[e].y * sc) * wt;
        sum.z += (v[e].z * sc) * wt;
        sum.w += (v[e].w * sc) * wt;
    }
    ((float4 *) (dst + token * n_embd))[c4] = sum;
}

template <int N>
static __global__ void moe_weighted_reduction_f32_vec4(const float * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       float * __restrict__ dst,
                                                       const int64_t n_embd) {
    const int64_t token = blockIdx.y;
    const int64_t c4    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (4 * c4 >= n_embd) {
        return;
    }
    const uint64_t first_row = (uint64_t) token * N;

    float4 v[N];
#pragma unroll
    for (int e = 0; e < N; ++e) {
        v[e] = ((const float4 *) (experts + (first_row + e) * n_embd))[c4];
    }
    // same association as the scalar kernel: ((x * scale) * weight) summed in expert order
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int e = 0; e < N; ++e) {
        const float sc = expert_scale != nullptr ? expert_scale[first_row + e] : 1.0f;
        const float wt = weights[first_row + e];
        sum.x += (v[e].x * sc) * wt;
        sum.y += (v[e].y * sc) * wt;
        sum.z += (v[e].z * sc) * wt;
        sum.w += (v[e].w * sc) * wt;
    }
    ((float4 *) (dst + token * n_embd))[c4] = sum;
}

// grids are (column chunks, tokens): with the token in x, the workgroups dispatched together read the same columns of
// rows n_expert_used*n_embd*4 bytes apart and crowded a few memory channels (MI50, 2048 tokens, n_embd 4096: 6 experts
// 771 -> 311 us, 8: 992 -> 397, 10: 1206 -> 507; ~305 -> ~750 GB/s)
static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    static const bool no_vec = [] { const char * e = getenv("GGML_MOE_RED_NO_VEC4"); return e && atoi(e) != 0; }();
    if (!no_vec && n_embd % 4 == 0 && ((uintptr_t) experts) % 16 == 0 && ((uintptr_t) dst) % 16 == 0 &&
            (n_expert_used == 6 || n_expert_used == 8 || n_expert_used == 10)) {
        const dim3 blocks4((n_embd / 4 + threads - 1) / threads, n_tokens, 1);
        if (n_expert_used == 10) {
            moe_weighted_reduction_f32_vec4<10><<<blocks4, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd);
        } else if (n_expert_used == 6) {
            // DeepSeek V4 (6 experts): the scalar kernel took 2.9 ms per layer for a 2048-token ubatch on MI50
            moe_weighted_reduction_f32_vec4<6><<<blocks4, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd);
        } else {
            moe_weighted_reduction_f32_vec4<8><<<blocks4, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd);
        }
        return;
    }
    const dim3 blocks((n_embd + threads - 1) / threads, n_tokens, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    if (ggml_cuda_is_f16_inplace(ctx, experts)) {
        GGML_ASSERT(n_embd % 4 == 0 && (n_expert_used == 8 || n_expert_used == 10) &&
                    ggml_cuda_f16_inplace_ld(ctx, experts, n_embd) == n_embd);
        constexpr int threads = 256;
        const dim3 blocks4((n_embd / 4 + threads - 1) / threads, n_tokens, 1);
        const half  * e16 = (const half *) experts->data;
        const float * es  = expert_scale ? (const float *) expert_scale->data : nullptr;
        if (n_expert_used == 10) {
            moe_weighted_reduction_f16in<10><<<blocks4, threads, 0, stream>>>(e16, es, (const float *) weights->data, (float *) dst->data, n_embd);
        } else {
            moe_weighted_reduction_f16in<8><<<blocks4, threads, 0, stream>>>(e16, es, (const float *) weights->data, (float *) dst->data, n_embd);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
