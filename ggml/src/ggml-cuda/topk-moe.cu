#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 384: // DeepSeek V4.1 Flash (sqrt(softplus) gating with bias; the 10-op chain ran unfused)
            ggml_cuda_kernel_launch(topk_moe_cuda<384, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

// GCN variant for softmax gating without bias (qwen3next / qwen4exp routers, 512 experts): one 64-lane wave per row,
// 8 experts per lane (e = lane + 64 j). The warp kernel above is latency bound on gfx906 (~10 us of execution for 1 row:
// n_expert_used rounds of a 32-lane argmax, each 5 dependent shuffle pairs through LDS, on half a wave). Here each lane
// sorts its 8 probabilities in registers, then n_expert_used rounds take the wave maximum of the list heads with DPP
// moves (no LDS) and pop the winner; ties go to the lowest expert index as in the argmax rounds. The softmax sum and the
// weight sum are accumulated exactly like the warp kernel (32-lane partials over experts L + 32 i in order i, then the
// same butterfly), so ids and weights are bit-identical.
#ifdef GGML_USE_HIP
template <int ctrl, int row_mask = 0xF>
static __device__ __forceinline__ uint32_t topk_dpp(const uint32_t v) {
    return (uint32_t) __builtin_amdgcn_update_dpp((int) v, (int) v, ctrl, row_mask, 0xF, false);
}

// wave-wide unsigned max / min (result in every lane)
template <bool MAX>
static __device__ __forceinline__ uint32_t topk_wave_reduce(uint32_t v) {
    const auto op = [](uint32_t a, uint32_t b) { return MAX ? max(a, b) : min(a, b); };
    v = op(v, topk_dpp<0xB1>(v));        // quad_perm [1,0,3,2]
    v = op(v, topk_dpp<0x4E>(v));        // quad_perm [2,3,0,1]
    v = op(v, topk_dpp<0x141>(v));       // row_half_mirror
    v = op(v, topk_dpp<0x140>(v));       // row_mirror: every lane has its row's result
    v = op(v, topk_dpp<0x142, 0xA>(v));  // row_bcast15
    v = op(v, topk_dpp<0x143, 0xC>(v));  // row_bcast31: lane 63 has the wave's result
    return (uint32_t) __builtin_amdgcn_readlane((int) v, 63);
}

template <int NE>
__launch_bounds__(64)
static __global__ void topk_moe_sorted_gcn(const float * __restrict__ logits, float * __restrict__ weights, int32_t * __restrict__ ids,
                                           const int n_expert_used, const float clamp_val, const float scale_val, const bool with_norm) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__) // selected on GCN only: other targets compile an empty kernel
    constexpr int NL = 32;       // lanes of the reference warp kernel
    constexpr int J  = NE/64;    // values per lane
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    __shared__ float sp[NE];
    __shared__ float sel_p[64];
    __shared__ int   sel_e[64];
    __shared__ float red;

    float v[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        v[j] = logits[(int64_t) row*NE + lane + 64*j];
    }
    float m = v[0];
#pragma unroll
    for (int j = 1; j < J; ++j) {
        m = fmaxf(m, v[j]);
    }
    m = warp_reduce_max<64>(m);
#pragma unroll
    for (int j = 0; j < J; ++j) {
        v[j] = expf(v[j] - m);
        sp[lane + 64*j] = v[j];
    }
    __syncthreads();
    // reference: lane L < 32 sums experts L + 32 i for i = 0 .. NE/32 - 1 in order, then a 32-lane butterfly
    if (lane < NL) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < NE/NL; ++i) {
            s += sp[lane + NL*i];
        }
        s = warp_reduce_sum<NL>(s);
        if (lane == 0) {
            red = s;
        }
    }
    __syncthreads();
    const float inv_sum = 1.0f / red;
    uint32_t key[J];
    int      ex[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        float p = v[j] * inv_sum;
        if (__isnanf(p)) {
            p = -FLT_MAX;
        }
        const uint32_t u = __float_as_uint(p);
        key[j] = u & 0x80000000u ? ~u : u | 0x80000000u; // larger float -> larger key, never 0
        ex[j]  = lane + 64*j;
    }
    // odd-even transposition sort, descending key (equal keys keep the lower expert first: e grows with j)
#pragma unroll
    for (int pass = 0; pass < J; ++pass) {
#pragma unroll
        for (int j = pass & 1; j + 1 < J; j += 2) {
            if (key[j + 1] > key[j]) {
                const uint32_t tk = key[j]; key[j] = key[j + 1]; key[j + 1] = tk;
                const int      te = ex[j];  ex[j]  = ex[j + 1];  ex[j + 1]  = te;
            }
        }
    }
    for (int r = 0; r < n_expert_used; ++r) {
        const uint32_t mk = topk_wave_reduce<true>(key[0]);
        const uint64_t tied = __ballot(key[0] == mk);
        int win;
        if (__popcll(tied) == 1) {
            win = __ffsll((unsigned long long) tied) - 1;
        } else {
            const uint32_t me = topk_wave_reduce<false>(key[0] == mk ? (uint32_t) ex[0] : 0xFFFFFFFFu);
            win = __ffsll((unsigned long long) __ballot(key[0] == mk && (uint32_t) ex[0] == me)) - 1;
        }
        if (lane == win) {
            const uint32_t k = key[0];
            sel_p[r] = __uint_as_float(k & 0x80000000u ? k & 0x7FFFFFFFu : ~k);
            sel_e[r] = ex[0];
#pragma unroll
            for (int j = 0; j + 1 < J; ++j) {
                key[j] = key[j + 1];
                ex[j]  = ex[j + 1];
            }
            key[J - 1] = 0;
            ex[J - 1]  = 0x7FFFFFFF;
        }
    }
    __syncthreads();
    if (lane < n_expert_used) {
        ids[(int64_t) row*NE + lane] = sel_e[lane];
    }
    if (lane < NL) {
        float inv = 1.0f;
        if (with_norm) {
            float s = 0.0f;
            for (int q = 0; q < n_expert_used; ++q) {
                if (sel_e[q] % NL == lane) {
                    s += sel_p[q];
                }
            }
            s = warp_reduce_sum<NL>(s);
            s = fmaxf(s, clamp_val);
            inv = 1.0f / s;
        }
        for (int q = lane; q < n_expert_used; q += NL) {
            weights[(int64_t) row*n_expert_used + q] = (with_norm ? sel_p[q]*inv : sel_p[q]) * scale_val;
        }
    }
#endif // gfx9 device code
}
#endif // GGML_USE_HIP

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

#ifdef GGML_USE_HIP
    static const bool rank_env = [] { const char * e = getenv("GGML_CUDA_TOPK_MOE_RANK"); return !e || atoi(e) != 0; }();
    if (rank_env && !bias && !config.use_sigmoid && !config.use_sqrt_softplus && !config.delayed_softmax &&
            n_expert_used <= 64 && GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ctx.device].cc) &&
            (n_experts == 128 || n_experts == 256 || n_experts == 512)) {
        cudaStream_t stream = ctx.stream();
        switch (n_experts) {
            case 128: topk_moe_sorted_gcn<128><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
            case 256: topk_moe_sorted_gcn<256><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
            default:  topk_moe_sorted_gcn<512><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif // GGML_USE_HIP

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 384 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

// ---- single-token router matvec + sqrt(softplus) top-k (DeepSeek V4.1 decode) ----
// logits = W (bf16 [K, E]) . x, then as topk_moe_cuda with sqrt_softplus, selection bias, weight normalization and scale.
// Workgroups of 256 threads take 8 experts (2 per wave, 16-byte bf16 lane loads). Every load of a lane (its token chunks
// and both rows' words, interleaved per chunk) is issued at once, so the dot products run while the rest arrives (the
// lane's chunk order and the reduction tree are unchanged, so the logits are too). Each workgroup gates its own experts
// (sqrt(softplus), the selection bias) and stores the probability and an order-preserving 32-bit key of the biased value;
// the last workgroup to finish (arrival counter) runs only the top-k on one wave (E/64 experts per lane): per pick the
// wave maximum of the keys (DPP within rows, readlane across them) and the owner lane from a ballot (lowest expert index
// on equal values, as before).
static constexpr int ROUTER1_EPW = 8; // reference only: the launch uses 4 experts per workgroup (GGML_CUDA_ROUTER1_EPW=8: two per wave, half the workgroups; tg 93.8 vs 93.1)

template <int ctrl> static __device__ __forceinline__ uint32_t router1_dpp(const uint32_t v) {
    return (uint32_t) __builtin_amdgcn_update_dpp(0, (int) v, ctrl, 0xF, 0xF, true);
}

// F32W: f32 weights (4 values per 16-byte chunk, GLM-5-Next's router) instead of bf16 (8); SIGMOID: sigmoid gating instead of
// sqrt(softplus); E need not be a multiple of 64 (288: the top-k lanes past E hold taken keys)
// F16W: f16 weights (8 values per chunk, DeepSeek V4 Flash: 256 experts, sqrt(softplus))
// NT tokens (1..4, XLDS only): the weight words stay in registers, each token is staged in LDS in turn and gated into its
// own scratch rows; the last workgroup runs one top-k wave per token (GLM-5.3's 2-token MTP verify ran the f32 router
// matvec and the top-k as two launches)
template <int E, int NI, bool XLDS, int WEPW, bool F32W = false, bool SIGMOID = false, bool F16W = false, int NT = 1> // NI: 16-byte chunks per lane and row
static __global__ void __launch_bounds__(256) router1_topk_bf16(
        const char * __restrict__ W, const float * __restrict__ x, const float * __restrict__ bias,
        float * __restrict__ weights, int32_t * __restrict__ ids, float * __restrict__ scratch, int * __restrict__ counter,
        const int K, const int64_t s_row, const int n_used, const float clamp_val, const float scale_val,
        const int64_t sx_tok, const int64_t s_ids_tok, const int64_t s_w_tok) {
    static_assert(NT >= 1 && NT <= 4 && (NT == 1 || XLDS), "several tokens: XLDS, one top-k wave per token");
    constexpr int PW = 64;
    const int tid  = threadIdx.x;
    const int lane = tid % PW;
    const int wv   = tid / PW;
    __shared__ int s_last;
    // token t: its gated values at scratch + 2*E*t ([E]: sqrt(softplus(logit)) or sigmoid), the sort keys after them

    // two experts per wave (4 waves, 8 experts per workgroup)
    constexpr int NWAVE = 256/PW;
    constexpr int EPW   = WEPW/NWAVE;
    constexpr int VPC   = F32W ? 4 : 8;   // values per 16-byte chunk
    constexpr int XPC   = VPC/4;          // float4 of the token per chunk
    const int e0 = blockIdx.x*WEPW + wv*EPW;
    const int nc = K/VPC;

    // XLDS: the token once per workgroup into LDS instead of every wave holding all of it in registers (4x the L2 reads,
    // ~80 more VGPRs); its loads go out first, so they come back before the weight words
    __shared__ float4 xs4[XLDS ? XPC*NI*PW : 1];
    float4 xa[XLDS ? 1 : NI], xb[XLDS ? 1 : NI];
    constexpr int NX4 = (XPC*NI*PW + 255)/256;  // float4 of the token per thread (XLDS)
    float4 xl[XLDS ? NX4 : 1];
    if (XLDS) {
#pragma unroll
        for (int j = 0; j < NX4; ++j) {
            const int k = tid + 256*j;
            xl[j] = k < XPC*nc ? ((const float4 *) x)[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    uint4  w4[EPW][NI];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int  c  = lane + i*PW;
        const bool ok = c < nc;
        if (!XLDS) {
            xa[i] = ok ? ((const float4 *) x)[XPC*c] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (!F32W) {
                xb[i] = ok ? ((const float4 *) x)[2*c + 1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
#pragma unroll
        for (int r = 0; r < EPW; ++r) {
            w4[r][i] = ok ? *(const uint4 *) (W + (int64_t) (e0 + r)*s_row + 16*c) : make_uint4(0, 0, 0, 0);
        }
    }
    if (XLDS) {
#pragma unroll
        for (int j = 0; j < NX4; ++j) {
            const int k = tid + 256*j;
            if (k < XPC*NI*PW) {
                xs4[k] = xl[j];
            }
        }
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n s_barrier" ::: "memory"); // LDS only: the weight words stay in flight
    }
    float bl = 0.0f;                         // the bias of this lane's expert (lanes 0..EPW-1)
    if (lane < EPW) {
        bl = bias[e0 + lane];
    }
    for (int t = 0; t < NT; ++t) {
        if (t > 0) {
            // the next token into LDS once every wave is done with the previous one
            __syncthreads();
#pragma unroll
            for (int j = 0; j < NX4; ++j) {
                const int k = tid + 256*j;
                if (k < XPC*NI*PW) {
                    xs4[k] = k < XPC*nc ? ((const float4 *) (x + t*sx_tok))[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            __syncthreads();
        }
        float    * probs = scratch + 2*E*t;
        uint32_t * keys  = (uint32_t *) (probs + E);
        float acc[EPW] = {};
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            if (lane + i*PW < nc) {
                const int    c   = lane + i*PW;
                const float4 xai = XLDS ? xs4[XPC*c] : xa[i];
                if constexpr (F32W) {
#pragma unroll
                    for (int r = 0; r < EPW; ++r) {
                        const uint4 w = w4[r][i];
                        float a = acc[r];
                        a = fmaf(__uint_as_float(w.x), xai.x, a); a = fmaf(__uint_as_float(w.y), xai.y, a);
                        a = fmaf(__uint_as_float(w.z), xai.z, a); a = fmaf(__uint_as_float(w.w), xai.w, a);
                        acc[r] = a;
                    }
                } else if constexpr (F16W) {
                    // the token rounded to f16 as the separate f16 matvec (gcn_f16_mv / MMVF) rounds it: the same products, so
                    // the same experts on near ties
                    const float4 xbr = XLDS ? xs4[2*c + 1] : xb[i];
                    const float4 xa16 = make_float4(__half2float(__float2half(xai.x)), __half2float(__float2half(xai.y)),
                                                    __half2float(__float2half(xai.z)), __half2float(__float2half(xai.w)));
                    const float4 xbi  = make_float4(__half2float(__float2half(xbr.x)), __half2float(__float2half(xbr.y)),
                                                    __half2float(__float2half(xbr.z)), __half2float(__float2half(xbr.w)));
#pragma unroll
                    for (int r = 0; r < EPW; ++r) {
                        const uint4 w = w4[r][i];
                        const float2 w0 = __half22float2(*(const half2 *) &w.x), w1 = __half22float2(*(const half2 *) &w.y);
                        const float2 w2 = __half22float2(*(const half2 *) &w.z), w3 = __half22float2(*(const half2 *) &w.w);
                        float a = acc[r];
                        a = fmaf(w0.x, xa16.x, a); a = fmaf(w0.y, xa16.y, a); a = fmaf(w1.x, xa16.z, a); a = fmaf(w1.y, xa16.w, a);
                        a = fmaf(w2.x, xbi.x, a); a = fmaf(w2.y, xbi.y, a); a = fmaf(w3.x, xbi.z, a); a = fmaf(w3.y, xbi.w, a);
                        acc[r] = a;
                    }
                } else {
                const float4 xbi = XLDS ? xs4[2*c + 1] : xb[i];
#pragma unroll
                for (int r = 0; r < EPW; ++r) {
                    const uint4 w = w4[r][i];
                    float a = acc[r];
                    a = fmaf(__uint_as_float(w.x << 16), xai.x, a); a = fmaf(__uint_as_float(w.x & 0xFFFF0000u), xai.y, a);
                    a = fmaf(__uint_as_float(w.y << 16), xai.z, a); a = fmaf(__uint_as_float(w.y & 0xFFFF0000u), xai.w, a);
                    a = fmaf(__uint_as_float(w.z << 16), xbi.x, a); a = fmaf(__uint_as_float(w.z & 0xFFFF0000u), xbi.y, a);
                    a = fmaf(__uint_as_float(w.w << 16), xbi.z, a); a = fmaf(__uint_as_float(w.w & 0xFFFF0000u), xbi.w, a);
                    acc[r] = a;
                }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < EPW; ++r) {
            acc[r] = warp_reduce_sum<PW>(acc[r]);
        }
        // gating of the wave's experts: lane r < EPW takes expert e0 + r
        if (lane < EPW) {
            float l = acc[0];
#pragma unroll
            for (int r = 1; r < EPW; ++r) {
                l = lane == r ? acc[r] : l;
            }
            const float    pv = SIGMOID ? 1.0f/(1.0f + expf(-l)) : sqrtf(l > 20.0f ? l : logf(1.0f + expf(l)));
            const uint32_t u  = __float_as_uint((pv + bl) + 0.0f); // -0 -> +0: equal values, equal keys
            probs[e0 + lane] = pv;
            keys[e0 + lane]  = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(counter, 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last || wv >= NT) {
        return;
    }
    __threadfence();
    const int t = wv; // one wave per token
    const float    * probs = scratch + 2*E*t;
    const uint32_t * keys  = (const uint32_t *) (probs + E);

    // one wave: top-k by (biased value, lower index), normalization
    constexpr int NPL = (E + PW - 1)/PW;
    float    p[NPL];
    uint32_t key[NPL];                       // 0: taken (and the lanes past E)
#pragma unroll
    for (int k = 0; k < NPL; ++k) {
        const int  e  = lane + k*PW;
        const bool ok = E % PW == 0 || e < E;
        p[k]   = ok ? probs[min(e, E - 1)] : 0.0f;
        key[k] = ok ? keys[min(e, E - 1)]  : 0u;
    }
    float wsum = 0.0f;
    float wsel = 0.0f; // this lane's weight if it wrote ids[k] for its k == lane
    for (int k = 0; k < n_used; ++k) {
        // the lane's best (lowest index on equal keys), then the wave maximum
        uint32_t bk  = key[0];
        int      bkk = 0;
#pragma unroll
        for (int kk = 1; kk < NPL; ++kk) {
            if (key[kk] > bk) {
                bk = key[kk]; bkk = kk;
            }
        }
        uint32_t m = bk;
        m = max(m, router1_dpp<0xB1>(m));
        m = max(m, router1_dpp<0x4E>(m));
        m = max(m, router1_dpp<0x141>(m));   // row_half_mirror
        m = max(m, router1_dpp<0x140>(m));   // row_mirror
        m = max(max((uint32_t) __builtin_amdgcn_readlane((int) m, 0),  (uint32_t) __builtin_amdgcn_readlane((int) m, 16)),
                max((uint32_t) __builtin_amdgcn_readlane((int) m, 32), (uint32_t) __builtin_amdgcn_readlane((int) m, 48)));
        const uint64_t hit = __ballot(bk == m);
        int we;
        if (__popcll(hit) == 1) {
            const int wl = __ffsll((unsigned long long) hit) - 1;
            we = wl + PW*__builtin_amdgcn_readlane(bkk, wl);
        } else {                             // equal values in several lanes: the lowest expert index
            int cand = bk == m ? lane + PW*bkk : 0x7FFFFFFF;
#pragma unroll
            for (int off = PW/2; off > 0; off >>= 1) {
                cand = min(cand, __shfl_xor(cand, off, PW));
            }
            we = cand;
        }
        const int wl = we % PW;
        float pb = p[0];
#pragma unroll
        for (int kk = 1; kk < NPL; ++kk) {
            pb = kk == bkk ? p[kk] : pb;
        }
        const float best_p = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(pb), wl));
        if (lane == wl) {
#pragma unroll
            for (int kk = 0; kk < NPL; ++kk) {
                key[kk] = kk == bkk ? 0u : key[kk];
            }
        }
        if (lane == k) {
            ids[t*s_ids_tok + k] = we;
            wsel   = best_p;
        }
        wsum += best_p;
    }
    wsum = fmaxf(wsum, clamp_val);
    const float inv_sum = 1.0f/wsum;
    if (lane < n_used) {
        weights[t*s_w_tok + lane] = wsel*inv_sum*scale_val;
    }
    if (lane == 0 && t == 0) {
        *counter = 0;
    }
}

// bf16 [K <= 8192, 384] with sqrt(softplus) gating (DeepSeek V4.1) or f32 [K <= 4096, 288] with sigmoid gating (GLM-5-Next)
bool ggml_cuda_router1_supported(const ggml_tensor * mm, int n_used, bool sigmoid) {
    const ggml_tensor * W = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    // GGML_CUDA_ROUTER1_F16=1 (opt-in): DeepSeek V4 Flash's f16 router (256 experts) in this fused kernel at one token
    // (plain decode tg 47.7 -> 49.1, ub1 PPL 14.48 vs 14.43). Off by default: with DSpark the target's one-token steps then
    // route differently from its 6-token verify batches and greedy DSpark runs drafted worse on all 6 prompts tried (prose
    // 62.6 -> 44.4 t/s, story 45.5 -> 40.3); run-ds4-flash.sh sets it for NO_SPEC=1 only
    static const bool f16_env = [] { const char * e = getenv("GGML_CUDA_ROUTER1_F16"); return e && atoi(e) != 0; }();
    const bool combo = (!sigmoid && W->type == GGML_TYPE_BF16 && W->ne[1] == 384 && W->ne[0] % 8 == 0 && W->ne[0] <= 8192) ||
                       (!sigmoid && f16_env && W->type == GGML_TYPE_F16 && W->ne[1] == 256 && W->ne[0] % 8 == 0 && W->ne[0] <= 8192) ||
                       ( sigmoid && W->type == GGML_TYPE_F32  && W->ne[1] == 288 && W->ne[0] % 4 == 0 && W->ne[0] <= 4096);
    // several tokens (one top-k wave each, up to 4): GLM-5-Next's sigmoid router only (GGML_CUDA_ROUTER1_NT=0: one token)
    static const bool nt_env = [] { const char * e = getenv("GGML_CUDA_ROUTER1_NT"); return !e || atoi(e) != 0; }();
    const int64_t n_tok = x->ne[1];
    const bool tok_ok = n_tok == 1 || (nt_env && sigmoid && n_tok <= 4 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[1] % 16 == 0);
    return combo && x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
        W->ne[2] == 1 && W->ne[3] == 1 && W->nb[1] % 16 == 0 &&
        ((uintptr_t) W->data) % 16 == 0 && tok_ok && x->nb[0] == sizeof(float) && ((uintptr_t) x->data) % 16 == 0 &&
        n_used >= 1 && n_used <= 64 && GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

void ggml_cuda_router1_topk(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * weights, ggml_tensor * ids,
                            const ggml_tensor * bias, float clamp_val, float scale_val, bool sigmoid, int n_tok,
                            int64_t s_ids_tok, int64_t s_w_tok) {
    const ggml_tensor * W = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int E = (int) W->ne[1];
    const int K = (int) W->ne[0];
    const int n_used = (int) weights->ne[1]*(int) weights->ne[0];
    cudaStream_t stream = ctx.stream();
    if (ctx.router1_counter == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &ctx.router1_counter, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(ctx.router1_counter, 0, sizeof(int), stream));
    }
    GGML_ASSERT(n_tok >= 1 && n_tok <= 4);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), 2*E*n_tok);
    const int64_t sx_tok = x->nb[1]/sizeof(float);
    const bool f16w = W->type == GGML_TYPE_F16;
    GGML_ASSERT(sigmoid ? E == 288 : f16w ? E == 256 : E == 384);
    static const bool xlds_env = [] { const char * e = getenv("GGML_CUDA_ROUTER1_XLDS"); return !e || atoi(e) != 0; }();
    const bool xlds = xlds_env || n_tok > 1;
    static const int  wepw = [] { const char * e = getenv("GGML_CUDA_ROUTER1_EPW"); return e ? atoi(e) : 4; }();
#define ROUTER1_LAUNCH(NI) do { \
        if (wepw == 4) { if (xlds) { ROUTER1_LAUNCH_X(NI, true, 4); } else { ROUTER1_LAUNCH_X(NI, false, 4); } } \
        else           { if (xlds) { ROUTER1_LAUNCH_X(NI, true, 8); } else { ROUTER1_LAUNCH_X(NI, false, 8); } } } while (0)
#define ROUTER1_LAUNCH_X(NI, XL, EW) do { if (sigmoid) { ROUTER1_LAUNCH_T(288, NI, XL, EW, true, true, false); } \
                                         else if (f16w) { ROUTER1_LAUNCH_T(256, NI, XL, EW, false, false, true); } \
                                         else         { ROUTER1_LAUNCH_T(384, NI, XL, EW, false, false, false); } } while (0)
#define ROUTER1_LAUNCH_T(EE, NI, XL, EW, FW, SG, HW) do { if (XL) { switch (n_tok) { \
        case 2: ROUTER1_LAUNCH_N(EE, NI, true, EW, FW, SG, HW, 2); break; case 3: ROUTER1_LAUNCH_N(EE, NI, true, EW, FW, SG, HW, 3); break; \
        case 4: ROUTER1_LAUNCH_N(EE, NI, true, EW, FW, SG, HW, 4); break; default: ROUTER1_LAUNCH_N(EE, NI, true, EW, FW, SG, HW, 1); break; } } \
        else { ROUTER1_LAUNCH_N(EE, NI, false, EW, FW, SG, HW, 1); } } while (0)
#define ROUTER1_LAUNCH_N(EE, NI, XL, EW, FW, SG, HW, NT_) router1_topk_bf16<EE, NI, XL, EW, FW, SG, HW, NT_><<<(EE)/(EW), 256, 0, stream>>>( \
        (const char *) W->data, (const float *) x->data, (const float *) bias->data, (float *) weights->data, \
        (int32_t *) ids->data, scratch.get(), ctx.router1_counter, K, W->nb[1], n_used, clamp_val, scale_val, sx_tok, s_ids_tok, s_w_tok)
    // chunks per row: K/8 (bf16) or K/4 (f32), 64 lanes
    const int kc = sigmoid ? 2*K : K; // bf16-equivalent K for the NI choice
    if (kc <= 1024) {
        ROUTER1_LAUNCH(2);
    } else if (kc <= 2048) {
        ROUTER1_LAUNCH(4);
    } else if (kc <= 4096) {
        ROUTER1_LAUNCH(8);
    } else if (kc <= 5120) {
        ROUTER1_LAUNCH(10);
    } else {
        ROUTER1_LAUNCH(16);
    }
#undef ROUTER1_LAUNCH
#undef ROUTER1_LAUNCH_T
#undef ROUTER1_LAUNCH_N
#undef ROUTER1_LAUNCH_X
    CUDA_CHECK(cudaGetLastError());
}
