#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

// single-token router matvec (bf16, 384 experts) fused with the sqrt(softplus) + bias top-k (GCN)
bool ggml_cuda_router1_supported(const ggml_tensor * mm, int n_used, bool sigmoid = false);
void ggml_cuda_router1_topk(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * weights, ggml_tensor * ids,
                            const ggml_tensor * bias, float clamp_val, float scale_val, bool sigmoid, int n_tok = 1,
                            int64_t s_ids_tok = 0, int64_t s_w_tok = 0);
