#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// decode attention q/k chain (norm + rope for q and k, k/v cache rows) in one kernel; returns the extra nodes consumed
// (the k / v projections between the q rope and the k norm go through compute_middle first)
int ggml_cuda_attn_qk_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
                            bool (*compute_middle)(ggml_backend_cuda_context & ctx, ggml_tensor * t));
