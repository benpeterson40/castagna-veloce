#pragma once

#include "common.cuh"

// GGML_OP_DSV4_COMP_POOL: DeepSeek V4(.1) KV compressor pooling from the state (gather, softmax-gated pooling, RMS norm,
// optional NORM rope) in one kernel. See dsv4-comp-pool.cu.
bool ggml_cuda_dsv4_comp_pool_supported(const ggml_tensor * dst);
void ggml_cuda_dsv4_comp_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
