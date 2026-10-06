#pragma once

#include "common.cuh"

// GCN dense q8_0 matvec for a few tokens (MTP verify): activations staged in LDS once per block, long runs of rows
bool ggml_cuda_gcn_q8_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
