#pragma once

#include "common.cuh"

// Small F32/BF16-weight projections at prefill batch sizes (MoE router, HC inject, GDN alpha/beta, indexer):
// dst[n][m] = sum_k w[m][k] * x[n][k] with an R x T register tile per wave and K split over the block.
bool ggml_cuda_mm_tile_f_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mm_tile_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
