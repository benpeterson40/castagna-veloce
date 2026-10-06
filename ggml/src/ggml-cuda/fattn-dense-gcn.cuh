#pragma once

#include "common.cuh"

// Dense prefill flash attention for gfx906 (see fattn-dense-gcn.cu): D = 256 (K and V), f16 K/V, f32 Q/dst, GQA ratio in
// {1, 2, 3, 4, 6, 8, 12}, no sinks/ALiBi/softcap, f16 mask, at least GGML_CUDA_FA_DENSE_GCN_MIN_T (64) query tokens.
// Returns true when it computed dst, false when it does not apply (nothing was launched).
// GGML_CUDA_FA_DENSE_GCN=0 disables it (default 1); GGML_CUDA_FA_DENSE_GCN_NSEG (0 = auto), GGML_CUDA_FA_DENSE_GCN_VAR
// (0: 2 blocks/CU, default; 1: 1 block/CU software pipelined), GGML_CUDA_FA_DENSE_GCN_STATS=1 (per-call log with every
// fallback reason).
bool ggml_cuda_flash_attn_ext_dense_gcn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
