#pragma once

#include "common.cuh"

// Exact sparse QSA prefill attention for gfx906 (see fattn-qsa-gcn.cu). i = index of the CONT(TOP_K) node that starts the
// indexer mask chain CONT -> FILL -> SET_ROWS -> ADD -> (views) -> FLASH_ATTN_EXT. Returns the number of extra nodes
// consumed (up to and including the FLASH_ATTN_EXT), or 0 when the path does not apply (nothing was launched).
// GGML_CUDA_FA_QSA_GCN=0 disables it (default 1); GGML_CUDA_FA_QSA_MIN_T (16), GGML_CUDA_FA_QSA_MIN_KV (0 = by T: 2304 for
// T >= 32, else 3072), GGML_CUDA_FA_QSA_NSEG (0 = auto), GGML_CUDA_FA_QSA_STATS=1 (per-call log with every fallback reason;
// union statistics when not capturing, which syncs: debug only).
int ggml_cuda_qsa_attn_gcn(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
