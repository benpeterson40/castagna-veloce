#pragma once

#include "common.cuh"

// dense q8_0 MUL_MAT for GCN with 8x4 register tiles (default on GCN, GGML_CUDA_GCN_GEMM=0 disables); returns false if the shape is not covered
bool ggml_cuda_gcn_q8_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// prefill HC up projection fused with the gated DSV4_HC_PRE (GCN q8_0 GEMM with a mixing epilogue)
bool ggml_cuda_gcn_hc_up_mix_supported(int cc, const ggml_tensor * mm, const ggml_tensor * pre);
void ggml_cuda_gcn_hc_up_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * pre);

// dense q4_K / q5_K / q6_K / iq4_xs MUL_MAT for GCN with the same tiling (default on, GGML_CUDA_GCN_KQ=0 disables)
bool ggml_cuda_gcn_kq_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID with q4_K / q5_K / q6_K experts on the v4 tiles, grouped by expert (prefill; GGML_CUDA_GCN_KQ_MOE=0 off,
// GGML_CUDA_GCN_KQ_MOE_MIN_AVG: minimum average tokens per expert, default 12). up_src0: the gate/up pair, dst =
// SwiGLU-clamp(src0 . x, up_src0 . x) with glu_limit (INFINITY: plain SwiGLU)
bool ggml_cuda_gcn_kq_moe_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst);
void ggml_cuda_gcn_kq_moe(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                          const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up_src0 = nullptr,
                          float glu_limit = INFINITY);
