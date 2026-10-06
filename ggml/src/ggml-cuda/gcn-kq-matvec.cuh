#pragma once

#include "common.cuh"

// GCN dense q4_K / q5_K / q6_K / iq4_xs matvec for 1..4 tokens: lane-coalesced 16-byte quant runs, q8_1 activations in LDS
bool ggml_cuda_gcn_kq_matvec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// add: an addend of dst's shape added as dst is written (ggml_cuda_gcn_kq_matvec_add_supported); nullptr: plain MUL_MAT
void ggml_cuda_gcn_kq_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_tensor * add = nullptr);
bool ggml_cuda_gcn_kq_matvec_add_supported(int cc, const ggml_tensor * mm, const ggml_tensor * add_node, const ggml_tensor * addend);
// gate/up + SwiGLU at one token (q4_K / q5_K, GGML_CUDA_GCN_KQ_MV1=1); false: not handled
bool ggml_cuda_gcn_kq_matvec_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst, float glu_limit = INFINITY);
