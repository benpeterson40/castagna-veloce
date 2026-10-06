#pragma once

#include "common.cuh"

// GCN dense F16 x F32 matvec for 1..8 tokens: lane-coalesced 16-byte weight loads, f32 sums (replaces MMVF there)
bool ggml_cuda_gcn_f16_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
bool ggml_cuda_gcn_f16_matvec_supported(int device, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// up to GGML_CUDA_F16MV_MAXM F16 matrices of one K on the same activations in one launch (MUL_MAT nodes mms[0..n))
#define GGML_CUDA_F16MV_MAXM 8
struct ggml_cuda_f16mv_list {
    const char * W[GGML_CUDA_F16MV_MAXM];
    float      * dst[GGML_CUDA_F16MV_MAXM];
    int64_t      w_stride[GGML_CUDA_F16MV_MAXM];
    int64_t      d_stride[GGML_CUDA_F16MV_MAXM];
    int          row_end[GGML_CUDA_F16MV_MAXM]; // cumulative rows
    int          nm;
};
bool ggml_cuda_gcn_f16_matvec_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mms, int n);
// GCN dense F32 x F32 matvec for 1..4 tokens and 65..8192 rows (e.g. MoE routers): coalesced 16-byte loads, K split over waves
bool ggml_cuda_gcn_f32_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
