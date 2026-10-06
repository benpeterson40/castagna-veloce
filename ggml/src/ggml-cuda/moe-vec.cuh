#pragma once

#include "common.cuh"

// MUL_MAT_ID with few tokens per expert (GCN): tokens sorted by expert, tiles of up to 16 tokens, weights read once per tile
bool ggml_cuda_moe_vec_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                 const ggml_tensor * dst);

// up_src0: fused gate/up + SwiGLU (dst = silu(src0 . x) * (up_src0 . x)), see ggml_cuda_moe_vec_pair; glu_limit: the
// SWIGLU_CLAMP limit (INFINITY: plain SwiGLU)
void ggml_cuda_moe_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up_src0 = nullptr,
                       float glu_limit = INFINITY);

// MUL_MAT_ID gate + MUL_MAT_ID up + SWIGLU / SWIGLU_CLAMP on the GCN row-lane path: one preparation, the up launch applies it
bool ggml_cuda_moe_vec_pair_supported(int cc, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu);
void ggml_cuda_moe_vec_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * glu);

// decode-sized MUL_MAT_ID (few tokens) on GCN through the K-split row-lane kernel
bool ggml_cuda_moe_vec_decode_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                        const ggml_tensor * dst);

// dense q2_K / q3_K matmul with 2..16 tokens on GCN through the row-lane kernels (multi-sequence decode)
bool ggml_cuda_rowlane_dense_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_rowlane_dense(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// single-token MoE FFN with a shared expert as one extra "expert" (DeepSeek V4.1 decode): routed gate/up (q2_K) + shared
// gate/up -> SwiGLU-clamp, then routed down (q3_K) weighted by the router + shared down, summed in expert order; replaces
// MUL_MAT_ID x3, GLU, MUL, VIEW/ADD reduction, MUL_MAT x3, GLU and the final ADD (2 kernels)
struct ggml_cuda_moe1_args {
    const ggml_tensor * x;          // f32 [K] (one token)
    const ggml_tensor * ids;        // i32 [n_used]
    const ggml_tensor * weights;    // f32 [n_used] router weights
    const ggml_tensor * gate_exps;  // q2_K [K, M, n_expert]
    const ggml_tensor * up_exps;
    const ggml_tensor * down_exps;  // q3_K [M, N, n_expert]
    const ggml_tensor * gate_sh;    // q2_K [K, M]
    const ggml_tensor * up_sh;
    const ggml_tensor * down_sh;    // q3_K [M, N]
    float               limit;      // SwiGLU clamp
    ggml_tensor *       dst;        // f32 [N]
};
bool ggml_cuda_moe1_supported(int cc, const ggml_cuda_moe1_args & a);
void ggml_cuda_moe1_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_moe1_args & a);

// single-token dense q2_K matvec (GCN): src1 one column per channel
bool ggml_cuda_gemv1_q2k_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// several single-token q2_K matvecs sharing src1 (and K) in one launch
#define GGML_CUDA_GEMV1_MULTI_MAX 6
struct ggml_cuda_gemv1_multi {
    const char * w[GGML_CUDA_GEMV1_MULTI_MAX];
    float      * y[GGML_CUDA_GEMV1_MULTI_MAX];
    int64_t      s_row[GGML_CUDA_GEMV1_MULTI_MAX];
    int          M[GGML_CUDA_GEMV1_MULTI_MAX];
    int          wg0[GGML_CUDA_GEMV1_MULTI_MAX];
    int          n;
};
// RMS_NORM -> MUL (weight) -> ROPE (norm mode, pairs [n_offs, n_offs + n_dims)) -> SET_ROWS (f16 row) of one segment's
// single-token output, done by the segment's last workgroup (the decode kv projection's norm, rope and cache write)
struct ggml_cuda_gemv1_epi {
    int             seg;          // segment index (-1: none)
    const float   * nw;           // norm weight
    float           eps;
    int             n_dims, n_offs;
    const int32_t * pos;
    float           freq_scale, ext_factor, attn_factor, theta_scale, corr0, corr1;
    const float   * freq_factors; // or null
    half          * dst;          // cache rows
    const int64_t * row_idx;      // the token's row
    int64_t         stride;       // elements per cache row
    int           * counter;      // arrivals of the segment's workgroups (reset by the last)
};
void ggml_cuda_gemv1_q2k_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mms, int n,
                               const ggml_cuda_gemv1_epi * epi = nullptr);
// norm_w: the token is RMS_NORM(src1, norm_eps)*norm_w first (fused RMS_NORM -> MUL -> MUL_MAT)
void ggml_cuda_gemv1_q2k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         const float * norm_w = nullptr, float norm_eps = 0.0f);
