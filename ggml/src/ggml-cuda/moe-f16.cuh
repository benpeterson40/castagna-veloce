#pragma once

#include "common.cuh"

// MUL_MAT_ID for large batches on RDNA3 via F16 WMMA: expert weights are dequantized to F16 on their way
// into shared memory and the whole K range accumulates in F32 (no per-block int8 scale epilogue).
bool ggml_cuda_moe_f16_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                 const ggml_tensor * dst);

// out16: dst stored as F16 in place (caller verified that every consumer reads F16)
void ggml_cuda_moe_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       const ggml_tensor * ids, ggml_tensor * dst, bool out16 = false);

// gate/up MUL_MAT_ID pair + SWIGLU fused: one expert sort, one activation conversion, silu(gate)*up stored into glu
bool ggml_cuda_moe_f16_pair_supported(int cc, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * glu);

// out16: glu is stored as F16 in place (only when its sole consumer is a MUL_MAT_ID taking the F16 MoE path)
void ggml_cuda_moe_f16_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
                            ggml_tensor * glu, bool out16);

// dense q8_0 MUL_MAT for large batches on RDNA3, same F16-WMMA kernel without the expert gather
bool ggml_cuda_dense_f16_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_dense_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                         ggml_tensor * dst);

// hyper-connection up projection (q8_0) fused with the gated DSV4_HC_PRE stream mix: mm = MUL_MAT(w_up, lo),
// pre = DSV4_HC_PRE(xn, reshape(mm)); writes pre, never stores mm
bool ggml_cuda_hc_up_mix_supported(int cc, const ggml_tensor * mm, const ggml_tensor * pre);

void ggml_cuda_hc_up_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * pre);

// [DSV4_HC_POST ->] RMS_NORM -> MUL -> RESHAPE -> RESHAPE -> MUL_MAT(q8_0 HC down): one combine+norm pass that stores
// the combine and xn outputs and hands an F16 xn straight to the down projection. post_node may be nullptr.
bool ggml_cuda_hc_norm_down_supported(int cc, const ggml_tensor * post_node, const ggml_tensor * rms,
                                      const ggml_tensor * mul, const ggml_tensor * mm, const ggml_tensor * raw_post = nullptr);

// raw_post: post weights = ps2 * sigmoid(ps1 * raw_post) instead of post_node->src[2]
void ggml_cuda_hc_norm_down(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                            ggml_tensor * mul, ggml_tensor * mm, bool xn_f16_inplace,
                            const ggml_tensor * raw_post = nullptr, float ps1 = 1.0f, float ps2 = 1.0f);

// SIGMOID(z) -> MUL(a, .) -> RESHAPE -> MUL_MAT(q8_0 dense): the gated product goes to the projection as F16 directly
bool ggml_cuda_gate_proj_supported(int cc, const ggml_tensor * sig, const ggml_tensor * mul, const ggml_tensor * mm);
void ggml_cuda_gate_proj(ggml_backend_cuda_context & ctx, const ggml_tensor * sig, const ggml_tensor * mul, ggml_tensor * mm);

// DSV4_HC_POST -> RMS_NORM -> MUL in one pass (any GPU, any batch); raw_post: weights ps2*sigmoid(ps1*raw_post)
bool ggml_cuda_hc_combine_norm_supported(const ggml_tensor * post_node, const ggml_tensor * rms, const ggml_tensor * mul,
                                         const ggml_tensor * raw_post = nullptr);
void ggml_cuda_hc_combine_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                               ggml_tensor * mul, const ggml_tensor * raw_post, float ps1, float ps2);

bool ggml_cuda_hc_xn_f16_inplace_safe(const ggml_tensor * post_node, const ggml_tensor * mul, const ggml_tensor * raw_post);

// decode / verify (< 16 tokens) variant of the combine + norm: one block per (stream, token)
bool ggml_cuda_hc_combine_norm_dec_supported(const ggml_tensor * post_node, const ggml_tensor * rms, const ggml_tensor * mul,
                                             const ggml_tensor * raw_post);
void ggml_cuda_hc_combine_norm_dec(ggml_backend_cuda_context & ctx, ggml_tensor * post_node, const ggml_tensor * rms,
                                   ggml_tensor * mul, const ggml_tensor * raw_post, float ps1, float ps2,
                                   void * xq = nullptr, int64_t s_xq = 0); // optional q8_1 copy of xn (rows of 4*n_embd)
