#pragma once

#include "common.cuh"

// decode (<= 4 tokens), GCN: SCALE -> SILU -> q8_0 HC up projection -> RESHAPE -> gated DSV4_HC_PRE in one kernel
bool ggml_cuda_hc_up_pre_dec_supported(int cc, const ggml_tensor * scale, const ggml_tensor * silu, const ggml_tensor * mm,
                                       const ggml_tensor * pre);
void ggml_cuda_hc_up_pre_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * scale, const ggml_tensor * mm, ggml_tensor * pre);

// decode (<= 4 tokens), GCN: GDN alpha/beta projections with their epilogues (dt bias, softplus, A; sigmoid) in one kernel
bool ggml_cuda_gdn_ab_dec_supported(int cc, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * sp,
                                    const ggml_tensor * mul, const ggml_tensor * mm_b, const ggml_tensor * sig);
// gate = the MUL node, sig = the SIGMOID node
void ggml_cuda_gdn_ab_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * mm_b,
                          ggml_tensor * gate, ggml_tensor * sig);

// shared-expert gate tail: MUL_MAT(1-row gate, x) -> SIGMOID -> MUL(shexp) -> ADD(moe) in one kernel (<= 8 tokens)
bool ggml_cuda_shexp_gate_add_supported(const ggml_tensor * mm, const ggml_tensor * sig, const ggml_tensor * mul,
                                        const ggml_tensor * add);
void ggml_cuda_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig,
                              const ggml_tensor * mul, ggml_tensor * add);
