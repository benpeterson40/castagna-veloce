#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// deferred conv-state concat for the gated-DeltaNet short convolution (see ssm-conv.cu)
bool ggml_cuda_ssm_conv_deferred_ok(const ggml_tensor * concat, const ggml_tensor * conv, const ggml_tensor * silu);
void ggml_cuda_ssm_conv_deferred(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * conv,
                                 ggml_tensor * silu, ggml_tensor * tail);

// decode / verify: the whole gated-DeltaNet conv block (concat, conv-state tails, conv + SILU, q/k l2 norms) in one kernel
bool ggml_cuda_gdn_conv_block_dec_ok(const ggml_tensor * concat, const ggml_tensor * conv, const ggml_tensor * silu,
                                     const ggml_tensor * const * tails_cpy, int n_slots,
                                     const ggml_tensor * q_rms, const ggml_tensor * q_scale,
                                     const ggml_tensor * k_rms, const ggml_tensor * k_scale, bool * need_barrier);
void ggml_cuda_gdn_conv_block_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * conv,
                                  ggml_tensor * silu, const ggml_tensor * const * tails_view, const ggml_tensor * const * tails_cpy,
                                  int n_slots, const ggml_tensor * q_rms, ggml_tensor * q_scale,
                                  const ggml_tensor * k_rms, ggml_tensor * k_scale, bool need_barrier,
                                  const ggml_tensor * state_rows = nullptr);

// GLM-5-Next KDA decode / verify: the conv block of separate q / k / v projections (their CONCATs, the weight CONCATs,
// the conv-state concat and tails, conv + SILU, q/k L2 norms) in one kernel
bool ggml_cuda_kda_conv_block_ok(const ggml_tensor * qk, const ggml_tensor * qkv, const ggml_tensor * concat,
                                 const ggml_tensor * conv, const ggml_tensor * silu, const ggml_tensor * const * tails_cpy,
                                 int n_slots, const ggml_tensor * q_l2, const ggml_tensor * k_l2, bool * need_barrier);
void ggml_cuda_kda_conv_block(ggml_backend_cuda_context & ctx, const ggml_tensor * qk, const ggml_tensor * qkv,
                              const ggml_tensor * concat, const ggml_tensor * conv, ggml_tensor * silu,
                              const ggml_tensor * const * tails_view, const ggml_tensor * const * tails_cpy, int n_slots,
                              ggml_tensor * q_l2, ggml_tensor * k_l2, bool need_barrier, const ggml_tensor * state_rows = nullptr);
