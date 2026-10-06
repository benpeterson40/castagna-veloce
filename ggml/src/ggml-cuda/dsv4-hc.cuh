#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op);

void ggml_cuda_op_dsv4_hc_post_rawpost(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * raw,
                                       float ps1, float ps2);
// HC_POST -> HC_MIX -> HC_PRE -> RMS_NORM -> MUL (weight) in one launch (decode-sized batches)
bool ggml_cuda_dsv4_hc_step_supported(const ggml_tensor * post, const ggml_tensor * mixn, const ggml_tensor * pre,
                                      const ggml_tensor * rn, const ggml_tensor * mul,
                                      bool * yn_temp = nullptr);
void ggml_cuda_dsv4_hc_step(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * mixn, const ggml_tensor * pre,
                            const ggml_tensor * rn, ggml_tensor * mul,
                            bool yn_temp = false);
