#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// true if the GDN node takes the column-per-thread prefill kernel with q/k normalized in-kernel from q_raw/k_raw
bool ggml_cuda_gdn_colthread_qknorm_ok(const ggml_tensor * gdn, const ggml_tensor * q_raw, const ggml_tensor * k_raw);

// recurrent-state GET_ROWS that the decode GDN kernel can replace by reading the cache row directly
bool ggml_cuda_gdn_state_rows_ok(const ggml_tensor * get_rows, const ggml_tensor * gdn);
