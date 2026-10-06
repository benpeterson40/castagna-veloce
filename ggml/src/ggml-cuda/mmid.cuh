#pragma once

struct ggml_cuda_pool;

// With a pool and enough tokens, a stable counting sort (O(slots)) replaces the per-expert scan (O(experts*slots));
// both produce identical output (slots of an expert in increasing token order).
void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream,
        ggml_cuda_pool * pool = nullptr);
