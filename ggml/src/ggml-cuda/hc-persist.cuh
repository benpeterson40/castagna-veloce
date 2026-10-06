#pragma once
#include "common.cuh"

// persistent decode HC pre (1-4 tokens): RMSNorm -> MUL -> 2x RESHAPE -> HC down -> SCALE -> SILU -> HC up -> RESHAPE
// -> gated DSV4_HC_PRE in one kernel with one grid barrier (GCN, 60 CUs, one device per GPU: GGML_CUDA_VIRTUAL_PER_GPU=1).
// Opt-in GGML_CUDA_HC_PERSIST: 1 = the chain, 2 = also the preceding combine (SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST),
// 3 = also the inject MUL_MAT (computed by the previous persistent kernel into a stash), 4 = also the router logits and the
// shared expert (computed after two more grid barriers into a stash the next kernel adds). GGML_CUDA_HC_PERSIST_DEBUG=1 logs
// the first launches.
bool ggml_cuda_hc_persist_enabled();
// i = index of the RMS_NORM node; returns true when nodes i..i+9 form the chain and the device can run it
bool ggml_cuda_hc_persist_match(int device, const ggml_cgraph * cgraph, int i);
void ggml_cuda_hc_persist(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// GGML_CUDA_HC_PERSIST=2: also the preceding combine, i = index of the SCALE of SCALE -> SIGMOID -> SCALE -> DSV4_HC_POST
// inj >= 0: node inj (right before i) is the inject MUL_MAT whose result is in the stash (starts the fused range)
bool ggml_cuda_hc_persist_post_match(int device, const ggml_cgraph * cgraph, int i, int inj);
void ggml_cuda_hc_persist_post(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool raw_from_stash);
// mode 4: extra fused node (the router logits MUL_MAT) after the chain at i (0 or 1), and the shared expert consumer: i = the
// shared expert gate MUL_MAT recorded in ctx.hcp_shexp_node (23 nodes up to the next chain's DSV4_HC_PRE)
int  ggml_cuda_hc_persist_ffn_ext(const ggml_cgraph * cgraph, int i);
bool ggml_cuda_hc_persist_shexp_match(int device, const ggml_cgraph * cgraph, int i);
void ggml_cuda_hc_persist_shexp(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// the recorded shared expert run when the combine is not in this graph (-sm tensor splits): ffn_out = moe_out + stash
bool ggml_cuda_hc_persist_shexp_local_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_hc_persist_shexp_local(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// -sm tensor: an inject MUL_MAT at i whose stash was produced in an earlier split (records the pairing otherwise)
bool ggml_cuda_hc_persist_inject_other(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// mode 4: nodes the persistent kernel already computed (early matvecs); returns the group length to skip (0: none)
int ggml_cuda_hc_persist_done(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
// GDN gated norm (RMS_NORM, MUL, RESHAPE, SIGMOID, MUL) in one kernel, with the q8_1 copy of the output projection input
int  ggml_cuda_hc_gdn_gated_norm_match(const ggml_cgraph * cgraph, int i, int * mm_idx, int * z_mm);
void ggml_cuda_hc_gdn_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int mm_idx, int o = 0);
// fused tensor-parallel AllReduce (GGML_CUDA_HCP_AR=1): the comm backend defers qualifying AllReduces (2 GPUs, [2560, <=4]
// f32); each graph evaluation's start decides whether a persistent kernel absorbs it, else runs the exchange right away
bool ggml_cuda_hcp_ar_defer(const int * dev_ids, int n, ggml_tensor ** tensors, void * const * streams);
void ggml_cuda_hcp_ar_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);
// attention output gate at decode: CONT(gate view) -> SIGMOID -> MUL(attn) [-> q8_0 MUL_MAT (q8 cache)] in one kernel;
// returns the node count (0: no), GGML_CUDA_ATTN_GATE=0 off
int  ggml_cuda_hc_attn_gate_match(const ggml_cgraph * cgraph, int i, int * mm_idx);
void ggml_cuda_hc_attn_gate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int mm_idx);
// indexer sparse mask (long contexts): CONT(top_k) -> FILL -> SET_ROWS -> ADD in one kernel (GGML_CUDA_QSA_MASK_FUSE=0 off);
// returns the extra nodes consumed
int ggml_cuda_hc_qsa_mask(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// the chain those fusions consume (shared matcher): i = the CONT(TOP_K) node; the CONT, FILL and SET_ROWS results are read
// only inside the chain (directly or through views), so skipping them is safe
struct ggml_cuda_qsa_chain {
    int idx[4];                 // node indices of CONT, FILL, SET_ROWS, ADD (only views in between)
    const ggml_tensor * tk;     // the TOP_K result, I32 [width, T]
    const ggml_tensor * mask;   // the KQ mask, F16 [n_kv, T]
    ggml_tensor * ad;           // the ADD: the sparse mask (mask at the selected cells, -inf elsewhere)
    int width;
};
// why (optional): the reason when it returns false (for GGML_CUDA_FA_QSA_STATS)
bool ggml_cuda_hc_qsa_chain_match(const ggml_cgraph * cgraph, int i, ggml_cuda_qsa_chain & c, const char ** why = nullptr);
bool ggml_cuda_hc_is_view_op(const ggml_tensor * t);

// PLE depthwise conv tail + residual adds in one kernel; returns the extra node count (0: not fused)
int ggml_cuda_hc_ple_conv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
