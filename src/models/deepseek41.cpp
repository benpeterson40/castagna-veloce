#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced. V4 computes and
//    consumes in the same sublayer.
// 2. There is no learned hyper-connection head. Nothing is left to collapse the copies with in V4,
//    which is why it needs one; here the last layer's FFN mix is still unused and does the job.
//    These two facts explain each other, and the file ships no output_hc_* tensors.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs. V4 compresses KV on every layer at one of two fixed ratios;
// V4.1 compresses on a few source layers and the layers after each source read the same rows, and
// it derives index keys from that shared latent rather than from a second compressor. See
// build_attention_v41().
//
// Not implemented: the two level candidate mask. Measured against the reference, it selects every
// block until the compressed length passes candidate_topk_blocks * candidate_block_size, so it
// changes nothing below that and the context is capped there instead.

// mean over the hyper-connection copies; deepseek4.cpp keeps its own copy of this
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] == (uint32_t) il) {
            return (int) e;
        }
    }
    return -1;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, engram_n_layer);
    if (engram_n_layer == 0 || engram_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", engram_n_layer));
    }
    ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_layer_ids);

    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
    ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
    ml.get_key(LLM_KV_ENGRAM_PAD_ID,         engram_pad_id);

    if (hparams.engram_n_head == 0 || hparams.engram_max_ngram_size < 2) {
        throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
    }

    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] >= hparams.n_layer()) {
            throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
        }
    }

    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS, engram_multipliers);
    ml.get_arr(LLM_KV_ENGRAM_PRIMES,      engram_primes);
    ml.get_arr(LLM_KV_ENGRAM_OFFSETS,     engram_offsets);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,   engram_token_map);

    // the hash indexes straight into these, so a short array would read past the end
    const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;

    if (engram_multipliers.size() != (size_t) engram_n_layer * hparams.engram_max_ngram_size) {
        throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (engram_primes.size() != (size_t) engram_n_layer * n_bucket ||
        engram_offsets.size() != engram_primes.size()) {
        throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (uint64_t p : engram_primes) {
        if (p == 0) {
            throw std::runtime_error("engram prime of zero would divide by zero in the hash");
        }
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t o_groups      = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult       = hparams.dsv4_hc_mult;
    const int64_t hc_dim        = hc_mult * n_embd;
    const int64_t hc_mix_dim    = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    if ((size_t) n_vocab > engram_token_map.size()) {
        throw std::runtime_error(format("engram token map has %zu entries, too few for %" PRId64 " tokens",
                                        engram_token_map.size(), n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups),
        // so reshape at load and keep the graph free of it
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it.
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);

        const int eg = engram_index(i);
        if (eg >= 0) {
            const int64_t n_cols = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
            const int64_t key_len = hparams.engram_key_length;

            // The table has hundreds of millions of rows and is far too large to hold in memory,
            // but each token only touches n_cols of them, so read those rows on demand.
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();
            const auto * embd_w = ml.get_weight(embd_name.c_str());
            if (embd_w == nullptr) {
                throw std::runtime_error(format("%s is missing", embd_name.c_str()));
            }
            const int64_t n_rows = embd_w->tensor->ne[1];

            // a row index is a bucket offset plus a hash, so the last bucket has to end inside
            uint64_t max_row = 0;
            for (int64_t b = 0; b < n_cols; ++b) {
                const size_t k = (size_t) eg*n_cols + b;
                max_row = std::max(max_row, engram_offsets[k] + engram_primes[k]);
            }
            if ((int64_t) max_row > n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the engram buckets (%" PRIu64 ")",
                                                embd_name.c_str(), n_rows, max_row));
            }

            layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {key_len, n_rows}, TENSOR_READ_LAZY);
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, 0);
        }

    }

    // Work out which layer publishes the stream each layer reads. Only a source carries a
    // compressor, only an index key owner carries indexer_attn_k, and only an index source
    // carries indexer_attn_q_b, so the file itself says which layer plays which role.
    hparams.dsv41_kv_source.fill(-1);
    hparams.dsv41_index_key_source.fill(-1);
    hparams.dsv41_topk_source.fill(-1);

    int32_t last_kv_source    = -1;
    int32_t last_key_owner    = -1;
    int32_t last_index_source = -1;

    for (int i = 0; i < n_layer; ++i) {
        const auto & layer = layers[i];

        if (layer.attn_comp_wkv)    { last_kv_source    = i; }
        if (layer.indexer_attn_k)   { last_key_owner    = i; }
        if (layer.indexer_attn_q_b) { last_index_source = i; }

        if (hparams.dsv4_compress_ratios[i] == 0) {
            // pure sliding window, no compressed stream to read
            continue;
        }

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            throw std::runtime_error(format("layer %d reads a compressed stream before any layer publishes one", i));
        }

        // the row layout of a stream follows the ratio it was compressed at, so a reader that
        // disagrees with its source would index into rows that stand for different positions
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_kv_source]) {
            throw std::runtime_error(format("layer %d compresses at ratio %u but reads layer %d, compressed at %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_kv_source, hparams.dsv4_compress_ratios[last_kv_source]));
        }

        hparams.dsv41_kv_source[i]        = last_kv_source;
        hparams.dsv41_index_key_source[i] = last_key_owner;
        hparams.dsv41_topk_source[i]      = last_index_source;
    }

    // a compressor with no gate only makes sense where there is nothing to pool
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_is_kv_source(i) && !layers[i].attn_comp_wgate && hparams.dsv4_compress_ratios[i] != 1) {
            throw std::runtime_error(format("layer %d compresses %u tokens per row but has no pooling gate",
                                            i, hparams.dsv4_compress_ratios[i]));
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek41 & pmodel,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg) : pmodel(pmodel), mctx(mctx), eg(eg) {}
    virtual ~llm_graph_input_engram() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (pmodel.hparams.engram_max_ngram_size - 1) * pmodel.hparams.engram_n_head;
        return rows ? rows->ne[0] == n_cols * params.ubatch.n_tokens : emb->ne[1] == params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [n_cols * n_tokens]
    // table in host memory: set_input gathers the rows itself into emb, F32 [key_len * n_cols, n_tokens], instead of a
    // CPU get_rows split in the middle of the graph (LLAMA_DSV41_ENGRAM_HOST=0: off)
    ggml_tensor * emb  = nullptr;
    std::vector<float> ebuf;

    const llama_model_deepseek41 & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult = pmodel.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime = pmodel.engram_primes.data()  + (size_t) eg*n_cols;
    const uint64_t * offset = pmodel.engram_offsets.data() + (size_t) eg*n_cols;

    // image positions (an embd ubatch, ubatch->token null) skip the engram, see the layer loop; text after
    // an image stops its look-back at the image cells, whose ext.tok is LLAMA_TOKEN_NULL (the reference's DEAD)
    const int32_t pad = (int32_t) pmodel.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= pmodel.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) pmodel.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        std::vector<uint64_t> ctx(n_gram);
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    if (rows) {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
        return;
    }

    const ggml_tensor * table = pmodel.layers[hp.engram_layer_ids[eg]].engram_embd;
    const int64_t key_len = table->ne[0];
    const auto * traits = ggml_get_type_traits(table->type);
    ebuf.resize((size_t) key_len*n_cols*n_tokens);
    for (size_t k = 0; k < idx.size(); ++k) {
        GGML_ASSERT(idx[k] >= 0 && idx[k] < table->ne[1]);
        const char * row = (const char *) table->data + (size_t) idx[k]*table->nb[1];
        if (table->type == GGML_TYPE_F32) {
            memcpy(ebuf.data() + k*key_len, row, key_len*sizeof(float));
        } else {
            traits->to_float(row, ebuf.data() + k*key_len, key_len);
        }
    }
    ggml_backend_tensor_set(emb, ebuf.data(), 0, ebuf.size()*sizeof(float));
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_engram>(pmodel, mctx_cur->get_raw(), pmodel.engram_index(il));

    const ggml_tensor * table = model.layers[il].engram_embd;
    static const bool host_env = [] { const char * e = getenv("LLAMA_DSV41_ENGRAM_HOST"); return !e || atoi(e) != 0; }();
    if (host_env && table->buffer && ggml_backend_buffer_is_host(table->buffer) && table->ne[0] == key_len &&
            ggml_get_type_traits(table->type)->to_float) {
        inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, key_len * n_cols, n_tokens);
        ggml_set_input(inp->emb);
        ggml_tensor * emb = inp->emb;
        res->add_input(std::move(inp));
        cb(emb, "engram_embd", il);
        return emb;
    }

    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols * n_tokens);
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.layers[il].engram_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before
    // llama-quant.cpp learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, as_f32(model.layers[il].engram_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(model.layers[il].engram_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    // The reference uses copysign, which treats +0 as positive where ggml_sgn gives 0, so a dot
    // product of exactly zero gates 0.5 here against 0.50025 there. qwen4exp's PLE gate is built
    // the same way.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

// Rope settings for one layer. A layer that reads a compressed stream rotates with the
// compressor's base and YaRN; a plain sliding window layer rotates with the model's.
struct dsv41_rope_cfg {
    float   base;
    float   scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
};

dsv41_rope_cfg llama_model_deepseek41::graph::rope_cfg(int il) const {
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv4_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llama_model_deepseek41::graph::build_attention_tail(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il,
        bool deroped) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = (n_head/n_groups)*n_embd_head;

    const dsv41_rope_cfg rc = rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    if (!deroped) { // the sparse decode op may undo the rotation itself
        out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        out = ggml_rope_set_offset(out, n_embd_head_nope);
    }
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    // a single token's permuted product is already contiguous: a view instead of a copy (LLAMA_DSV41_OA_VIEW=0: copy)
    static const bool oa_view = [] { const char * e = getenv("LLAMA_DSV41_OA_VIEW"); return !e || atoi(e) != 0; }();
    oa = oa_view && ggml_is_contiguous(oa) ? ggml_reshape_2d(ctx0, oa, o_lora_rank*n_groups, nt)
                                           : ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

// Score this layer's queries against the shared index keys and keep the best compressed positions.
// The keys were published by an earlier layer, so this only builds the query side.
ggml_tensor * llama_model_deepseek41::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
    cb(idx_q, "idx_q", il);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;
    if (idx_k_rot) {
        idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
        cb(idx_q, "idx_q_rot", il);
    }

    // one weight per head, scaled so the score matches the reference's
    // softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];
    idx_q = ggml_view_4d(ctx0, idx_q,
            idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
    idx_w = ggml_view_4d(ctx0, idx_w,
            idx_w->ne[0], idx_w->ne[1]/n_stream, idx_w->ne[2], n_stream,
            idx_w->nb[1], idx_w->nb[2]/n_stream, idx_w->nb[3]/n_stream, 0);

    ggml_tensor * score = nullptr;
    ggml_tensor * mask  = inp_comp.kq_mask;
    if (cparams.fused_lid && mask->type == GGML_TYPE_F16) {
        // the fused op (as V4): relu(q.k) weighted over the heads plus the mask, without the per-head score tensor
        // [n_comp, T, heads] and its two transposing copies
        score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, mask);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
        idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        score = ggml_mul_mat(ctx0, idx_k, idx_q);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, idx_w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        // the attention mask is F16 when flash attention is on, and this score is F32. the mask only
        // ever holds 0 or -inf, so widening it is exact.
        if (mask->type != score->type) {
            mask = ggml_cast(ctx0, mask, score->type);
        }

        score = ggml_add(ctx0, score, mask);
    }
    cb(score, "idx_score", il);

    const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, n_top_k));
    cb(top_k, "idx_top_k", il);

    return top_k;
}

// a KV source layer's compressor (shared by the full graph and the CED encoder-only graph, which runs only this part of
// the first decoder layer): the compressed rows into the CSA cache, the index keys, the carried compressor state
void llama_model_deepseek41::graph::build_kv_source_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t ratio       = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(ratio > 0);

    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.state_pos && "a KV source needs a plan with compressor state");

    ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
    cb(state_kv, "comp_state_kv", il);

    // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
    // then runs over a single element and returns 1.0 whatever the score holds, so the
    // values reach the cache unweighted, which is what a plain projection means.
    ggml_tensor * state_score = layer.attn_comp_wgate
        ? build_lora_mm(layer.attn_comp_wgate, cur)
        : state_kv;
    cb(state_score, "comp_state_score", il);

    const dsv4_state_tensors restored = dsv4_build_state_restore(ctx0, inp_comp, comp_state, il);

    ggml_tensor * base_kv = dsv4_view_2d(
            ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
    ggml_tensor * base_score = dsv4_view_2d(
            ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

    ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
    ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

    // the indexer reads the latent before it is rotated, so ask for both forms at once
    ggml_tensor * latent_pre = nullptr;

    ggml_tensor * latent = build_hca_compressed_kv_from_state(
            source_kv,
            source_score,
            inp_comp.state_read_idxs,
            inp_comp.state_write_pos,
            layer.attn_comp_norm,
            ratio,
            n_embd_head,
            "comp_kv",
            il,
            &latent_pre);

    if (hparams.dsv41_owns_index_k(il)) {
        const int64_t n_idx_head_dim  = hparams.indexer_head_size;
        const int64_t n_idx_head_rope = hparams.n_rot();

        ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
        idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
        idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
        cb(idx_k, "idx_k_new", il);

        if (inp_dsv4->get_lid().k_rot) {
            idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(
                    ctx0, idx_k, inp_comp.state_write_idxs, il));
    }

    if (inp_dsv4->get_csa().k_rot) {
        latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
        cb(latent, "comp_kv_rot", il);
    }

    ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(
                ctx0, latent, inp_comp.state_write_idxs, il));

    // carry whatever did not complete a row into the next ubatch
    ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
    ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

    const dsv4_state_tensors snapshot = dsv4_build_state_snapshot(
            ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
    if (snapshot.kv != nullptr) {
        ggml_build_forward_expand(gf, snapshot.kv);
    }
    if (snapshot.score != nullptr) {
        ggml_build_forward_expand(gf, snapshot.score);
    }

    ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inp_comp.state_persist_src_idxs);
    ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

    ggml_build_forward_expand(gf, comp_state->cpy_kv(
                ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
    ggml_build_forward_expand(gf, comp_state->cpy_score(
                ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback, so a reader builds no compressor at all.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    // the sliding window KV projection reads the same token: expanded right after the query's low rank projection, the
    // two run as one launch on the GPU, which also does the kv norm, rope and cache write (LLAMA_DSV41_QKV_ADJ=0 off)
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    static const bool qkv_adj = [] { const char * e = getenv("LLAMA_DSV41_QKV_ADJ"); return !e || atoi(e) != 0; }();
    if (qkv_adj) {
        ggml_build_forward_expand(gf, qr);
        ggml_build_forward_expand(gf, kv);
    }
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    ggml_tensor * q_unrot = q; // for the sparse decode op, which rotates the query itself
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    // the sliding window KV, which every layer keeps for itself
    // reshape before the (per-row) norm so RMS_NORM, MUL, ROPE, VIEW, SET_ROWS stay adjacent for the CUDA fusion
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(model, out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        build_kv_source_v41(model, inp_dsv4, cur, il);
    }

    // an index source picks the positions; the layers in between reuse what it picked (the reference's Reuse layers
    // attend over the last published top-k rows: attending to every compressed position instead only matches it while
    // the stream holds at most top_k rows). LLAMA_DSV41_TOPK_REUSE=0: the old dense readers
    static const bool topk_reuse = [] { const char * e = getenv("LLAMA_DSV41_TOPK_REUSE"); return !e || atoi(e) != 0; }();
    ggml_tensor * top_k = nullptr;
    if (hparams.dsv41_is_index_source(il)) {
        top_k = build_indexer_top_k(model, inp_dsv4, inp_comp, qr, cur, inp_pos, il);
        top_k_of_layer[il] = top_k;
    } else if (topk_reuse && hparams.dsv41_topk_source[il] >= 0) {
        top_k = top_k_of_layer[hparams.dsv41_topk_source[il]];
        GGML_ASSERT(top_k != nullptr && "a reuse layer needs its index source's ranking");
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    // the sparse decode op takes the unrotated query when it rotates it itself (LLAMA_DSV41_SA_ROPE=0: separate rope
    // kernels); the explicit expand only orders the query before the cache writes
    static const bool sparse_op = [] { const char * e = getenv("LLAMA_DSV41_SPARSE_ATTN"); return !e || atoi(e) != 0; }();
    static const bool sa_rope   = [] { const char * e = getenv("LLAMA_DSV41_SA_ROPE"); return !e || atoi(e) != 0; }();
    const bool sparse_ok = sparse_op && top_k && !k_rot && cparams.flash_attn && nt <= 8;
    const bool fuse_rope = sparse_ok && sa_rope && rope_type == LLAMA_ROPE_TYPE_NORM;

    ggml_build_forward_expand(gf, fuse_rope ? q_unrot : q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    // CED: the decoder layers see only the window cells they wrote themselves (an encoder-only prompt token has none)
    ggml_tensor * raw_mask_l = ced_n_enc >= 0 && il >= ced_n_enc ? inp_attn->get_kq_mask_dec() : inp_attn->get_kq_mask();

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    // decode: attend straight over the window cells and each row's top-k compressed rows (GGML_OP_DSV4_SPARSE_ATTN)
    // instead of concatenating the whole compressed cache to the window and masking all but the top-k
    // (LLAMA_DSV41_SPARSE_ATTN=0: the concat + mask path)
    if (sparse_ok && raw_k->ne[1] == 1 && raw_k->ne[3] == 1 && comp_k->ne[1] == 1 && comp_k->ne[3] == 1) {
        // the op applies the query rope and undoes it on the output
        out = ggml_dsv4_sparse_attn(ctx0, fuse_rope ? q_unrot : q, raw_k, raw_mask_l, comp_k, inp_comp.kq_mask,
                top_k, layer.attn_sinks, kq_scale);
        if (fuse_rope) {
            const int64_t n_embd_head_rope = hparams.n_rot();
            ggml_dsv4_sparse_attn_set_rope(out, inp_pos, (int) n_embd_head_rope, (int) (n_embd_head - n_embd_head_rope),
                    rc.n_ctx_orig, rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        }
        cb(out, "attn_out_raw", il);
        out = ggml_reshape_2d(ctx0, out, n_embd_head*n_head, nt);
        return build_attention_tail(model, out, inp_pos, nt, il, fuse_rope);
    }

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
    cb(k_all, "k_all", il);

    ggml_tensor * raw_mask  = raw_mask_l;
    ggml_tensor * comp_mask = top_k
        ? build_top_k_mask(inp_comp.kq_mask, top_k, "comp_top_k_mask", il)
        : inp_comp.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "kq_mask", il);

    const int64_t n_kv_max = top_k
        ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0]
        : 0;

    out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
            nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
}

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    // Causal Encoder-Decoder: layers [0, n_layer/2) are the encoder; the first decoder layer is the decoder's KV source
    ced_n_enc = n_layer/2;
    const bool ced_enc_only = inp_dsv4->mctx->ced_enc_only();
    GGML_ASSERT(!ced_enc_only || hparams.dsv41_is_kv_source(ced_n_enc));

    // an encoder-only ubatch has no outputs (and an unused output-ids input would have no buffer)
    ggml_tensor * inp_out_ids = ced_enc_only ? nullptr : build_inp_out_ids();

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream. An image arrives as an embd
        // ubatch (no token ids, all of it image span) and the reference zeroes the engram gate there: skip it
        if (pmodel.engram_index(il) >= 0 && ubatch.token != nullptr) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        // CED encoder-only ubatch (prompt tokens before the decoder's replay window): of the first decoder layer only
        // what writes the decoder's global KV and index keys from the encoder's output, C_l = H_{L/2} W_l^KV
        if (ced_enc_only && il == ced_n_enc) {
            cur = build_hc_pre(inpL, pre_mix, il);
            cb(cur, "hc_attn_pre", il);
            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);
            build_kv_source_v41(model, inp_dsv4, cur, il);
            return;
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);
        // place the mix right after the stream it reads (the previous layer's HC post), not where its results are first
        // used: the backend then fuses post -> mix -> pre -> norm into one kernel at decode
        ggml_build_forward_expand(gf, attn_pre);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention_v41(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
