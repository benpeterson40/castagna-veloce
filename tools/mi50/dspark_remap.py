#!/usr/bin/env python3
# dspark_remap.py SRC TARGET OUT: DeepSeek-V4-Flash DSpark support GGUF (arch deepseek4-dspark, stages as mtp.N.*) ->
# the layout llama.cpp's DFlash loader reads for a DSV4 DSpark draft (arch dflash, see src/models/dflash.cpp):
#   mtp.N.<block tensor>            -> blk.N.<block tensor>        (full DSV4 blocks, uncompressed sliding-window attention)
#   mtp.0.main_proj / main_norm     -> fc / enc.output_norm        (fusion of the target's layers dspark.target_layer_ids)
#   mtp.<last>.norm                 -> output_norm
#   mtp.<last>.hc_head_*            -> output_hc_*                 (present = V4 flavor: same-sublayer hyper-connections)
#   mtp.<last>.markov_head.*        -> markov_w1 / markov_w2       (DSpark Markov head)
#   mtp.<last>.confidence_head.proj -> conf_proj
# Hyperparameters and the tokenizer come from the target GGUF; the draft's mask token is DSpark's noise token.
# Token embeddings and the LM head are shared with the target at run time (ctx_other).
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))  # the repository's gguf-py
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType as VT  # noqa: E402

src_path, tgt_path, out_path = sys.argv[1:4]
src = GGUFReader(src_path)
tgt = GGUFReader(tgt_path)


def fv(r, k):
    f = r.fields.get(k)
    return f.contents() if f is not None else None


def ftype(r, k):
    f = r.fields[k]
    return f.types[0], (f.types[-1] if f.types[0] == VT.ARRAY else None)


A = "dflash"
TA = fv(tgt, "general.architecture")
assert TA == "deepseek4", TA
assert fv(src, "general.architecture") == "deepseek4-dspark"
n_layers = int(fv(src, "dspark.n_layers"))
last = n_layers - 1

w = GGUFWriter(out_path, A)
w.add_name(str(fv(src, "general.name") or "DeepSeek V4 Flash DSpark"))

# target hyperparameters: same key under the draft's arch; per-layer arrays cut to the draft's stages
for key in ["context_length", "embedding_length", "attention.head_count", "attention.head_count_kv", "rope.freq_base",
            "rope.dimension_count", "attention.layer_norm_rms_epsilon", "expert_used_count", "expert_gating_func",
            "attention.key_length", "attention.value_length", "vocab_size", "attention.q_lora_rank",
            "attention.output_lora_rank", "attention.output_group_count", "expert_feed_forward_length", "expert_count",
            "expert_shared_count", "expert_weights_scale", "expert_weights_norm", "attention.sliding_window",
            "hyper_connection.count", "hyper_connection.sinkhorn_iterations", "hyper_connection.epsilon",
            "swiglu_clamp_exp", "swiglu_clamp_shexp"]:
    tk = f"{TA}.{key}"
    if tk not in tgt.fields:
        print(f"note: target has no {tk}")
        continue
    vt, st = ftype(tgt, tk)
    val = fv(tgt, tk)
    if vt == VT.ARRAY:
        val = list(val)[:n_layers]
    w.add_key_value(f"{A}.{key}", val, vt, sub_type=st)

w.add_key_value(f"{A}.block_count", n_layers, VT.UINT32)
w.add_key_value(f"{A}.attention.compress_ratios", [0]*n_layers, VT.ARRAY, sub_type=VT.UINT32)
w.add_key_value(f"{A}.block_size", int(fv(src, "dspark.block_size")), VT.UINT32)
w.add_key_value(f"{A}.target_layers", [int(x) for x in fv(src, "dspark.target_layer_ids")], VT.ARRAY, sub_type=VT.INT32)
w.add_key_value(f"{A}.has_confidence_head", True, VT.BOOL)

# tokenizer of the target, mask token = DSpark noise token
noise = int(fv(src, "dspark.noise_token_id"))
for f in tgt.fields.values():
    if not f.name.startswith("tokenizer."):
        continue
    if f.name == "tokenizer.ggml.mask_token_id":
        continue
    vt, st = ftype(tgt, f.name)
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_key_value("tokenizer.ggml.mask_token_id", noise, VT.UINT32)

# tensors
head = {
    "main_proj.weight": "fc.weight",
    "main_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "hc_head_fn.weight": "output_hc_fn.weight",
    "hc_head_base.weight": "output_hc_base.weight",
    "hc_head_scale.weight": "output_hc_scale.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_w2.weight",
    "confidence_head.proj.weight": "conf_proj.weight",
}
out = []
for t in src.tensors:
    m = re.match(r"mtp\.(\d+)\.(.*)", t.name)
    assert m, t.name
    il, rest = int(m.group(1)), m.group(2)
    if rest in head:
        if rest.startswith("hc_head") and il != last:
            print(f"skip {t.name} (only the last stage's HC head is the output head)")
            continue
        name = head[rest]
    else:
        assert rest.startswith(("attn_", "ffn_", "hc_", "exp_probs_b")), t.name
        name = f"blk.{il}.{rest}"
    out.append((name, t))
names = [n for n, _ in out]
assert len(names) == len(set(names)), "duplicate tensor names"
for need in ["fc.weight", "enc.output_norm.weight", "output_norm.weight", "output_hc_fn.weight", "markov_w1.weight"]:
    assert need in names, need

for name, t in out:
    w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, t in out:
    w.write_tensor_data(t.data, tensor_endianess=src.endianess)
w.close()
print(f"wrote {out_path}: {len(out)} tensors, {n_layers} stages, mask token {noise}")
