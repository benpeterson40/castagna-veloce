# GLM-5.3-Flash on 4 or 8 MI50s

GLM-5.3-Flash, a hybrid of KDA linear attention and DSA sparse attention with
hyper-connections, mixture-of-experts layers and a vision tower. It runs on 4 cards at 2 bits
or 8 cards at 4 bits. Speculative decoding uses the model's own MTP (nextn) layer.

| | |
| --- | --- |
| Cards | 4 (TP4) for UD-Q2_K_XL (109 GB); 8 (two TP4 groups) for UD-Q4_K_XL (200 GB) |
| Weights | [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) at revision `621d456e`, plus `mmproj-F16.gguf` for images |
| Speed (Q2, 4 cards) | 978 tok/s prompt processing; 79.5 tok/s with MTP, 70.2 tok/s plain ([benchmarks](../BENCHMARKS.md)) |
| Context | 64K by default on 4 cards; 128K fits, 200K does not |

## Q2 on 4 cards

```sh
hf download unsloth/GLM-5.3-Flash-GGUF --revision 621d456e93e926e4b52f85cff5f634358c1828f9 \
  --include "UD-Q2_K_XL/*" "mmproj-F16.gguf" --local-dir models/GLM-5.3-Flash-GGUF

HIP_VISIBLE_DEVICES=0,1,2,3 LLAMA_TP_GROUP=4 LLAMA_CKPT_LAST_VERIFY=1 \
./build/bin/llama-server \
  -m models/GLM-5.3-Flash-GGUF/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  -ngl 999 -sm tensor -fa on -c 65536 -b 4096 -ub 1024 --jinja -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  --host 0.0.0.0 --port 8080
```

For images, add `--mmproj models/GLM-5.3-Flash-GGUF/mmproj-F16.gguf --image-max-tokens 2048`.

| Setting | Why |
| --- | --- |
| `--spec-draft-n-max 1` | One MTP draft per step is fastest: 79 tok/s, against 67 with 2 drafts. |
| `-ub 1024` | Up to 64K context. Above that, use 512 so the prefill buffers fit. |
| `--image-max-tokens 2048` | Caps huge photos. A 3000×4006 photo otherwise needs a 1.8 GB encoder buffer on the first card and fails. |
| `LLAMA_CKPT_LAST_VERIFY=1` | Gives the last prompt chunk the shape of a verify batch, so the first MTP step reuses a ready graph. |

**Faster prompts, text only:** `-ub 1536` speeds up prompt processing by 6–14% at 64K context
(4.4K-token prompt: 785 → 891 tok/s; 40K: 771 → 817). It leaves only about 0.3 GB free per card, so it
doesn't fit next to the vision encoder.

## Reasoning

The chat template always opens `<think>`; `enable_thinking` has no effect. Control the depth
per request:

```json
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

`"high"` and `"low"` are much faster than the default `"max"`. A greedy palindrome-function
prompt took 2,625 tokens at max, 468 at high and 386 at low. To turn reasoning off for the
whole server, use `--reasoning-budget 0`.

## Q4 on 8 cards

```sh
hf download unsloth/GLM-5.3-Flash-GGUF --revision 621d456e93e926e4b52f85cff5f634358c1828f9 \
  --include "UD-Q4_K_XL/*" --local-dir models/GLM-5.3-Flash-GGUF

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLAMA_TP_GROUP=4 LLAMA_TP_LAYER_SPLIT=22,23 \
./build/bin/llama-server \
  -m models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  -ngl 999 -sm tensor -fa on -c 32768 -b 4096 -ub 1024 --jinja -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  --host 0.0.0.0 --port 8080
```

This runs two tensor-parallel groups of 4 cards. `LLAMA_TP_LAYER_SPLIT=22,23` divides the
layers between the groups, which is best for prefill. Up to 32K context use `-ub 1024`;
above that use `-ub 512`. Measured on an older build (2026-10-02): 1,210 tok/s prompt
processing and 55.0 tok/s plain decoding. The 8-card setup has not been re-verified on the
current build yet.
