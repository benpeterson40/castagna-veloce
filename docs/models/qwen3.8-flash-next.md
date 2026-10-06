# Qwen3.8 Flash-Next on 4 MI50s

Qwen3.8 Flash-Next: a mixture of experts with hyper-connections, Gated DeltaNet and sparse
attention, plus a vision tower. It runs at 4 bits on 4 cards, with Unsloth's export of its
MTP head as the drafter. There are two serving profiles: one tuned for decoding and one for
long prompts.

| | |
| --- | --- |
| Cards | 4 |
| Weights | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66) at revision `38bb39ee`: `UD-Q4_K_XL/` (111 GB, 4 files), `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.8 GB), `mmproj-F16.gguf` |
| Speed | Decode profile: 1,939 tok/s prompt processing; 119.0 tok/s with MTP, 79.5 plain. Prefill profile: 2,615 tok/s prompt processing; 79.1 tok/s with MTP, 52.3 plain ([benchmarks](../BENCHMARKS.md)) |
| Context | 32K by default |

## Download

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-F16.gguf" \
  --local-dir models/Qwen3.8-Flash-Next-GGUF
```

## Decode profile (chat, agents)

The 4 cards run as two tensor-parallel pairs, with persistent kernels for the
hyper-connection and MoE steps: 119.0 tok/s with MTP (79.5 plain) and 1,939 tok/s prompt
processing.

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 \
GGML_CUDA_VIRTUAL_PER_GPU=1 GGML_CUDA_HC_PERSIST=4 GGML_CUDA_HCP_AR=1 GGML_CUDA_HCP_INJ=1 \
GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_MTP_DRAFT_VOCAB=98304 LLAMA_CKPT_LAST_VERIFY=1 \
./build/bin/llama-server \
  -m models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 999 -fa on -sm tensor -b 2048 -ub 320 -c 32768 --jinja -np 1 \
  -md models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 999 \
  --host 0.0.0.0 --port 8080
```

| Setting | Why |
| --- | --- |
| `GGML_CUDA_VIRTUAL_PER_GPU=1` | One pipeline stage per card; the build default of several helps only prompt processing. |
| `GGML_CUDA_HC_PERSIST=4` | Persistent kernels for the hyper-connection and MoE steps of each layer. |
| `GGML_CUDA_HCP_AR=1` | The all-reduce of each pair fused into those kernels. |
| `GGML_CUDA_HCP_INJ=1` | The hyper-connection inject step fused as well. It is correct with pairs only, so never combine it with `LLAMA_TP_GROUP=4`. |
| `GGML_CUDA_DISABLE_GRAPHS=1` | HIP graphs off: +4% plain decoding and +2% with MTP for this model. |
| `LLAMA_MTP_DRAFT_VOCAB=98304` | The drafter scores only the first 96K token ids, split across each pair. BPE order puts the common tokens first, and the target still checks every draft against the full vocabulary. |

## Prefill profile (long prompts, documents)

Layer split over the 4 cards with large batches: 2,615 tok/s on a 2,048-token prompt, and
79.1 tok/s with MTP (52.3 plain). Pick it for long prompts and documents; pick the decode
profile when answers are long compared with the prompts. Like every prompt-processing figure
here, the 2,615 tok/s is measured without the drafter loaded.

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 \
LLAMA_MTP_DRAFT_VOCAB=98304 LLAMA_GRAPH_CACHE2=0 LLAMA_CKPT_LAST_VERIFY=1 \
./build/bin/llama-server \
  -m models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 999 -fa on -sm layer -b 2048 -ub 320 -c 32768 --jinja -np 1 \
  -md models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-ngl 999 \
  --host 0.0.0.0 --port 8080
```

`LLAMA_GRAPH_CACHE2=0` turns off the drafter's second graph slot, which trades a little
decoding speed for faster prompt processing in this profile. Loading the drafter costs about
8% of prompt-processing speed (measured on an earlier build). For the fastest possible
prompts, drop `-md` and the `--spec-*` arguments, at the price of 52 tok/s decoding.

## Notes

- For images, add `--mmproj models/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf`. Images work in
  both profiles with MTP on (last checked 2026-10-02).
- Qwen thinks by default. Use `"chat_template_kwargs": {"enable_thinking": false}` for
  direct answers.
