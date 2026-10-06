# DeepSeek V4 Flash on 4 MI50s

DeepSeek V4 Flash as antirez's 2-bit GGUF (87 GB), served on 4 cards with tensor
parallelism and the DSpark block drafter.

| | |
| --- | --- |
| Cards | 4 (TP4) |
| Weights | [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf/tree/e7f04037032990db0346398d249baf9fb9df1ccc): `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (86.7 GB) and `DeepSeek-V4-Flash-DSpark-support-0731.gguf` (6.0 GB) |
| Speed | 1,161 tok/s prompt processing; 81.3 tok/s with DSpark, 63.5 tok/s plain ([benchmarks](../BENCHMARKS.md)) |
| Context | 32K by default |

## Download

```sh
hf download antirez/deepseek-v4-gguf --revision e7f04037032990db0346398d249baf9fb9df1ccc \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --local-dir models/deepseek-v4-gguf
```

## Convert the DSpark drafter

The DSpark GGUF uses its own architecture name, which the loader does not know.
[`tools/mi50/dspark_remap.py`](../../tools/mi50/dspark_remap.py) rewrites it into the
layout the engine loads. It takes hyperparameters and the tokenizer from the target, and the
draft shares the target's embeddings and output head at run time. It needs Python 3 with
`numpy` and uses the repository's own `gguf-py`.

```sh
cd models/deepseek-v4-gguf
python3 ../../tools/mi50/dspark_remap.py \
  DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  DeepSeek-V4-Flash-DSpark-dflash-0731.gguf
cd ../..
```

## Serve

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 LLAMA_TP_GROUP=4 \
LLAMA_DSV4_OUTPUT_SPLIT=1 LLAMA_CKPT_LAST_VERIFY=1 GGML_CUDA_GCN_Q8_MVK=1 \
./build/bin/llama-server \
  -m models/deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  -ngl 999 -sm tensor -fa on -c 32768 -b 2048 -ub 2048 --jinja -np 1 \
  -md models/deepseek-v4-gguf/DeepSeek-V4-Flash-DSpark-dflash-0731.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-ngl 999 \
  --host 0.0.0.0 --port 8080
```

| Setting | Why |
| --- | --- |
| `LLAMA_TP_GROUP=4` | One tensor-parallel group over all 4 cards. Without it, 4 cards run as two pairs. |
| `LLAMA_DSV4_OUTPUT_SPLIT=1` | Splits the output head's rows across the cards, so no card reads the whole head every token. |
| `LLAMA_CKPT_LAST_VERIFY=1` | Gives the last prompt chunk the shape of a verify batch, so the first speculative step reuses a ready graph. |
| `GGML_CUDA_GCN_Q8_MVK=1` | A multi-token Q8_0 matvec for the 6-token verify batches: 79.1 → 81.3 tok/s. |
| `-ub 2048` | Prompt processing in 2,048-token batches: 1,161 tok/s. |
| `--spec-draft-n-max 5` | DSpark's block size; 6 or 7 behave like 5. |

**Plain decoding:** drop the `-md` and `--spec-*` arguments and add `GGML_CUDA_ROUTER1_F16=1`.
That fuses the f16 router at one token, which is faster for plain decoding. It is off with
DSpark because one-token steps then route slightly differently from the verify batches,
and fewer drafts are accepted.

## Notes

- DSpark speed depends on the text. Code, lists and math run 1.4–2× faster than plain
  decoding, while stories and poems run at about 0.7×. The 81.3 tok/s is the average over
  20 mixed prompts.
- The model is text-only. For images, see [DeepSeek V4.1 Flash](deepseek-v4.1-flash.md).
