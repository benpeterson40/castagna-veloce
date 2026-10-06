# DeepSeek V4.1 Flash on 8 MI50s

DeepSeek V4.1 Flash at Q2_K with its vision projector, served on 8 cards as two
tensor-parallel groups of 4.

| | |
| --- | --- |
| Cards | 8 (two TP4 groups) |
| Weights | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/c4a085541cb53f67ee5e57b63d255e80cef286e7) at revision `c4a08554`: `DeepSeek-V4.1-Flash-Q2_K-*` (7 files, 265 GB); vision projector from [smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/fb2ce0313f74a0e1c3950cdf3e051c124dd0b39e/mmproj-DeepSeek-V4.1-Flash-BF16.gguf) at revision `fb2ce031` |
| Speed | 1,316 tok/s prompt processing; 85.7 tok/s plain decoding, 86.4 in llama-bench ([benchmarks](../BENCHMARKS.md)) |
| Memory | 26.3 of 34.3 GB per card at 8K context; the two engram tables (64.5 GB) stay in system RAM |
| Context | 8K in the tested setup |

## Download

```sh
hf download vcruz305/DeepSeek-V4.1-Flash-GGUF --revision c4a085541cb53f67ee5e57b63d255e80cef286e7 \
  --include "DeepSeek-V4.1-Flash-Q2_K-*" --local-dir models/DeepSeek-V4.1-Flash-GGUF
hf download smalinin/DeepSeek-V4.1-Flash-GGUF --revision fb2ce0313f74a0e1c3950cdf3e051c124dd0b39e \
  mmproj-DeepSeek-V4.1-Flash-BF16.gguf --local-dir models/DeepSeek-V4.1-Flash-GGUF
```

## Serve

```sh
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLAMA_TP_GROUP=4 \
./build/bin/llama-server \
  -m models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
  -ngl 999 -sm tensor -fa on -c 8192 --jinja -np 1 \
  --mmproj models/DeepSeek-V4.1-Flash-GGUF/mmproj-DeepSeek-V4.1-Flash-BF16.gguf \
  --host 0.0.0.0 --port 8080
```

## Speculative decoding (DSpark)

V4.1 ships its own DSpark drafter. smalinin publishes it as a GGUF:

```sh
hf download smalinin/DeepSeek-V4.1-Flash-GGUF --revision fb2ce0313f74a0e1c3950cdf3e051c124dd0b39e \
  DeepSeek-V4.1-Flash-DSpark-AUTO.gguf --local-dir models/DeepSeek-V4.1-Flash-GGUF
```

To use it, add these arguments to the serve command:

```sh
  -md models/DeepSeek-V4.1-Flash-GGUF/DeepSeek-V4.1-Flash-DSpark-AUTO.gguf \
  --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-ngl 999
```

It works, but on these cards it is slower than plain decoding: at best 62.7 tok/s with 2
draft tokens, against 85.7 plain. Verifying even 2 tokens costs about twice a plain token,
because V4.1's multi-token path has not been tuned the way DeepSeek V4's has
([details](../BENCHMARKS.md#deepseek-v41-with-dspark)). Serve V4.1 without the drafter
until that work is done.

## Notes

- **Turn thinking off.** With thinking on, this 2-bit model falls into endless reasoning
  loops in about half of all runs or more, for text and image prompts alike. Send
  `"chat_template_kwargs": {"enable_thinking": false}`. With thinking off it answered 7 of
  7 image test cases correctly.
- Multi-image prompts that ask for every image's title can make it invent the second
  image's title.
- The vision encoder accumulates in F32. Its output is 0.16–0.22% off a full-precision
  reference, against 4.7–6.6% for the F16-accumulating path.
