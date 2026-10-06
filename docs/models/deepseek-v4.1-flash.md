# DeepSeek V4.1 Flash on 8 MI50s

DeepSeek V4.1 Flash at Q2_K with its vision projector, served on 8 cards as two
tensor-parallel groups of 4.

| | |
| --- | --- |
| Cards | 8 (two TP4 groups) |
| Weights | [vcruz305/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/c4a085541cb53f67ee5e57b63d255e80cef286e7) at revision `c4a08554`: `DeepSeek-V4.1-Flash-Q2_K-*` (7 files, 265 GB); vision projector from [smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/fb2ce0313f74a0e1c3950cdf3e051c124dd0b39e/mmproj-DeepSeek-V4.1-Flash-BF16.gguf) at revision `fb2ce031` |
| Speed | 1,218 tok/s prompt processing, 79.3 tok/s plain decoding (2026-10-02, older build) |
| Context | 8K in the tested setup |

> [!NOTE]
> This setup was last verified on 2026-10-03, on an earlier build than the 4-card models.

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

## Notes

- **Turn thinking off.** With thinking on, this 2-bit model falls into endless reasoning
  loops in about half of all runs or more, for text and image prompts alike. Send
  `"chat_template_kwargs": {"enable_thinking": false}`. With thinking off it answered 7 of
  7 image test cases correctly.
- Multi-image prompts that ask for every image's title can make it invent the second
  image's title.
- The vision encoder accumulates in F32. Its output is 0.16–0.22% off a full-precision
  reference, against 4.7–6.6% for the F16-accumulating path.
