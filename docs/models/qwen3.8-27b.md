# Qwen3.8 27B on 2 MI50s

The dense Qwen3.8 27B, with Gated DeltaNet and full-attention layers and a vision tower, at
4 bits on 2 cards. Speculative decoding uses Unsloth's export of the model's MTP head.

| | |
| --- | --- |
| Cards | 2 (TP2) |
| Weights | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) at revision `4ca72078`: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.6 GB), `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.4 GB), `mmproj-F16.gguf` (0.9 GB) |
| Speed | 702 tok/s prompt processing; 69.5 tok/s with MTP, 43.9 tok/s plain ([benchmarks](../BENCHMARKS.md)); against stock llama.cpp: [Qwen3.8 27B benchmarks](qwen3.8-27b-benchmarks.md) |
| Context | 64K by default |

## Download

```sh
hf download unsloth/Qwen3.8-27B-GGUF --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
  Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf mmproj-F16.gguf \
  --local-dir models/Qwen3.8-27B-GGUF
```

## Serve

```sh
HIP_VISIBLE_DEVICES=0,1 LLAMA_CKPT_LAST_VERIFY=1 \
./build/bin/llama-server \
  -m models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 999 -sm tensor -fa on -c 65536 -ub 1024 --jinja -np 1 \
  -md models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-ngl 999 \
  --host 0.0.0.0 --port 8080
```

For images, add `--mmproj models/Qwen3.8-27B-GGUF/mmproj-F16.gguf`.

| Setting | Why |
| --- | --- |
| `--spec-draft-n-max 2` | The best all-round setting. For mostly code, 4 drafts are faster (82 against 69 tok/s on a code prompt), but stories get slower. |
| `-ub 1024` | Prompt processing in 1,024-token batches; 2,048 and 4,096 are no faster. |
| `LLAMA_CKPT_LAST_VERIFY=1` | Gives the last prompt chunk the shape of a verify batch, so the first MTP step reuses a ready graph. |

## Choosing cards

- **Two x16 cards** if you can. The per-layer all-reduces are faster, and prompt processing
  rises by about 4% (704 → 735 tok/s).
- **Four cards** speed up prompt processing (943 tok/s as two pairs) but not decoding, so
  two cards are the better use of hardware.

## Notes

- Images: a 980×1288 photo becomes 1,257 image tokens and takes 3.1 s.
  `--image-max-tokens 1024` brings that to 2.6 s, but it loses fine detail; at the default
  the model read a faint watermark that it missed at 1024.
- Known issue: the 2-card server sometimes crashes during shutdown, after all requests are
  done. Serving is not affected.
