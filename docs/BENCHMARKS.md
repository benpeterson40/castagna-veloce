# Benchmarks

All numbers are for one user and one request at a time, on the system below. Each
[model guide](models) has the exact serve command used.

## Test system

| Part | Details |
| --- | --- |
| GPUs | 8× AMD Instinct MI50 32 GB (VBIOS 113-D1631700-111); a model uses 2, 4 or 8 of them as listed |
| PCIe | 4.0; 3 of the 8 slots are x16 and the others x8 |
| Host | AMD 64-thread server CPU, ~250 GB RAM |
| OS | Ubuntu 26.04.1 LTS, Linux 7.0.0 |
| ROCm | Ubuntu's own ROCm 7.1 packages (HIP 7.1, rocBLAS/hipBLAS 7.1), Clang 21.1.8 |
| Power | amdgpu defaults: performance level `auto`, except card 0 at `high` |

Build: the CMake configuration from the [README](../README.md#1-build).

## Prompt processing and plain decoding

`llama-bench` with each model's serve settings: the same cards, split mode, ubatch and
environment. Each test runs 4 or 5 times. The first run is a warm-up and is dropped, and the
table shows the mean of the rest; those agree within 1%. Measured 2026-10-06.

| Model | Cards | Settings | pp2048 | tg128 (AR) |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash IQ2XXS | 4 | TP4, ubatch 2048 | 1,160.8 | 63.5 |
| GLM-5.3-Flash UD-Q2_K_XL | 4 | TP4, ubatch 1024 | 978.4 | 70.2 |
| Qwen3.8 27B UD-Q4_K_XL | 2 | TP2, ubatch 1024 | 702.1 | 43.9 |
| Qwen3.8 27B UD-Q4_K_XL | 4 | two TP2 pairs, ubatch 1024 | 943.2 | – |
| Qwen3.8 Flash-Next UD-Q4_K_XL | 4 | decode profile (two TP2 pairs, persistent kernels, ubatch 320) | 1,939.2 | 79.5 |
| Qwen3.8 Flash-Next UD-Q4_K_XL | 4 | prefill profile (layer split, ubatch 320) | 2,614.7 | 52.3 |
| GLM-5.3-Flash UD-Q4_K_XL | 8 | two TP4 groups, ubatch 1024 | 1,298.6 | 63.7 |
| DeepSeek V4.1 Flash Q2_K | 8 | two TP4 groups, ubatch 512 | 1,316.4 | 86.4 |

Example, GLM-5.3-Flash:

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 LLAMA_TP_GROUP=4 ./build/bin/llama-bench \
  -m models/GLM-5.3-Flash-GGUF/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  -ngl 999 -sm tensor -fa 1 -b 4096 -ub 1024 -p 2048 -n 128 -r 4 -o json
```

## Speculative decoding (serving)

Speculative decoding speed depends on the text, so one prompt says little: single prompts
swing by ±20%. [`tools/mi50/srvbig.py`](../tools/mi50/srvbig.py) sends 20 varied prompts
to a running `llama-server`:
- code in four languages;
- explanations;
- a poem, a story and a haiku;
- lists, translations and math.

Each prompt runs once:
- 200 tokens, greedy (temperature 0);
- prompt cache off;
- `enable_thinking: false`.

The result is the total number of generated tokens divided by the total decode time, plus
the share of drafted tokens the model accepted.

```sh
python3 tools/mi50/srvbig.py 8080
```

| Model | Drafter | Decode t/s | Drafts accepted | Prompt time per request |
| --- | --- | --- | --- | --- |
| DeepSeek V4 Flash | DSpark, 5 tokens | 81.3 | 43.1% | 364 ms |
| GLM-5.3-Flash Q2 | MTP, 1 token | 79.5 | 83.4% | 423 ms |
| Qwen3.8 27B (2 cards) | MTP, 2 tokens | 69.5 | 75.5% | 239 ms |
| Qwen3.8 Flash-Next (decode profile) | MTP, 3 tokens | 119.0 | 69.3% | 200 ms |
| Qwen3.8 Flash-Next (prefill profile) | MTP, 3 tokens | 79.1 | 70.2% | 175 ms |
| GLM-5.3-Flash Q4 (8 cards) | MTP, 1 token | 78.9 | 86.8% | 476 ms |
| DeepSeek V4.1 Flash (8 cards) | none: plain decoding is faster (below) | 85.7 | – | 242 ms |

Measured 2026-10-05 (Flash-Next and the 8-card models 2026-10-06) on the current code.

GLM-5.3-Flash ignores `enable_thinking` and always reasons, so its 200 tokens per prompt
are mostly reasoning text.

The text matters a lot: with DSpark, DeepSeek V4 Flash runs code, lists and math 1.4–2× faster
than plain decoding, but stories, poems and haiku at about 0.7×.

## DeepSeek V4.1 with DSpark

DeepSeek V4.1 ships its own DSpark drafter. It runs here, but on these cards it is slower
than plain decoding. Same 20-prompt set, 2026-10-06:

| Setting | Decode t/s | Drafts accepted |
| --- | --- | --- |
| Plain decoding | **85.7** | – |
| DSpark, 1 draft token | 62.1 | 73.3% |
| DSpark, 2 draft tokens | 62.7 | 60.8% |
| DSpark, 3 draft tokens | 62.3 | 52.6% |
| DSpark, 5 draft tokens | 47.8 | 37.3% |
| DSpark, adaptive length (up to 5) | 49.8 | 42.8% |
| DSpark, confidence cut-off 0.5 | 30.5 | 75.0% |

The verify step is the problem. A forward pass over 2, 4 or 6 tokens costs about 22, 34 or
49 ms on V4.1, against 11.7 ms for one plain token. DeepSeek V4, whose multi-token path has
been tuned, takes 20, 25 and 30 ms against 16.6 ms. V4.1's multi-token path has not had that
work yet.

## How changes are validated

- **Kernels and fusions:** each one has a `test-backend-ops` case that compares it with the
  CPU backend. The fused paths are confirmed to actually run in the real model graph, with
  `GGML_CUDA_OP_PROFILE`.
- **Perplexity:** every model's perplexity on a fixed text is re-checked after each change,
  at the batch sizes the change affects. For example, ubatch 2 or 3 covers the MTP verify
  batches.
  - Changes meant to be exact must stay bit-identical.
  - Changes that alter rounding are compared with the noise floor of other rounding changes,
    and judged on the 20-prompt serving set, not on one prompt.
- **Stability:** a stress run (long generations and multi-turn chats with prompt-cache
  reuse) before deploying.
