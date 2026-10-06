# Castagna Veloce: the AMD Instinct MI50 inference engine

<p align="center">
  <img src="assets/castagna-veloce.jpg" alt="Castagna Veloce logo: a cartoon chestnut running fast" width="520">
</p>

Castagna Veloce is an inference engine built and tuned for a single GPU: the AMD Instinct MI50
(Vega 20, `gfx906`), with 32 GB of HBM2 per card. It runs current large models across 2 to
8 cards: the DeepSeek V4 Flash, GLM-5.3-Flash and Qwen3.8 Flash-Next mixtures of experts and
the dense Qwen3.8 27B. The cards are joined by tensor parallelism over plain PCIe, and
speculative decoding (MTP, DSpark) makes single-user generation fast.

It is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). It uses the same GGUF
weights and the same `llama-server` with its OpenAI-compatible API and Web UI. On top of that
it adds gfx906 kernels, fused operations for MoE-era architectures, and multi-GPU plumbing
designed for cards without matrix cores or fast interconnects.

**Issues and contributions are welcome,** especially from other MI50 and MI60 owners.

## Models and benchmarks

| Model | Cards | Inference modes | Hugging Face weights | Single-user speed |
| --- | --- | --- | --- | --- |
| [DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) | 4 (TP4) | AR, DSpark | antirez [IQ2XXS 0731](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf) · [DSpark 0731](https://huggingface.co/antirez/deepseek-v4-gguf/blob/e7f04037032990db0346398d249baf9fb9df1ccc/DeepSeek-V4-Flash-DSpark-support-0731.gguf) | **1,161 tok/s pp**; **81.3 tok/s tg** with DSpark (63.5 AR) |
| [GLM-5.3-Flash](docs/models/glm-5.3-flash.md) | 4 (TP4) | AR, MTP, images | Unsloth [UD-Q2_K_XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q2_K_XL) · [mmproj F16](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/blob/621d456e93e926e4b52f85cff5f634358c1828f9/mmproj-F16.gguf) | **978 tok/s pp**; **79.5 tok/s tg** with MTP (70.2 AR) |
| [Qwen3.8 27B](docs/models/qwen3.8-27b.md) | 2 (TP2) | AR, MTP, images | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) · [MTP Q4_0](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/MTP/mtp-Qwen3.8-27B-Q4_0.gguf) · [mmproj F16](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/mmproj-F16.gguf) | **702 tok/s pp**; **69.5 tok/s tg** with MTP (43.9 AR) |
| [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next.md) | 4 | AR, MTP, images | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q4_K_XL) · [MTP Q8_0](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf) · [mmproj F16](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf) | **2,615 tok/s pp** (prefill profile); **119.0 tok/s tg** with MTP (decode profile; 79.5 AR) |
| [GLM-5.3-Flash Q4](docs/models/glm-5.3-flash.md#q4-on-8-cards) | 8 (TP4×2) | AR, MTP, images | Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q4_K_XL) | 1,210 tok/s pp; 55.0 tok/s tg AR† |
| [DeepSeek V4.1 Flash](docs/models/deepseek-v4.1-flash.md) | 8 (TP4×2) | AR, images | vcruz305 [Q2_K](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/c4a085541cb53f67ee5e57b63d255e80cef286e7) · smalinin [mmproj BF16](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/fb2ce0313f74a0e1c3950cdf3e051c124dd0b39e/mmproj-DeepSeek-V4.1-Flash-BF16.gguf) | 1,218 tok/s pp; 79.3 tok/s tg AR† |

**pp** is llama-bench prompt processing of a 2,048-token prompt. **AR** is plain autoregressive
decoding (llama-bench, 128 tokens). The speculative **tg** numbers are the `llama-server`
decode rate over [20 varied prompts](docs/BENCHMARKS.md#speculative-decoding-serving)
of 200 greedy tokens each. Speculative speed depends on the text: code, lists and
translations draft far better than stories. All numbers are for a single user, measured
2026-10-05/06 on the current code. † Measured 2026-10-02 on an older build, from before most
of the kernel and speculative-decoding work; the 8-card setups have not been re-verified
since. Each model guide lists the exact command and settings, and
[BENCHMARKS.md](docs/BENCHMARKS.md) has the full method.

## New features

Everything below was written for Castagna Veloce, and none of it is in upstream llama.cpp
(checked against v0.6.0). Most of it can be switched off with an environment variable for
A/B tests. The speed figures were measured on 4 cards (Qwen3.8 27B on 2) when each feature
landed.

### Kernels for the MI50

- **i-quant MoE decode kernels** for the IQ2_XXS, IQ2_XS, IQ3_XXS and IQ4_XS experts of the
  2-bit models, for batches of up to 64 tokens: GLM-5.3 decoding went from 41 to 62 tok/s.
- **Expert-grouped quantized MoE GEMM for prefill**, i-quants included: GLM-5.3 prompt
  processing 682 → 960 tok/s.
- **Small-batch matvecs** for K-quants (q4_K, q5_K, q6_K, iq4_xs), Q8_0 (with a split-K
  variant), F16 and F32, for 1 to 16 tokens. They are sized for speculative-decoding verify
  batches.
- **q2_K MoE kernel with aligned 16-byte loads** for DeepSeek V4's expert down projection:
  101 → 71 µs at 6 tokens.
- **64-head lightning indexer** for DeepSeek V4's verify and prefill batches: a verify call
  went from 118 to 15 µs, and prefill rose 6%.
- **Sparse attention for DeepSeek V4's compressed layers** in decode and prefill: prefill
  +16%. At contexts up to about 2K tokens the indexer is skipped (decode 53 → 58 tok/s).
- **`DSV4_COMP_POOL`**, a new op that runs DeepSeek V4's compressor gather, pooling and
  norm as one kernel: decode +8%. Skipping its padding blocks gave another +7%.
- **Prefill GEMM fixes:** split-K F16/F32 GEMMs for few-row, long-K shapes, and grids
  reordered for memory channels in MoE reduction and HC mixing: DeepSeek V4 prefill
  1,050 → 1,143 tok/s.

### Fused operations

- **Hyper-connection step** in one kernel (HC post, mix, pre, RMS norm and scale), with F16
  or quantized weights at 1–6 tokens: DeepSeek V4 decoding +11%, and DSpark serving
  75 → 80 tok/s.
- **DeepSeek V4 state-row, position-embedding and multi-matrix fusions:** kernel launches
  per token went from 4,400 to 1,200, and decoding from 37 to 44 tok/s.
- **Router and top-k in one kernel:** matvec, gating, top-k and weight normalization, for
  DeepSeek V4.1 at one token and GLM-5.3 at 1–4 tokens.
- **Gate/up + SwiGLU, matvec + residual add, broadcast gate and small-concat kernels** for
  the verify batches. Together with the small-batch matvecs they took Qwen3.8 27B MTP from
  65.6 to 69.4 tok/s.
- **KDA gated-norm fusion, Q8 activations reused across weight types, and batched sibling
  matvecs:** GLM-5.3 about +4%.

### Multiple cards

- **PCIe all-reduce for tensor parallelism:** kernel peer writes for decode-sized tensors and
  copy-engine DMA for prefill-sized ones (DMA: +2% prefill).
- **Persistent hyper-connection and MoE kernels** with each pair's all-reduce fused in, for
  the Qwen3.8 Flash-Next decode profile.
- **Split output head** for DeepSeek V4, with a mirrored copy for the DSpark drafter: plain
  decoding +3%.
- **Several virtual devices per card** for pipelined layer-split prefill.

### Speculative decoding and serving

- **DSpark for DeepSeek V4 Flash:** a converter from antirez's DSpark GGUF to llama.cpp's
  DFlash draft format, plus the KV-cache and crash fixes that made it run. It reaches
  81 tok/s, against 64 for plain decoding.
- **MTP drafter graph reuse** and **catch-up merged into the draft step:** GLM-5.3
  +4–8% and +5–9%.
- **Verify-shaped last prompt chunk** (`LLAMA_CKPT_LAST_VERIFY`): one graph rebuild fewer per
  request, +1–2% throughput and a faster first token.
- **Draft vocabulary prefix** (`LLAMA_MTP_DRAFT_VOCAB`): the drafter scores only the first
  96K token ids, split across the cards, while the target still checks every draft against
  the full vocabulary. This is the idea of
  [FR-Spec](https://arxiv.org/abs/2502.14856) applied to an MTP head.

### Accuracy

- **F32-accumulating vision encoders** for GLM-5.3, Qwen3.8 and DeepSeek V4.1: 0.13–1.7% off a
  full-precision reference, against 4.7–23% before. This costs some encode time, not speed
  elsewhere.

## Philosophy

- One target: AMD Instinct MI50/MI60 (`gfx906`). Every kernel choice is measured on these
  cards. Other GPUs still build, but they are untested.
- 4 cards per model as the main target, the sweet spot for 32 GB cards and today's large
  MoE models at 2–4 bits; 8 cards as two 4-card groups.
- Single-user latency first: speculative decoding is the default wherever a model has a
  drafter.
- Preserve quality. Every new kernel or fusion has a `test-backend-ops` case checked
  against the CPU backend, and every model's perplexity is re-checked after each change.
  Changes meant to be exact are verified bit-identical. Changes that alter rounding say so
  and are judged over many prompts, not one.
- Nearly every optimization has a runtime switch, usually a `GGML_CUDA_*` environment
  variable, to turn it off for A/B tests and bug reports.
- Stay close to llama.cpp: GGUF, `llama-server`, the usual tools and flags.

## Quickstart

### 1. Build

Supported setup: Ubuntu 26.04 LTS with its own ROCm 7.1 packages, which still ship `gfx906`
code. Kernel 7.0, HIP 7.1, LLVM/Clang 21.

```sh
sudo apt install build-essential cmake git hipcc libamdhip64-dev libhipblas-dev librocblas-dev \
  rocm-device-libs-21 rocminfo rocm-smi
git clone https://github.com/benpeterson40/castagna-veloce
cd castagna-veloce
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_FA=ON \
  -DGGML_SCHED_MAX_COPIES=8 -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server llama-cli llama-bench llama-perplexity
rocminfo | grep -c gfx906   # one line per card
```

Your user needs access to `/dev/kfd` and `/dev/dri/renderD*`, usually through the `render`
and `video` groups. Log out and back in after changing membership. If CMake picks the wrong
compiler, pass `-DCMAKE_HIP_COMPILER=/usr/lib/llvm-21/bin/clang++`.

### 2. Download a model

For example GLM-5.3-Flash at 2 bits (about 109 GB, 4 cards):

```sh
pip install -U huggingface_hub   # provides the hf command
hf download unsloth/GLM-5.3-Flash-GGUF --revision 621d456e93e926e4b52f85cff5f634358c1828f9 \
  --include "UD-Q2_K_XL/*" --local-dir models/GLM-5.3-Flash-GGUF
```

### 3. Serve

```sh
HIP_VISIBLE_DEVICES=0,1,2,3 LLAMA_TP_GROUP=4 LLAMA_CKPT_LAST_VERIFY=1 \
./build/bin/llama-server \
  -m models/GLM-5.3-Flash-GGUF/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  -ngl 999 -sm tensor -fa on -c 65536 -b 4096 -ub 1024 --jinja -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  --host 0.0.0.0 --port 8080
```

Then, from another terminal, ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say something"}]}'
```

`LLAMA_TP_GROUP=4` makes the 4 cards one tensor-parallel group, and `--spec-type draft-mtp`
turns on the model's own MTP drafter. The [GLM-5.3-Flash guide](docs/models/glm-5.3-flash.md)
explains every setting.

The Web UI is at `http://localhost:8080`. For Open WebUI, the OpenAI SDK and similar
clients, set the base URL to `http://localhost:8080/v1`. Each [model guide](docs/models)
has the download and serve commands for that model.

## Usage notes

- **GLM-5.3-Flash always reasons:** its template opens `<think>` and has no
  `enable_thinking` switch. Set the depth per request with
  `"chat_template_kwargs": {"reasoning_effort": "low"}` (or `"high"`; the default is
  `"max"`), or turn reasoning off with `--reasoning-budget 0`.
- **Images:** add `--mmproj` with the model's projector file (see its guide). The encoder
  runs on the first card, and a very large photo can need over a gigabyte there, so cap
  images with `--image-max-tokens 2048`.
- **DeepSeek V4.1 Flash Q2_K** can fall into endless reasoning loops with thinking on. Use
  `"chat_template_kwargs": {"enable_thinking": false}`, especially for images.
- **One user per server** (`-np 1`) is the tuned configuration. The kernels for 1–6-token
  batches are what make speculative decoding fast.

## Hardware notes

- **Cards:** tested on MI50 32 GB (VBIOS 113-D1631700-111). The MI60 is the same `gfx906`
  chip and should work, but it is untested. The 16 GB MI50 needs more cards for the same
  models.
- **PCIe slots matter for prefill.** The tensor-parallel all-reduce moves several MB per
  layer during prefill. On our board only 3 of 8 slots are x16; a 2-card model on two x16
  cards prefills about 4% faster than on an x16 + x8 pair. Decode is barely affected.
- **Clocks:** fixing the cards at the high performance level
  (`echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level`)
  avoids clock ramps between drafting and verifying. With speculative decoding we measured
  +0.4% on GLM-5.3 and about +5% on Flash-Next (the latter on an earlier build). The
  published numbers use the default `auto` on all cards but the first.
- **Cooling:** the MI50 is passively cooled and needs real front-to-back airflow.

## License

Castagna Veloce is a fork of llama.cpp and keeps its [MIT license](LICENSE). The MI50-specific
changes are MIT licensed as well. Model weights are not included and keep their publishers'
licenses.

Castagna Veloce is not affiliated with or endorsed by AMD. AMD, Instinct and Radeon are
trademarks of Advanced Micro Devices, Inc.

## Acknowledgements

- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), the base of everything here.
- [danielhanchen](https://github.com/ggml-org/llama.cpp/pull/28243) for the Qwen3.8
  Flash-Next MTP draft head, the branch this fork started from.
- [timkhronos](https://github.com/ggml-org/llama.cpp/pull/27773) for GLM-5.3-Flash support,
  ported here from an earlier revision.
- [smalinin/llama.cpp](https://github.com/smalinin/llama.cpp/tree/my_build_deepseek41) for
  DeepSeek V4.1 support (the `deepseek41` model and its vision encoder), ported here from an
  earlier revision.
- [antirez](https://github.com/antirez/ds4) for the DeepSeek V4 Flash GGUFs and the DSpark
  drafter.
- [Unsloth](https://huggingface.co/unsloth) for the GLM-5.3-Flash and Qwen3.8 GGUFs,
  [vcruz305](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) for the DeepSeek V4.1
  Flash GGUF, and [smalinin](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF) for its
  vision projector.
- [gufo](https://github.com/gufo-org/gufo), the Strix Halo engine whose landing page this one
  follows.
