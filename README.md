# Index-TTS2 CUDA

High-throughput, low-latency serving engine for **Index-TTS2** (multilingual base + merged LoRA routing) powered by a zero-dependency custom C++/CUDA GPT-2 inference engine (`sm_89` / `sm_120`) and a TF32 Tensor-Core `s2mel` + `BigVGAN` pipeline.

Replaces the `vLLM` subprocess and duplicate PyTorch GPT-2 copies with a single in-process `2.7 GiB` CUDA library, cutting VRAM by **~4.5–8.8 GiB** and boosting end-to-end TTS throughput by **1.5×–3.0×** with zero precision degradation (GPT in exact FP16, `s2mel` / `BigVGAN` / `conformer` / `semantic_codec` strictly in FP32/TF32 — never FP16).

---

## Highlights

- **Zero-dependency C++/CUDA GPT-2 engine (`csrc/gpt_engine/`)**
  - Fatbin compiled for Ada Lovelace (`sm_89`), Blackwell (`sm_120`), and `compute_89` PTX.
  - Holds both pristine base (`en`/`zh`) and merged adapter (16 routed languages) weights in FP16 with zero runtime LoRA/Punica overhead, deduplicating the 201 shared tensors across models.
  - Static KV cache (`max_batch=4, max_seq_len=1818`), async micro-batching scheduler, and a decoupled `gpt.forward()` workspace so latent passes run concurrently with autoregressive decode.
  - Custom CUDA-graphed kernels: 1-warp zero-smem LayerNorm + fused double-LN (`ln_f → final_norm`), multi-vector FP16 GEMVs, Split-K=16 Flash-Decode attention, and 1024-thread in-GPU top-k / top-p / temperature / repetition-penalty (`10.0`) sampling with zero per-token CPU-GPU sync.
- **Optimized FP32/TF32 `s2mel` + `BigVGAN` Pipeline**
  - Custom `sm_89` TF32 Tensor-Core FlashAttention-2 (`D=64`, FP32 I/O) registered as `torch.ops.indextts.tf32_sdpa_d64` (full-graph `torch.compile` compatible, **2.9× faster** than PyTorch scalar-FP32 SDPA).
  - Weight-norm folding at load time, unmasked SDPA, cached timestep / AdaLN / WaveNet projections, hoisted step-invariant input projections out of the Euler ODE loop, cached speaker `emovec`, fast-path English text normalization, and per-worker CUDA streams with multi-sentence pipelining.
- **Drop-in HTTP API & Instant Rollback**
  - Preserves `/health`, `/tts_url`, `/tts_stream` (NDJSON sentence streaming), GlotLID multilingual routing, and optional CTC `code_aligner` word timings.
  - Every optimization is env-gated (`GPT_ENGINE=cuda|vllm`, `S2MEL_STEPS`, `S2MEL_OPT`, `TTS_TF32`, `TTS_STREAMS`).

---

## Benchmarks

### Standalone Custom CUDA GPT-2 Engine (`tests/test_cuda_gpt_engine.py`)
*Exact FP16 (964.7 MB weights streamed/token), `max_batch=4`, `max_seq_len=1818`, total GPT VRAM = **2.70 GiB** for both base + merged adapter + KV cache:*

| GPU | `B=1` Decode (base & adapter) | `B=2` Aggregate | `B=3` Aggregate | `B=4` Aggregate | `gpt_forward` (`T=256 / 512 / 1024`) |
|---|---:|---:|---:|---:|---:|
| **RTX 5090** (400W) | **984 tok/s** | **1,916 tok/s** | **2,776 tok/s** | **3,532 tok/s** | `3.5 / 7.9 / 21.2 ms` |
| **RTX 4090** (`sm_89`, live prod) | **730–734 tok/s** | — | — | — | — |

### End-to-End Multilingual TTS (`bench_suite.py`, RTX 5090)
*Compared against single-engine vLLM LoRA baseline (`S2MEL_STEPS=12`, `S2MEL_COMPILE=1`, `BIGVGAN_CUDA_KERNEL=1`, `mel_workers=4`, `code_aligner` enabled):*

| Workload | Baseline (`vLLM`, `steps=12`) | **CUDA Engine (`steps=12`)** | **CUDA Engine (`steps=18`, default)** | Speedup (`steps=12` / `18`) |
|---|---:|---:|---:|---:|
| `en_short` (~3s) | `465 ms` | **`261 ms`** | **`306 ms`** | **`1.78×`** / **`1.52×`** |
| `en_med` (~11s) | `1,237 ms` | **`790 ms`** | **`820 ms`** | **`1.57×`** / **`1.51×`** |
| `en_long` (~25s, 2 sents, TTFA / total) | `1,626` / `2,025 ms` | **`926` / `1,134 ms`** | **`1,075` / `1,381 ms`** | **`1.79×`** / **`1.47×`** |
| `ru_short` (~4.5s, adapter) | `955 ms` | **`320 ms`** | **`440 ms`** | **`2.99×`** / **`2.17×`** |
| `ru_med` (~9.5s, adapter) | `1,616 ms` | **`676 ms`** | **`798 ms`** | **`2.39×`** / **`2.03×`** |
| `zh_short` (~5s) | `723 ms` | **`462 ms`** | **`397 ms`** | **`1.57×`** / **`1.82×`** |
| **`burst4`** (2 EN + 2 RU, wall / mean TTFA) | `1,588` / `1,322 ms` | **`847` / `636 ms`** | **`968` / `706 ms`** | **`1.87×–2.08×`** / **`1.64×–1.87×`** |
| **Cold load + warmup** | `177.5 s` | **`42.6 s`** | **`33.9 s`** | **`4.2×–5.2×`** |
| **VRAM savings** | — | **`-4.5 GiB`** (5090) / **`-8.8 GiB`** (4090 prod) | **`-4.4 GiB`** (5090) / **`-8.8 GiB`** (4090 prod) | — |

### Numerical & Audio Quality Parity
- **GPT-2 (`base` & `adapter`):** `gpt_forward` cosine vs PyTorch FP32 = **`0.999998`**, prefill logits cosine = **`0.999997`** (`top-1` exact), 32-step greedy decode = **`32/32` exact**, `B=4 == B=1` bit-exact.
- **`s2mel` (`S2MEL_OPT=1 + TTS_TF32=1 + TF32-FA2`):** mel cosine vs FP32 baseline = **`0.9999999`**.
- **`BigVGAN` (`TTS_TF32=1`):** waveform cosine = **`0.999957`**, mel cosine = **`0.9999997`**, SNR = **`40.6 dB`**.
- **End-to-end A/B (`tests/eval_quality.py`):** CAMPPlus speaker similarity `0.811` (baseline `0.784`), SQUIM PESQ `2.58` (baseline `2.50`), zero clipping, verified with `openai/whisper-large-v3` across EN/RU/ZH.

---

## Quick Start

### 1. Build the CUDA Engine
Requires CUDA toolkit (`nvcc`) and a compatible C++ host compiler (defaults to `/opt/cuda/bin/nvcc` and `/usr/bin/g++-15`; override via `NVCC` / `CXX`):

```bash
bash csrc/gpt_engine/build.sh
```

This compiles `csrc/gpt_engine/libindextts_gpt.so` and verifies `sm_89` / `sm_120` / `compute_89` PTX presence via `cuobjdump`.

### 2. Run the API Server

```bash
export GPT_ENGINE=cuda        # cuda | vllm (rollback)
export TTS_TF32=1             # TF32 for FP32 s2mel/BigVGAN (never FP16)
export S2MEL_OPT=1            # hoisted invariants + unmasked TF32-FA2
export S2MEL_COMPILE=1        # torch.compile DiT step
export S2MEL_STEPS=18         # 12 (fastest), 18 (default), 25 (max steps)
export TTS_STREAMS=1          # dedicated worker CUDA streams
export BIGVGAN_CUDA_KERNEL=1  # fused anti-alias activation kernel

python api_server_v2.py \
  --host 0.0.0.0 --port 5113 \
  --base_model_dir /path/to/models/base_pristine \
  --adapter_model_dir /path/to/models/lora_step12000 \
  --lid_model /path/to/models/glotlid/model.bin \
  --qwen_emo_mode disabled \
  --ref_cache_size 32 \
  --mel_workers 4 \
  --code_aligner /path/to/code_aligner.pt \
  --code_aligner_threads 8
```

For single-model mode (without multilingual routing), pass `--model_dir /path/to/model` instead of `--base_model_dir` / `--adapter_model_dir` / `--lid_model`.

### 3. Tests & Benchmarks

```bash
# GPT-2 parity, VRAM gate, and B=1..4 throughput
python tests/test_cuda_gpt_engine.py

# End-to-end latency, RTF, burst-4 concurrency, and WAV generation
python bench_suite.py --label run1 --out bench_out/run1.json --wav-dir bench_out/wavs

# HTTP API smoke test (/health, /tts_url, /tts_stream)
python tests/test_api_smoke.py
```

---

## API Reference

- **`GET /health`** — returns engine status, mode (`dual`/`single`), active backend (`cuda`/`vllm`), and VRAM usage.
- **`POST /tts_url`** — synthesizes full audio; returns `audio/wav` bytes (or JSON with base64 `audio`, `sampling_rate`, and word-timed `segments` when `"return_segments": true`).
- **`POST /tts_stream`** — streams sentence chunks as NDJSON (`{"text", "language", "start", "end", "words", "sampling_rate", "audio"}`), terminated by `{"done": true}`.

Example request:
```bash
curl -X POST http://127.0.0.1:5113/tts_url \
  -H "Content-Type: application/json" \
  -d '{"spk_audio_path": "/path/to/ref.wav", "text": "Hello from the custom CUDA engine."}' \
  --output out.wav
```

---

## Configuration & Rollback

| Env Var | Default | Description |
|---|---|---|
| `GPT_ENGINE` | `vllm` (`cuda` in `run_multilingual.sh`) | `cuda` for custom `sm_89` engine, `vllm` for instant rollback |
| `GPT_CUDA_MAX_BATCH` | `4` | Static KV cache slot count (keep `<= 4` for `<= 2.8 GiB` VRAM) |
| `S2MEL_STEPS` | `25` (`18` in `run_multilingual.sh`) | Euler ODE diffusion steps (`12` / `18` / `25`) |
| `S2MEL_OPT` | `0` (`1` in `run_multilingual.sh`) | Hoisted DiT invariants, weight-norm removal, custom TF32-FA2 |
| `S2MEL_COMPILE` | `0` (`1` in `run_multilingual.sh`) | `torch.compile(fullgraph=True, dynamic=True)` on DiT step |
| `TTS_TF32` | `0` (`1` in `run_multilingual.sh`) | TF32 matmul & cuDNN acceleration for FP32 `s2mel`/`BigVGAN` |
| `TTS_STREAMS` | `0` (`1` in `run_multilingual.sh`) | Dedicated per-worker CUDA streams + multi-sentence tail pipelining |
| `BIGVGAN_CUDA_KERNEL` | `0` (`1` in `run_multilingual.sh`) | Fused BigVGAN anti-alias activation CUDA extension |

---

## Repository Layout

- `csrc/gpt_engine/` — custom CUDA C++ GPT-2 engine, TF32 FlashAttention-2 kernel, and fatbin build script
- `indextts/gpt/cuda_gpt_engine.py` — ctypes wrapper, async micro-batching scheduler, and `indextts::tf32_sdpa_d64` op
- `indextts/infer_vllm_v2.py` & `indextts/gpt/model_vllm_v2.py` — pipeline integration & dual-view routing
- `indextts/s2mel/` — optimized `s2mel` DiT / WaveNet / BigVGAN modules
- `api_server_v2.py` — FastAPI server (`/health`, `/tts_url`, `/tts_stream`)
- `tests/` & `bench_suite.py` — parity, quality, smoke, and end-to-end benchmark suites

## Credits & License

Built on top of [Index-TTS](https://github.com/index-tts/index-tts) and [index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm). See [LICENSE](LICENSE) and [INDEX_MODEL_LICENSE](INDEX_MODEL_LICENSE).
