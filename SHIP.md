# SHIP — one button, old model keeps serving until you press it

`UNPACK.md` describes the multilingual server and how to deploy it by hand. This
file is the automated version of its "Acceptance before flipping prod" section.

**The button:**

```bash
/home/forsen/repos/index-tts-ml/ship.sh
```

**Rollback:**

```bash
/home/forsen/repos/index-tts-ml/ship.sh rollback
```

## The staging layout

Nothing about the running service was changed to prepare this. Prod is still
`index-tts-updated.service` running master out of `/home/forsen/repos/index-tts-vllm-fast`,
and it stays that way — across restarts and reboots — until `ship.sh` is run.

| piece | where | state |
|---|---|---|
| multilingual code | `/home/forsen/repos/index-tts-ml` (git worktree, branch `multilingual-routing`) | checked out, tests pass (198 passed, 26 skipped) |
| weights | `/home/forsen/tts_models` | 25GB, no dangling symlinks |
| extra deps | conda env `index-tts-vllm-fast` | `fasttext-predict`, `num2words`, `pytest` installed — three new packages, nothing existing was upgraded, so the running server is unaffected |
| launch | `run_multilingual.sh` | single-engine LoRA mode per UNPACK.md, port 5113 |
| unit | `index-tts-multilingual.service` | installed in `/etc/systemd/system`, **disabled and stopped** |
| old prod | `index-tts-updated.service` | active, enabled — untouched |

The new server was deliberately **never booted** on this box (the user chose a
cold press): the first real boot happens when you press the button, and the
script rolls back if it does not come up. Expect the flip to take several
minutes — first boot compiles the torch.compile/BigVGAN caches.

## What the button does

1. **Preflight** — branch, clean worktree, weights present + no dangling symlinks
   + ~24GB on disk, deps importable, new modules import, installed unit matches
   the repo copy, VRAM math, and that the old unit is currently serving (so the
   rollback target is known). Any failure aborts before anything is touched.
   `ship.sh preflight` runs exactly this and nothing else, any time, safely.
2. **Baseline** — records what English sounds like on the live old server right
   now (`ship_artifacts/baseline_en.wav`), for the post-flip comparison.
3. **10s abort window** (skip with `ship.sh -y`).
4. **Flip** — stops the old unit, waits for the VRAM and port 5113 to actually
   come back, starts the multilingual unit, polls `/health` for up to 15 min.
5. **Acceptance checks** (`tools/ship_checks.py smoke`) — health/mode/engines,
   EN forced + EN auto route to the base view, ZH tripwire, RU digits route to
   the adapter without runaway, a mixed EN+RU message splitting across both
   views, the 400 guards, and a `/tts_stream` NDJSON run (the only path
   forsen_ai_api uses).
6. **Auto-rollback** on any hard failure: stops the new unit, waits for VRAM,
   starts the old one, re-checks health, and prints the GPU process list so an
   orphan EngineCore holding ~15GB is visible immediately.

Success leaves a **runtime-only** flip: a reboot still brings back the old
server. `ship.sh promote` makes the multilingual unit the boot default, and is
meant to be run after a soak, not straight after the flip.

`ship.sh status` prints what is serving, `/health`, and the GPU process list.

## What the button deliberately does not decide

- **Whether it sounds right.** The checks prove routing and plumbing, not
  quality. After a flip, listen to `ship_artifacts/`: `baseline_en.wav` vs
  `new_en_forced.wav` must be indistinguishable (same weights, same engine — an
  audible difference means misconfiguration), and `new_ru_digits.wav` must speak
  "500" as Russian words. ba/tt/be/ja have never been ear-checked by anyone.
- **The English duration comparison is a WARN, never a rollback.** TTS sampling
  is stochastic — two runs of the same text on the same model differed by 15% in
  testing. It flags "go listen", it does not decide.
- **Go-side work.** `forsen_ai_api` needs no changes: `language` is optional and
  auto-detection is the default, so the client keeps working across the flip
  untouched. The viewer-facing `{ru}`-style universal-TTS filter tag (UNPACK.md
  "API changes", designed in index-finetune `STATE.md` §15-C on the training
  machine) is separate work, and can ship before or after this flip.

## One correction to UNPACK.md

UNPACK's acceptance item 4 says `"?!?!"` must return HTTP 400. That predates the
later decision (commit a92c5db) that punctuation/emoji-only input **synthesizes**
— vocalizing it is a product feature. The smoke suite asserts the current
behaviour: only empty text and an unknown `language` code are 400.

## The BigVGAN CUDA kernel — REQUIRED, and not in git

`load.py` imports a prebuilt
`indextts/BigVGAN/alias_free_activation/cuda/build/anti_alias_activation_cuda.so`
directly. That directory is gitignored, so **a fresh checkout of this branch has
to be given the `.so` by hand** or the server silently runs the torch fallback:

```bash
cp /home/forsen/repos/index-tts-vllm-fast/indextts/BigVGAN/alias_free_activation/cuda/build/anti_alias_activation_cuda.so \
   <checkout>/indextts/BigVGAN/alias_free_activation/cuda/build/
```

`ship.sh preflight` checks for it. `INDEXTTS_BIGVGAN_PREBUILT_SO` overrides the
path. Confirm at boot with `journalctl -u index-tts-multilingual -g "custom CUDA kernel"`
— it must say `Preload`, not `Failed to load`.

**Why it can't just be rebuilt:** `cpp_extension.load()` always re-runs ninja,
and ninja rebuilds whenever the compile command changes — which it does for any
new checkout, since the command embeds absolute source paths. That rebuild
cannot succeed here: g++-15 no longer compiles the installed torch headers
(`ATen/core/List_inl.h:202`, `-Wtemplate-body`; `-fpermissive` and
`-Wno-template-body` both fail), and nvcc 13.3 rejects the default gcc 16.1.1
outright. The `.so` is from 2026-06-10 and is what prod ran for weeks.

**It matters under load, not at idle** — which is why the first pass through
this deployment wrongly called the fallback harmless after measuring an idle
box. Measured `bigvgan_time`, 13 concurrent requests: **1.62s median on the
fallback vs 0.29s with the kernel**. Idle it is ~0.12s either way. Standalone,
the fused kernel is 2.8-3.4x faster per activation with `max|diff|` 1.3e-2 to
2.4e-2 against a signal peaking near 5 (~-46dB) — not bit-identical (filter size
12 is hard-coded, built with `--use_fast_math`), but it is what users have been
hearing all along; the fallback was the deviation.

If a torch upgrade ever breaks the `.so`'s ABI, `load.py` falls back to
compiling and then to torch, so it degrades instead of crashing — but the
`Preload` line is the thing to check after any upgrade.

## Tuning knobs

- `MEL_WORKERS=2 ship.sh` — `run_multilingual.sh` defaults to 4 workers (what
  prod has run since the streaming switch); UNPACK's 14.6GiB measurement was
  taken at 2. Drop to 2 if the card gets tight.
- Removing `--single_engine_lora` and `--lora_peft_dir` from
  `run_multilingual.sh` falls back to two engines: same API, ~+2.25GiB VRAM,
  faster adapter path.
- VRAM headroom at the time of staging: 19.7GiB would be available after the
  swap (5.4GiB free + 14.2GiB released by the old server) against ~15GiB needed.
  `ollama`'s llama-server is the other big tenant at ~24.9GiB.
