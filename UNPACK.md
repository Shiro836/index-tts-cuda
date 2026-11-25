# UNPACK — multilingual dual-engine TTS handoff (serv2 agent, read this first)

You are on serv2 (prod). This branch (`multilingual-routing`) contains the complete
Python-side implementation of multilingual serving: 16 new languages via a LoRA-merged
adapter model, **English/Chinese still served by the pristine base model, bit-identical to
current prod** (the finetune wrecks English — routing is mandatory, see docs/ROUTING.md).
Everything was built and validated on the training machine (5090). Your job: deploy this
server, and do the Go-side integration in forsen_ai_api. Nothing here retrains anything.

## What is in this branch

- `docs/ROUTING.md` — design + measured rationale. Read before touching code.
- `indextts/lang_router.py` — per-sentence language decision (script gate + GlotLID),
  measured 98.7% on a 21k-row benchmark of short/informal/code-switched text.
- `indextts/text_frontend_serve.py` — per-language serve-time sanitizer (numbers
  verbalized in the right language etc.). Training/serving parity depends on it.
- `indextts/infer_vllm_v2.py`, `indextts/gpt/model_vllm_v2.py`, `api_server_v2.py` —
  dual-engine mode (two vLLM engines + two torch views in one process), `language` API
  param, input guards. Single `--model_dir` mode is untouched back-compat.
- `tests/` — unit tests (`python -m pytest tests/ -x -q` in the serving env).

## Weights (NOT in git) — should already be at `/home/forsen/tts_models/`

The user rsyncs them from the training machine to `/home/forsen/tts_models/`:

| dir | what | size |
|---|---|---|
| `tts_models/base_pristine/` | base engine+model dir (en/zh) | ~11 GB |
| `tts_models/lora_step12000/` | adapter engine+model dir (16 langs) | ~11 GB |
| `tts_models/glotlid/model.bin` | GlotLID LID model | 1.6 GB |
| `tts_models/lora_peft_step12000/` | PEFT-format LoRA adapter (single-engine mode) | 126 MB |

**VERIFY BEFORE USING: no dangling symlinks.** The dirs are symlink farms on the source
machine; a plain `rsync -a` copies broken links (this happened once). Check with
`find /home/forsen/tts_models -xtype l` — must print nothing — and confirm
`du -sh /home/forsen/tts_models` is roughly 24 GB, not 4. If broken, ask the user to
re-run the transfer with `rsync -aL` (dereference).

The two model dirs duplicate several byte-identical components (bigvgan, campplus,
semantic_codec, w2v-bert-2.0, qwen emo); if disk is tight, dedupe with local symlinks
after `md5sum` confirms.

## Python env

Use the existing prod conda env (`index-tts-vllm-fast`). Two extra packages are needed:

```bash
~/miniconda3/envs/index-tts-vllm-fast/bin/pip install fasttext-predict num2words
```

(`fasttext-predict`, NOT `fasttext` — plain fasttext wheels don't build on py>=3.10.)

## Launch (dual mode)

```bash
python api_server_v2.py \
  --base_model_dir    <path>/base_pristine \
  --adapter_model_dir <path>/lora_step12000 \
  --lid_model         <path>/glotlid/model.bin \
  --single_engine_lora --lora_peft_dir <path>/lora_peft_step12000 \
  --gpu_memory_utilization 0.10 \
  --base_kv_cache_memory_bytes    1000000000 \
  --adapter_kv_cache_memory_bytes 2500000000 \
  --qwen_emo_mode lazy --ref_cache_size 32 --mel_workers 4 --port 5113
```

**`--single_engine_lora` is the chosen production mode (user decision):** ONE vLLM engine
holds the pristine base weights; adapter languages attach a per-request LoRA; en/zh run
bit-exact base. The two KV budgets are summed into one 3.5 GB pool (no static split to
tune). Measured cost vs two engines: ~+57 ms/clip on en/zh, ~+120 ms per audio-second on
the 16 adapter languages at low concurrency; saves ~2.25 GiB. Dropping the two LoRA flags
falls back to the two-engine mode — same API, more VRAM, faster adapter path.

**VRAM measured on the 5090, single-engine mode with `--mel_workers 2`: ~14.6 GiB
total** — parent (torch stack + both views) 9.3 GiB + one EngineCore 5.3 GiB (3.5 GB
unified KV). Two-engine fallback measures ~16.9 GiB. **Prod is a 4090 (24 GB): fits with
plenty of headroom, but soak-test before switching the systemd unit.** Boot takes a few
minutes on first run (torch.compile cache — later boots are fast).

Validated on the 5090 before handoff (all pass): all 18 languages route to the right
engine under auto-detect; a mixed EN+RU message splits per-sentence across BOTH engines
with the same voice; the code-switch case ("привет ... macdonalds ... big tasty за 500
рублей") routes ru with digits verbalized as Russian words; 30/30 English harness
sentences fully base-routed (incl. ALL-CAPS shouts and "Mr./Dr." abbreviation traps);
empty/punctuation/emoji/over-length/bad-code requests → HTTP 400; SIGTERM → clean exit
in ~5 s, both EngineCore processes gone, VRAM fully released (verified twice).

## API changes (what the Go side must do)

- `POST /tts_stream` and `/tts_url` accept an OPTIONAL `"language"` field:
  omitted or `"auto"` → per-sentence auto-detection (the default, works for all current
  callers with zero changes); or force one of:
  `en zh ja ru uk be ba tt de fr es it pt pl tr nl cs sv`. Unknown code → 400.
- Per-sentence detected language is returned in each segment/stream chunk.
- The only new 400s: empty text (previously a server-side crash) and an unknown
  `language` code. Punctuation/emoji-only input SYNTHESIZES (vocalization is a product
  feature; routes to the base view when no language is decidable), and there is NO input
  length cap by default (`max_total_chars` knob exists, off — playback truncation at
  TtsLimit stays the Go side's job, unchanged).
- Suggested Go work (design in the index-finetune repo, STATE.md §15-C, on the training
  machine): pass `language` from a viewer-facing `{ru}`-style universal-TTS filter tag
  (precedent: `{old}` → StyleTTS2); NO detection in Go — the server owns it.
- forsen_ai_api on this box has 6 uncommitted dirty files — commit/push them BEFORE
  branching for this work (STATE.md §15-H).

## Acceptance before flipping prod

1. `/health` shows dual mode, both engines up.
2. English request (`"language":"en"` and also auto) → output must be indistinguishable
   from current prod (it is the same weights on the same engine — any audible difference
   means misconfiguration; check you pointed --base_model_dir at base_pristine).
3. Russian request with digits ("возьмем биг тейсти за 500 рублей") → digits spoken as
   Russian words, adapter voice, no runaway length.
4. "?!?!" → HTTP 400, not audio.
5. After `systemctl stop`, `nvidia-smi` shows NO leftover EngineCore processes (there are
   TWO now; orphans hold ~15 GB and the next start OOMs).
6. ZH tripwire: a Chinese sentence through auto → routes base, sounds like current prod.

## Language quality expectations (measured, WER on held-out smokes)

it .017 · de .033 · ru .033 · cs .040 · nl .063 · pl .068 · pt .083 · tr .096 · fr .108 ·
es .112 · sv .144 · uk .154 · be .376 · ja .559 · tt .817 · ba 1.057.
**Caveat on the tail numbers: these are whisper-WER, and whisper is itself unreliable on
ba/tt (its language-ID scores 0% there even on correct audio), so ba/tt/be/ja may sound
much better than they score. Nobody has ear-checked them yet — the user's listening is
the authority in this project. Make no UI decisions about them without it.**
en/zh = base model = unchanged from prod.
