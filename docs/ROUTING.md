# Multilingual dual-engine routing — implementation spec

Goal: serve 18 languages from one process. **en/zh route to the PRISTINE base model**
(bit-identical to today's production — the finetune gave English a British accent and the
adapter path makes English unusable), the 16 new languages (ru uk be ba tt de fr es it pt
pl tr nl cs sv ja) route to the LoRA-merged adapter model. Routing is per SENTENCE, decided
by language: forced via API, else auto-detected.

Everything here was measured/audited first; sources: STATE.md §15 in the index-finetune
repo (sanity audit), a 21k-row LID benchmark (`eval/lid/` there), and a code audit of this
fork. Do not re-litigate design choices marked MEASURED.

## 1. The two views

| view | languages | vLLM engine dir | torch UnifiedVoice ckpt | tokenizer | number_text_tokens |
|---|---|---|---|---|---|
| base | en, zh | `<base_dir>/gpt` | `<base_dir>/gpt.pth` (12001 rows) | `<base_dir>/bpe.model` (= original) | 12000 |
| adapter | 16 new | `<adapter_dir>/gpt` | `<adapter_dir>/gpt.pth` (16239 rows, LoRA-merged) | `<adapter_dir>/bpe.model` (= frozen extended) | 16238 |

Model dirs: `vllm_models/base_pristine/` and `vllm_models/lora_step12000/` (built,
validated individually — see UNPACK.md for how to obtain them).

**Why per-view tokenizer + text embedding is NOT optional (MEASURED):** on 10,092 real EN
sentences, 46% tokenize differently on the extended tokenizer and 34% contain ids >= 12001
(the extension added pieces like `▁S`). The base view must use the ORIGINAL tokenizer and
the base 12001-row embedding, or English is no longer pristine.

**Why the torch-side model must ALSO be routed (MEASURED):** `_sentence_to_wav`
(`infer_vllm_v2.py` ~:469-480) runs the torch UnifiedVoice forward (incl. the 24-layer GPT2
body — exactly the LoRA-modified modules) to produce the s2mel latent. One shared torch
model would give EN base mel codes but adapter-weighted latents.

### Memory: share everything bit-identical

base vs merged `gpt.pth` differ in EXACTLY: the 96 `gpt.h.*` Conv1D tensors +
`text_embedding.weight` + `text_head.{weight,bias}` (verified tensor-by-tensor; 566/665
bit-identical, base-row prefixes identical). Therefore:

- Build the ADAPTER UnifiedVoice fully (it is the busy path).
- Build the BASE UnifiedVoice, then REPLACE every submodule EXCEPT `gpt`,
  `text_embedding`, `text_head` with a reference to the adapter view's submodule
  (conditioning encoder, perceiver, mel/emo embeddings, final_norm, speed_emb, etc. are
  bit-identical — sharing is free and keeps the ref cache valid for both).
- Extra VRAM ≈ base `gpt` body + base emb/head only (~1.6 GB fp32, ~0.8 GB fp16).
- Do NOT swap module attributes per request — `mel_workers>1` runs `_sentence_to_wav`
  concurrently; pass the view object through the call chain instead.
- `self.gpt.llm` and `self.sampling_params` become per-view.
- The ref cache (`_get_ref`) stays SINGLE and shared: its entries are computed by
  submodules that are bit-identical between views. Keep `get_conditioning` pinned to the
  adapter view's (shared) conditioning encoder.

### Engines

- Two `AsyncLLM` instances in one process is exercised code (QwenEmotion already builds a
  second engine). vLLM 0.22 V1 spawns one EngineCore process per engine, own ZMQ port.
- **Build BOTH engines before any parent-process CUDA init** (else `_maybe_force_spawn`
  switches to spawn — works, slow boot). Don't build engines while QwenEmotion mutates
  `CUDA_VISIBLE_DEVICES`.
- Memory: `gpu_memory_utilization` is a fraction of TOTAL vram checked against
  free-at-boot; keep the prod recipe — low util gate + explicit `kv_cache_memory_bytes`
  PER ENGINE (new CLI args; adapter engine gets the bigger budget). Boot sequentially.
- Pass `disable_log_stats=True` to the SECOND engine (each new Prometheus stat logger
  unregisters every existing vllm metric collector).
- Call `AsyncLLM.shutdown()` on BOTH in the FastAPI lifespan teardown (today nothing calls
  it; orphan EngineCore processes hold ~15 GB after parent death).

## 2. Language decision (per sentence)

Implemented in `indextts/lang_router.py`. Chosen by measurement — pipeline totals on the
21k-row benchmark: **GlotLID v3 98.70%** (99.4% informal, 96.9% on 3-word fragments,
99.3% code-switched) > NLLB-lid218e 98.36% > OpenLID 97.56% > lid.176 96.50% > lingua
91.15% (no ba/tt). GlotLID model: `cis-lmu/glotlid` `model.bin` (1.6 GB, CPU, ~0.4 ms,
Apache-2.0), loaded with `fasttext-predict` (plain fasttext wheels don't build on
py>=3.10).

Order:
1. **Forced code wins** (API `language` field; Go side may pass tags like `{ru}`).
2. Any kana -> `ja`. Han without kana -> GlotLID restricted to {ja, zh} (pure-kanji
   Japanese exists; a hard han->zh rule mislabeled 108/1200 ja benchmark rows).
3. UNIQUE Cyrillic markers are hard overrides: any of `ҡ ғ ҙ ҫ` -> ba, `җ` -> tt,
   `ў` -> be, `ї є ґ` -> uk. Shared markers (`ә ө ү ң һ`, `і`) must NOT decide
   (hard-deciding them misrouted 865/1200 tt rows).
4. Any remaining Cyrillic -> GlotLID restricted to {ru, uk, be, ba, tt}.
5. Pure Latin -> GlotLID restricted to {en, de, fr, es, it, pt, pl, tr, nl, cs, sv-SE}.
   Low confidence -> the message-level dominant language if another sentence decided one,
   else `en`.
6. **Asymmetry rule — bias every tie toward base/en:** the base tokenizer has ZERO
   Cyrillic, so new-script text misrouted to base is `<unk>` garbage, while Latin text
   misrouted to base merely renders with base-EN pronunciation.
7. GlotLID label mapping is FLORES-style (`__label__rus_Cyrl` etc.); restrict by argmax
   over the subset of our 18 labels from `predict(k=-1)` (this is what the 98.70% was
   measured with). `zxx_*`/`und_*` top-label = non-language input -> treat as junk (guard).

Routing: `ja` and the 15 others -> adapter view; `en`, `zh` -> base view.

## 3. Serve-time text front-end (ship blocker on its own)

Today the fork normalizes with `TextNormalizer` auto-routing: any text with no ASCII
letters goes down the CHINESE path — digits become Chinese numerals, hyphens become 减.
Training text was instead sanitized per-language (numbers verbalized in their own
language via num2words) and tokenized with an explicit language hint that lands in the
inert `_basic_cleanup` path. Serving must do the same or every digit/hyphen diverges from
training.

`indextts/text_frontend_serve.py` (ported from index-finetune
`data_prep/text_frontend.py`, same repo as STATE.md):
- `sanitize(text, lang)` — NFKC, per-language num2words, currency/percent expansion,
  decimal words, typographic-quote folding, Cyrillic stress-mark stripping,
  confusable folding inside Cyrillic words.
- Serve-time relaxations vs the training version: cross-script words PASS THROUGH
  (viewers code-switch; training-time rejection stays in data_prep), unknown chars are
  dropped/passed, never raise.
- ba/tt digits verbalize via Russian num2words (no ba/tt support in num2words; both
  communities read Russian numerals).
- en/zh/ja: passthrough (`sanitize` returns text unchanged) — their native normalizer
  routes are the base model's own and must not change.
- After sanitize, tokenize with the VIEW's tokenizer passing the explicit language hint
  so `TextNormalizer` takes `_basic_cleanup` for the 15 non-native languages.

## 4. API

`/tts_stream` parses via `_parse_tts_params`; `/tts_url` re-parses INLINE — patch BOTH:
- `language`: optional string. Omitted/`auto` -> per-sentence auto-detect. A code from
  {en zh ja ru uk be ba tt de fr es it pt pl tr nl cs sv} -> forced for ALL sentences.
  Unknown code -> 400.
- Response segments gain a `language` field per sentence (useful for the Go side + debug).
- Back-compat: no `language` field = auto = current callers keep working.

New CLI args (`api_server_v2.py`): `--base_model_dir`, `--adapter_model_dir` (if only
`--model_dir` is given, run single-engine exactly as today — back-compat mode),
`--lid_model` (GlotLID path; required for dual mode), `--base_kv_cache_memory_bytes`,
`--adapter_kv_cache_memory_bytes`.

## 5. Input guards (server-side)

1. Empty/whitespace text -> 400 (today it reaches `torch.cat([])` and crashes the
   request). This is a crash fix, nothing more.
2. **Punctuation/emoji-only input is NOT an error — it synthesizes.** Vocalizing "?!?!"
   is a product feature (the webui ships a punctuation stress-test button; user decision
   2026-07-31). Wordless sentences ride the forced/dominant language, else the base view.
3. `max_total_chars` exists as an optional knob but is OFF by default (user decision
   2026-07-31): previous behavior was no input cap anywhere — a long LLM reply
   synthesizes and the Go player cuts playback at the broadcaster's TtsLimit (80 s
   default), with streaming disconnect-cancel bounding the leftover GPU work. A 400
   would replace "viewer hears the first 80 s" with "viewer hears nothing".

## 6. Acceptance gates (run before calling it done)

1. **EN/ZH equivalence:** same text+ref+seed through (a) the dual server routed to base
   and (b) a single-engine server on `base_pristine` -> identical mel token streams /
   waveforms within sampling determinism. Any difference = a leak in the view separation.
2. **16-language smokes** through the router match the direct-adapter numbers
   (index-finetune `eval/` harness, §14-C sticks).
3. **Code-switch smoke**: RU+Latin-brand sentences route to adapter, digits verbalized
   Russian, no crash, audible output sane.
4. Guards: empty -> 400, "?!?!" -> 400/silence, 3000-char input -> 400.
5. VRAM fits with both engines + torch stack; measure and record for the 24 GB prod card.
6. Shutdown kills BOTH EngineCore processes (check `nvidia-smi` after SIGTERM).
