#!/usr/bin/env python3
"""Reproducible end-to-end benchmark for Index-TTS2 (exact serv2 prod config) on local RTX 5090.

Captures TTFA, total wall time, per-stage timings (via loguru sink), GPT tok/s,
RTF, burst-4 concurrency, VRAM (nvidia-smi + torch), and saves WAVs for quality checks.
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

import soundfile as sf
import torch
from loguru import logger

MODELS_ROOT = "/home/forsen/repos/index-finetune/vllm_models"
ALIGNER_CKPT = "/home/forsen/repos/index-finetune/aligner/ckpt_run8_sub1/best.pt"
REF_WAV = "/home/forsen/repos/index-finetune/eval/refs_prod_wav/forsen.wav"
SR = 22050

CASES = [
    ("en_short", None, "Hello there. This is a speed test of the local engine."),
    ("en_med", None,
     "The quick brown fox jumps over the lazy dog near the river bank, while the stream chat watches in complete disbelief. "
     "Every single word is synthesized on the fly with zero delay and crystal clear pronunciation across the whole broadcast."),
    ("en_long", None,
     "Welcome back everyone to another late night stream where we push our custom neural audio hardware to the absolute limit. "
     "Instead of relying on bloated generic frameworks that waste half the memory bandwidth on Python scheduling overhead, we built a dedicated kernel pipeline. "
     "Every layer runs directly on the GPU inside a single static graph, keeping latency razor sharp even when chat sends massive walls of text. "
     "Hopefully by the time this long message finishes playing, we will see a massive drop in both time to first audio and total generation time."),
    ("ru_short", "ru", "Привет всем! Это быстрый тест скорости озвучки на русском языке."),
    ("ru_med", "ru",
     "Сегодня отличный вечер для стрима, и мы проверяем новую систему синтеза речи прямо здесь. "
     "Звук должен быть чётким, естественным и работать максимально быстро без лишних задержек."),
    ("zh_short", "zh", "今天天气很好，我们一起去公园散步吧。这是第二句话，用来测试速度。"),
]

BURST_CASES = [
    ("en_b0", None, "The quick brown fox jumps over the lazy dog near the river bank today."),
    ("ru_b1", "ru", "Привет чат! Проверяем параллельную нагрузку на четыре потока прямо сейчас."),
    ("en_b2", None, "A calm voice makes the measurement much easier to compare across different runs."),
    ("ru_b3", "ru", "Новая система должна легко справляться с несколькими запросами одновременно."),
]

WARMUP_TEXTS = [
    (None, "Warm up the base English engine and discard this measurement."),
    ("ru", "Прогреваем адаптер для русского языка и отбрасываем это измерение."),
]

_RE_GPT_GEN = re.compile(r"\[gen time:\s*([0-9.]+)\]\s*\[gen len:\s*(\d+)\]\s*\[tok/s:\s*([0-9.]+)\]")
_RE_STAGE = re.compile(r">>\s*(gpt_gen_time|gpt_forward_time|s2mel_time|bigvgan_time|Total inference time|Generated audio length|RTF):\s*([0-9.]+)")


def vram_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        gpu_used = int(out)
    except Exception:
        gpu_used = None
    return {
        "gpu_used_mib": gpu_used,
        "torch_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "torch_max_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


async def run_one(m, name, lang, text, wav_path=None):
    gpt_calls = []
    stages = {}

    def sink(msg):
        s = str(msg)
        mg = _RE_GPT_GEN.search(s)
        if mg:
            gpt_calls.append({
                "gen_s": float(mg.group(1)),
                "gen_len": int(mg.group(2)),
                "tok_s": float(mg.group(3)),
            })
        ms = _RE_STAGE.search(s)
        if ms:
            stages[ms.group(1)] = float(ms.group(2))

    sink_id = logger.add(sink, level="INFO")
    try:
        t0 = time.perf_counter()
        ttfa = None
        segments, chunks = [], []
        async for seg, sr, chunk in m.infer_stream(REF_WAV, text, language=lang):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            segments.append(seg)
            chunks.append(chunk)
        total = time.perf_counter() - t0
    finally:
        logger.remove(sink_id)

    wav = torch.cat(chunks, dim=1)
    audio_s = wav.shape[-1] / SR
    if wav_path:
        pcm = (wav.squeeze(0).clamp(-32767, 32767) / 32768.0).float().numpy()
        sf.write(wav_path, pcm, SR, subtype="PCM_16")

    return {
        "name": name,
        "language": lang or "auto",
        "text": text,
        "ttfa_ms": round(ttfa * 1000, 1),
        "total_ms": round(total * 1000, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(total / max(audio_s, 1e-6), 4),
        "sentences": len(segments),
        "words_count": sum(len(s.get("words", [])) for s in segments),
        "gpt_gen_ms": round(stages.get("gpt_gen_time", 0.0) * 1000, 1),
        "gpt_forward_ms": round(stages.get("gpt_forward_time", 0.0) * 1000, 1),
        "s2mel_ms": round(stages.get("s2mel_time", 0.0) * 1000, 1),
        "bigvgan_ms": round(stages.get("bigvgan_time", 0.0) * 1000, 1),
        "gpt_calls": gpt_calls,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav-dir", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mel-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from indextts.infer_vllm_v2 import IndexTTS2

    before = vram_snapshot()
    t0 = time.perf_counter()
    m = IndexTTS2(
        base_model_dir=f"{MODELS_ROOT}/base_pristine",
        adapter_model_dir=f"{MODELS_ROOT}/lora_step12000",
        lid_model_path=f"{MODELS_ROOT}/glotlid/model.bin",
        single_engine_lora=True,
        lora_peft_dir=f"{MODELS_ROOT}/lora_peft_step12000",
        gpu_memory_utilization=0.10,
        base_kv_cache_memory_bytes=1_000_000_000,
        adapter_kv_cache_memory_bytes=2_500_000_000,
        qwen_emo_mode="disabled",
        ref_cache_size=32,
        mel_workers=args.mel_workers,
        code_aligner_path=ALIGNER_CKPT,
        code_aligner_threads=8,
    )
    load_s = time.perf_counter() - t0
    after_load = vram_snapshot()
    print(f"[{args.label}] loaded in {load_s:.1f}s | gpu_used={after_load['gpu_used_mib']} MiB "
          f"(before={before['gpu_used_mib']} MiB, delta={after_load['gpu_used_mib'] - before['gpu_used_mib']} MiB)", flush=True)

    # Warmup both base and adapter paths (2 rounds to settle torch.compile + CUDA graphs)
    t0 = time.perf_counter()
    for _ in range(2):
        for lang, txt in WARMUP_TEXTS:
            await run_one(m, "warmup", lang, txt)
    warmup_s = time.perf_counter() - t0
    after_warmup = vram_snapshot()
    print(f"[{args.label}] warmup done in {warmup_s:.1f}s | gpu_used={after_warmup['gpu_used_mib']} MiB", flush=True)

    if args.wav_dir:
        os.makedirs(args.wav_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    results = []
    for rep in range(args.reps):
        for name, lang, text in CASES:
            torch.manual_seed(args.seed + rep)
            wav_path = os.path.join(args.wav_dir, f"{args.label}_{name}_r{rep}.wav") if (args.wav_dir and rep == 0) else None
            r = await run_one(m, name, lang, text, wav_path=wav_path)
            r["rep"] = rep
            results.append(r)
            toks = [f"{g['gen_len']}@{g['tok_s']:.0f}" for g in r["gpt_calls"]]
            print(f"[{args.label}] {name:8s} r{rep}: ttfa={r['ttfa_ms']:6.1f}ms total={r['total_ms']:6.1f}ms "
                  f"audio={r['audio_s']:5.2f}s rtf={r['rtf']:.4f} | gpt={r['gpt_gen_ms']:6.1f}ms "
                  f"fwd={r['gpt_forward_ms']:4.1f}ms s2mel={r['s2mel_ms']:6.1f}ms voc={r['bigvgan_ms']:5.1f}ms | tok={toks}", flush=True)

    # Burst-4 concurrency test (2 repetitions)
    burst_runs = []
    for b_rep in range(2):
        torch.manual_seed(args.seed + 500 + b_rep)
        t0 = time.perf_counter()
        b_res = await asyncio.gather(*[
            run_one(m, bname, blang, btext) for bname, blang, btext in BURST_CASES
        ])
        b_wall = (time.perf_counter() - t0) * 1000
        ttfas = [x["ttfa_ms"] for x in b_res]
        totals = [x["total_ms"] for x in b_res]
        burst_runs.append({
            "rep": b_rep,
            "wall_ms": round(b_wall, 1),
            "mean_ttfa_ms": round(statistics.fmean(ttfas), 1),
            "worst_ttfa_ms": round(max(ttfas), 1),
            "ttfas_ms": ttfas,
            "totals_ms": totals,
        })
        print(f"[{args.label}] burst4 r{b_rep}: wall={b_wall:.1f}ms mean_ttfa={statistics.fmean(ttfas):.1f}ms "
              f"worst_ttfa={max(ttfas):.1f}ms ttfas={ttfas}", flush=True)

    after_bench = vram_snapshot()

    # Aggregate per-case means
    by_case = {}
    for name, _, _ in CASES:
        sub = [r for r in results if r["name"] == name]
        all_tok_s = [g["tok_s"] for r in sub for g in r["gpt_calls"]]
        by_case[name] = {
            "ttfa_ms": round(statistics.fmean(r["ttfa_ms"] for r in sub), 1),
            "total_ms": round(statistics.fmean(r["total_ms"] for r in sub), 1),
            "audio_s": round(statistics.fmean(r["audio_s"] for r in sub), 2),
            "rtf": round(statistics.fmean(r["rtf"] for r in sub), 4),
            "gpt_gen_ms": round(statistics.fmean(r["gpt_gen_ms"] for r in sub), 1),
            "gpt_forward_ms": round(statistics.fmean(r["gpt_forward_ms"] for r in sub), 1),
            "s2mel_ms": round(statistics.fmean(r["s2mel_ms"] for r in sub), 1),
            "bigvgan_ms": round(statistics.fmean(r["bigvgan_ms"] for r in sub), 1),
            "gpt_tok_s_mean": round(statistics.fmean(all_tok_s), 1) if all_tok_s else 0.0,
        }

    payload = {
        "label": args.label,
        "gpu": torch.cuda.get_device_name(0),
        "env": {k: os.environ.get(k) for k in ("GPT_ENGINE", "S2MEL_STEPS", "S2MEL_COMPILE", "S2MEL_OPT", "BIGVGAN_CUDA_KERNEL", "BIGVGAN_CUDAGRAPH", "TTS_TF32", "TTS_STREAMS")},
        "load_s": round(load_s, 1),
        "warmup_s": round(warmup_s, 1),
        "vram": {
            "before": before,
            "after_load": after_load,
            "after_warmup": after_warmup,
            "after_bench": after_bench,
            "delta_mib": (after_bench["gpu_used_mib"] - before["gpu_used_mib"]) if (after_bench["gpu_used_mib"] and before["gpu_used_mib"]) else None,
        },
        "by_case": by_case,
        "burst4": {
            "wall_ms": round(statistics.fmean(b["wall_ms"] for b in burst_runs), 1),
            "mean_ttfa_ms": round(statistics.fmean(b["mean_ttfa_ms"] for b in burst_runs), 1),
            "worst_ttfa_ms": round(statistics.fmean(b["worst_ttfa_ms"] for b in burst_runs), 1),
            "runs": burst_runs,
        },
        "calls": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[{args.label}] DONE -> {args.out} | vram_delta={payload['vram']['delta_mib']} MiB", flush=True)
    m.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
