#!/usr/bin/env python
"""Pre/post-flip checks for ship.sh (UNPACK.md "Acceptance before flipping prod").

Two subcommands:

  baseline  — run against the CURRENTLY LIVE (old) server before the flip and
              record what English sounds like today: duration, sample rate, wav.
  smoke     — run against the new multilingual server after the flip: health,
              routing, guards, streaming, and English parity vs the baseline.

`smoke` exits non-zero if any HARD check fails; ship.sh rolls back on that.
Soft checks (English duration parity) only warn: TTS sampling is stochastic, so
a duration delta is a hint to go listen, not proof of a broken deploy. Both
subcommands write the wavs they produced into --artifacts so the deltas can be
ear-checked — in this project the user's ears are the authority.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import soundfile as sf

SPK = "/home/forsen/repos/index-tts-ml/examples/voice_01.wav"

EN_TEXT = ("hello chat, this is the english parity check, "
           "the voice should sound exactly like it does today.")
ZH_TEXT = "大家好，这是一个中文测试句子。"
RU_TEXT = "возьмем биг тейсти за 500 рублей"
# Both halves need >= 3 alphabetic words. _plan_sentences (infer_vllm_v2.py:829)
# decides long sentences first, then makes SHORTER fragments inherit the message
# language instead of self-detecting — so "hello chat." (2 words) next to a
# Russian sentence is routed ru BY DESIGN (commit da4bfe4: short fragments are
# where detection fails). MIX_SHORT_TEXT below checks that rule on purpose.
MIX_TEXT = "hello chat how are you doing today. привет, как дела сегодня?"
MIX_SHORT_TEXT = "hello chat. привет, как дела сегодня?"

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def post(url, payload, timeout=180):
    """POST json, return (status, body_bytes, content_type)."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as ex:
        return ex.code, ex.read(), ex.headers.get("Content-Type", "")


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as ex:
        return ex.code, ex.read()


def wav_info(raw):
    """(duration_seconds, sample_rate, peak_amplitude) of in-memory wav bytes."""
    data, sr = sf.read(io.BytesIO(raw))
    peak = float(abs(data).max()) if len(data) else 0.0
    return len(data) / sr, sr, peak


def synth(base, text, artifacts, name, language=None, segments=True, timeout=180):
    """POST /tts_url. Returns (status, wav_bytes_or_None, segments_or_None)."""
    payload = {"text": text, "spk_audio_path": SPK}
    if language is not None:
        payload["language"] = language
    if segments:
        payload["return_segments"] = True

    status, body, ctype = post(base + "/tts_url", payload, timeout=timeout)
    if status != 200:
        return status, None, None

    if "json" in ctype:
        doc = json.loads(body)
        wav = base64.b64decode(doc["audio"])
        segs = doc.get("segments")
    else:
        wav, segs = body, None

    if artifacts:
        with open(os.path.join(artifacts, name + ".wav"), "wb") as fh:
            fh.write(wav)
    return status, wav, segs


class Checks:
    def __init__(self):
        self.hard_failures = []
        self.warnings = []

    def hard(self, name, ok, detail=""):
        print(f"  [{GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}] {name}"
              + (f" — {detail}" if detail else ""))
        if not ok:
            self.hard_failures.append(f"{name}: {detail}")
        return ok

    def soft(self, name, ok, detail=""):
        print(f"  [{GREEN + 'PASS' + RESET if ok else YELLOW + 'WARN' + RESET}] {name}"
              + (f" — {detail}" if detail else ""))
        if not ok:
            self.warnings.append(f"{name}: {detail}")
        return ok


def cmd_baseline(args):
    """Record today's English output from the live old server."""
    os.makedirs(args.artifacts, exist_ok=True)
    print("Capturing English baseline from the live server "
          f"({args.url}) — this is what prod sounds like right now")

    status, wav, _ = synth(args.url, EN_TEXT, args.artifacts, "baseline_en",
                           language=None, segments=False)
    if status != 200 or not wav:
        print(f"{YELLOW}WARN{RESET}: baseline capture failed (HTTP {status}). "
              "The flip can still proceed; English parity just won't be compared.")
        return 0

    dur, sr, peak = wav_info(wav)
    doc = {"text": EN_TEXT, "duration": dur, "sampling_rate": sr, "peak": peak,
           "captured_at": time.time()}
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  baseline: {dur:.2f}s @ {sr}Hz, peak {peak:.3f} "
          f"-> {os.path.join(args.artifacts, 'baseline_en.wav')}")
    return 0


def cmd_smoke(args):
    os.makedirs(args.artifacts, exist_ok=True)
    c = Checks()

    print("\nAcceptance checks (UNPACK.md) against " + args.url)

    # 1. health: dual mode, every engine up and not errored
    status, body = get(args.url + "/health")
    health = json.loads(body) if body else {}
    engines = health.get("engines", {})
    c.hard("health 200 + healthy", status == 200 and health.get("status") == "healthy",
           f"HTTP {status} {health.get('status')}")
    c.hard("routing mode is dual", health.get("mode") == "dual",
           f"mode={health.get('mode')} (single means routing is OFF — en/zh would "
           "run on the adapter)")
    c.hard("all engines running, none errored",
           bool(engines) and all(e.get("running") and not e.get("errored")
                                 for e in engines.values()),
           json.dumps(engines))
    print(f"       views: {', '.join(sorted(engines))}")

    # 2. English forced — the critical parity path (same weights as prod today)
    status, wav, segs = synth(args.url, EN_TEXT, args.artifacts, "new_en_forced",
                              language="en")
    ok = c.hard("EN forced -> 200 + audio", status == 200 and bool(wav), f"HTTP {status}")
    if ok:
        dur, sr, peak = wav_info(wav)
        c.hard("EN forced audio is not silent", peak > 0.01, f"peak {peak:.4f}")
        if args.baseline and os.path.exists(args.baseline):
            base_dur = json.load(open(args.baseline))["duration"]
            ratio = dur / base_dur if base_dur else 0
            c.soft("EN duration within 35% of pre-flip baseline",
                   0.65 <= ratio <= 1.35,
                   f"{dur:.2f}s vs {base_dur:.2f}s baseline ({ratio:.2f}x) — "
                   "sampling is stochastic, so compare by EAR: "
                   f"{args.artifacts}/baseline_en.wav vs new_en_forced.wav")
        else:
            c.soft("EN parity baseline available", False, "no baseline captured")

    # 3. English auto-detect must route to the base view, not the adapter
    status, wav, segs = synth(args.url, EN_TEXT, args.artifacts, "new_en_auto")
    ok = c.hard("EN auto -> 200 + audio", status == 200 and bool(wav), f"HTTP {status}")
    if ok and segs is not None:
        langs = {s.get("language") for s in segs}
        c.hard("EN auto routes every sentence to en", langs <= {"en"}, f"detected {langs}")

    # 4. ZH tripwire — also base view
    status, wav, segs = synth(args.url, ZH_TEXT, args.artifacts, "new_zh_auto")
    ok = c.hard("ZH auto -> 200 + audio", status == 200 and bool(wav), f"HTTP {status}")
    if ok and segs is not None:
        langs = {s.get("language") for s in segs}
        c.hard("ZH auto routes to zh", langs <= {"zh"}, f"detected {langs}")

    # 5. Russian with digits — the adapter path + the per-language sanitizer
    status, wav, segs = synth(args.url, RU_TEXT, args.artifacts, "new_ru_digits")
    ok = c.hard("RU digits -> 200 + audio", status == 200 and bool(wav), f"HTTP {status}")
    if ok:
        dur, sr, peak = wav_info(wav)
        c.hard("RU audio is not silent", peak > 0.01, f"peak {peak:.4f}")
        c.hard("RU length is sane (no runaway)", 1.0 < dur < 20.0, f"{dur:.2f}s")
        if segs is not None:
            langs = {s.get("language") for s in segs}
            c.hard("RU routes to ru", langs <= {"ru"}, f"detected {langs}")
        print(f"{YELLOW}       EAR-CHECK{RESET}: '500' must be spoken as Russian words "
              f"-> {args.artifacts}/new_ru_digits.wav")

    # 6. Mixed message splits per sentence across both views
    status, wav, segs = synth(args.url, MIX_TEXT, args.artifacts, "new_mixed")
    ok = c.hard("EN+RU mixed -> 200 + audio", status == 200 and bool(wav), f"HTTP {status}")
    if ok and segs is not None:
        langs = {s.get("language") for s in segs}
        c.hard("mixed message splits across both views", {"en", "ru"} <= langs,
               f"detected {langs}")

    # 6b. The short-fragment inherit rule, asserted rather than tripped over: a
    #     2-word English fragment in a Russian message rides ru. Soft, because it
    #     is a quality judgement (an EN shard gets a Russian accent), not breakage.
    status, wav, segs = synth(args.url, MIX_SHORT_TEXT, args.artifacts, "new_mixed_short")
    if status == 200 and segs is not None:
        langs = {s.get("language") for s in segs}
        c.soft("short EN fragment inherits the message language (by design)",
               langs == {"ru"},
               f"detected {langs} — expected only ru; the two-pass rule in "
               "_plan_sentences may have changed")

    # 7. Guards. Punctuation-only SYNTHESIZES on purpose (vocalization is a product
    #    feature — UNPACK's acceptance item 4 predates that decision, commit a92c5db);
    #    only empty text and an unknown language code are 400.
    status, _, _ = synth(args.url, "", args.artifacts, "guard_empty", segments=False)
    c.hard("empty text -> 400", status == 400, f"HTTP {status}")

    status, _, _ = synth(args.url, EN_TEXT, args.artifacts, "guard_lang",
                         language="xx", segments=False)
    c.hard("unknown language code -> 400", status == 400, f"HTTP {status}")

    status, wav, _ = synth(args.url, "?!?!", args.artifacts, "guard_punct", segments=False)
    c.hard("punctuation-only -> 200 (vocalization is a feature)",
           status == 200 and bool(wav), f"HTTP {status}")

    # 8. /tts_stream — the only path forsen_ai_api actually uses in prod
    payload = {"text": EN_TEXT, "spk_audio_path": SPK}
    req = urllib.request.Request(args.url + "/tts_stream",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks, done, err = 0, False, None
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                if doc.get("done"):
                    done = True
                elif doc.get("error"):
                    err = doc["error"]
                elif doc.get("audio"):
                    chunks += 1
    except Exception as ex:  # noqa: BLE001 - any stream failure is a hard fail
        err = repr(ex)
    c.hard("/tts_stream yields chunks then done:true (the Go client's path)",
           chunks > 0 and done and not err,
           f"{chunks} chunks, done={done}, error={err}")

    print()
    for w in c.warnings:
        print(f"{YELLOW}WARNING{RESET}: {w}")
    if c.hard_failures:
        print(f"\n{RED}{len(c.hard_failures)} hard check(s) failed{RESET}:")
        for f in c.hard_failures:
            print(f"  - {f}")
        return 1
    print(f"{GREEN}All hard checks passed.{RESET} Audio for ear-checking: {args.artifacts}/")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="record the live server's English output")
    b.add_argument("--url", default="http://127.0.0.1:5113")
    b.add_argument("--out", required=True)
    b.add_argument("--artifacts", required=True)
    b.set_defaults(func=cmd_baseline)

    s = sub.add_parser("smoke", help="acceptance checks against the new server")
    s.add_argument("--url", default="http://127.0.0.1:5113")
    s.add_argument("--baseline", default=None)
    s.add_argument("--artifacts", required=True)
    s.set_defaults(func=cmd_smoke)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
