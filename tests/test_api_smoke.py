#!/usr/bin/env python3
"""End-to-end HTTP smoke test for api_server_v2.py (/health, /tts_url, /tts_stream)."""
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import soundfile as sf

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_ROOT = "/home/forsen/repos/index-finetune/vllm_models"
ALIGNER_CKPT = "/home/forsen/repos/index-finetune/aligner/ckpt_run8_sub1/best.pt"
REF_WAV = "/home/forsen/repos/index-finetune/eval/refs_prod_wav/forsen.wav"
PORT = 16099
BASE_URL = f"http://127.0.0.1:{PORT}"


def http_get(path):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def http_post(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def main():
    env = os.environ.copy()
    env.setdefault("GPT_ENGINE", "cuda")
    env.setdefault("TTS_TF32", "1")
    env.setdefault("S2MEL_OPT", "1")
    env.setdefault("S2MEL_COMPILE", "1")
    env.setdefault("TTS_STREAMS", "1")
    env.setdefault("S2MEL_STEPS", "12")
    env.setdefault("BIGVGAN_CUDA_KERNEL", "1")
    env.setdefault(
        "INDEXTTS_BIGVGAN_PREBUILT_SO",
        "/home/forsen/repos/index-finetune/index-tts-vllm-ml/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build/anti_alias_activation_cuda.so",
    )

    cmd = [
        sys.executable,
        os.path.join(REPO_DIR, "api_server_v2.py"),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--base_model_dir", f"{MODELS_ROOT}/base_pristine",
        "--adapter_model_dir", f"{MODELS_ROOT}/lora_step12000",
        "--lid_model", f"{MODELS_ROOT}/glotlid/model.bin",
        "--single_engine_lora",
        "--lora_peft_dir", f"{MODELS_ROOT}/lora_peft_step12000",
        "--qwen_emo_mode", "disabled",
        "--ref_cache_size", "32",
        "--mel_workers", "4",
        "--code_aligner", ALIGNER_CKPT,
        "--code_aligner_threads", "8",
    ]
    proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # Wait for /health
        t0 = time.perf_counter()
        healthy = False
        while time.perf_counter() - t0 < 90:
            if proc.poll() is not None:
                out = proc.stdout.read()
                raise RuntimeError(f"Server exited early ({proc.returncode}):\n{out[-2000:]}")
            try:
                st, data = http_get("/health")
                if st == 200 and data.get("status") == "healthy":
                    print(f"[SMOKE] /health ready in {time.perf_counter() - t0:.1f}s: {json.dumps(data)}")
                    assert data["mode"] == "dual"
                    assert data["gpt_engine"] == env["GPT_ENGINE"]
                    assert data["engines"]["base"]["running"] and data["engines"]["adapter"]["running"]
                    healthy = True
                    break
            except Exception:
                time.sleep(0.5)
        assert healthy, "Server did not become healthy within 90s"

        # 1. /tts_url EN with return_segments=True
        st, raw = http_post("/tts_url", {
            "spk_audio_path": REF_WAV,
            "text": "Hello chat! Testing the custom CUDA engine API.",
            "return_segments": True,
        })
        assert st == 200
        resp = json.loads(raw.decode("utf-8"))
        wav_bytes = base64.b64decode(resp["audio"])
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        words = resp["segments"][0].get("words", [])
        print(f"[SMOKE] /tts_url EN: sr={sr} dur={len(audio)/sr:.2f}s lang={resp['segments'][0]['language']} words={len(words)}")
        assert sr == 22050 and len(audio) > 10000 and len(words) == 8

        # 2. /tts_stream RU with word timings
        st, raw = http_post("/tts_stream", {
            "spk_audio_path": REF_WAV,
            "text": "Привет чат! Проверяем потоковый вывод на русском языке.",
            "language": "ru",
        })
        assert st == 200
        lines = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        assert lines[-1].get("done") is True
        chunk_lines = lines[:-1]
        assert len(chunk_lines) >= 1
        c0 = chunk_lines[0]
        audio_ru, sr_ru = sf.read(io.BytesIO(base64.b64decode(c0["audio"])))
        print(f"[SMOKE] /tts_stream RU: chunks={len(chunk_lines)} sr={sr_ru} dur={len(audio_ru)/sr_ru:.2f}s lang={c0['language']} words={len(c0.get('words', []))}")
        assert sr_ru == 22050 and len(audio_ru) > 10000 and c0["language"] == "ru" and len(c0.get("words", [])) > 0

        print("ALL API SMOKE TESTS PASSED!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
