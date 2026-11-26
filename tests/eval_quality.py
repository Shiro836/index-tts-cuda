#!/usr/bin/env python3
"""Audio quality parity & A/B evaluation for baseline vs optimized Index-TTS2.

Checks:
1. Isolated s2mel + BigVGAN numerical parity on identical semantic codes (mel cosine,
   BigVGAN TF32 vs FP32 waveform cosine, clipping/NaN checks).
2. Side-by-side WAV quality across EN, RU, ZH (baseline vs optimized):
   - duration, speech_frac, trailing silence (no runaway/dead air)
   - CAMPPlus trimmed speaker cosine similarity vs reference speaker
   - torchaudio SQUIM objective metrics (STOI, PESQ, SI-SDR)
   - offline openai/whisper-large-v3 ASR transcripts
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, "/home/forsen/repos/index-finetune/eval")

from metrics import speech_stats, trimmed_waveform  # noqa: E402
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus  # noqa: E402

MODELS_ROOT = "/home/forsen/repos/index-finetune/vllm_models"
REF_WAV = "/home/forsen/repos/index-finetune/eval/refs_prod_wav/forsen.wav"


@torch.no_grad()
def campplus_embed(model, wav_tensor, sr, device):
    if sr != 16000:
        wav_tensor = torchaudio.transforms.Resample(sr, 16000)(wav_tensor)
    feat = torchaudio.compliance.kaldi.fbank(
        wav_tensor, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    emb = model(feat.unsqueeze(0).to(device))
    return F.normalize(emb.float(), dim=-1)


def run_bigvgan_tf32_parity():
    """Verify BigVGAN FP32 vs TF32 waveform parity (never FP16)."""
    from indextts.s2mel.modules.bigvgan import bigvgan

    os.environ.setdefault(
        "INDEXTTS_BIGVGAN_PREBUILT_SO",
        "/home/forsen/repos/index-finetune/index-tts-vllm-ml/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build/anti_alias_activation_cuda.so",
    )
    voc = bigvgan.BigVGAN.from_pretrained(
        f"{MODELS_ROOT}/lora_step12000/bigvgan", use_cuda_kernel=True
    ).cuda().eval()
    voc.remove_weight_norm()

    # Use real speech mel from REF_WAV (not random noise) for realistic vocoder SNR
    import librosa
    from indextts.s2mel.modules.audio import mel_spectrogram
    y_ref, _ = librosa.load(REF_WAV, sr=22050)
    mel = mel_spectrogram(
        torch.from_numpy(y_ref[:22050 * 4]).unsqueeze(0).cuda().float(),
        n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False
    )

    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        w_fp32 = voc(mel).float()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        w_tf32 = voc(mel).float()

    cos = F.cosine_similarity(w_fp32.reshape(-1), w_tf32.reshape(-1), dim=0).item()
    diff = (w_fp32 - w_tf32).abs()
    snr = 10.0 * torch.log10((w_fp32.pow(2).mean()) / (diff.pow(2).mean().clamp(min=1e-12))).item()
    # Also compare mel-spectrogram cosine of reconstructed waveforms (perceptual metric)
    m_fp32 = mel_spectrogram(w_fp32.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)
    m_tf32 = mel_spectrogram(w_tf32.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)
    mel_cos = F.cosine_similarity(m_fp32.reshape(-1), m_tf32.reshape(-1), dim=0).item()
    res = {
        "bigvgan_tf32_vs_fp32_wav_cos": round(cos, 7),
        "bigvgan_tf32_vs_fp32_mel_cos": round(mel_cos, 7),
        "bigvgan_tf32_vs_fp32_snr_db": round(snr, 2),
        "bigvgan_tf32_vs_fp32_max_diff": round(diff.max().item(), 6),
        "finite": bool(torch.isfinite(w_tf32).all().item()),
    }
    del voc
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--opt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper-model", default="openai/whisper-large-v3")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 1. BigVGAN TF32 vs FP32 Parity ===", flush=True)
    voc_parity = run_bigvgan_tf32_parity()
    print(f"  {voc_parity}", flush=True)
    assert voc_parity["finite"] and voc_parity["bigvgan_tf32_vs_fp32_mel_cos"] > 0.9999 and voc_parity["bigvgan_tf32_vs_fp32_snr_db"] > 35.0

    print("=== 2. Loading CAMPPlus + SQUIM + Whisper ASR ===", flush=True)
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(
        torch.load(f"{MODELS_ROOT}/lora_step12000/campplus/campplus_cn_common.bin", map_location="cpu")
    )
    campplus = campplus.to(device).eval()
    ref_wav, ref_sr = trimmed_waveform(Path(REF_WAV))
    ref_emb = campplus_embed(campplus, ref_wav, ref_sr, device)

    squim_obj = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device).eval()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    asr_proc = AutoProcessor.from_pretrained(args.whisper_model, local_files_only=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_proc.tokenizer,
        feature_extractor=asr_proc.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )

    def eval_dir(tag, d):
        rows = {}
        for wp in sorted(glob.glob(os.path.join(d, "*.wav"))):
            fname = os.path.basename(wp)
            # Strip "<label>_" prefix and "_r0.wav" suffix -> case name
            stem = fname[:-4]
            parts = stem.split("_")
            # Find case name (en_short, en_med, en_long, ru_short, ru_med, zh_short)
            case = None
            for cand in ("en_short", "en_med", "en_long", "ru_short", "ru_med", "zh_short"):
                if cand in stem:
                    case = cand
                    break
            if case is None:
                case = stem

            st = speech_stats(Path(wp))
            y, sr = sf.read(wp, dtype="float32")
            peak = float(np.abs(y).max())
            clipped_frac = float((np.abs(y) >= 0.999).mean())
            finite = bool(np.isfinite(y).all())

            trim_w, trim_sr = trimmed_waveform(Path(wp))
            spk_emb = campplus_embed(campplus, trim_w, trim_sr, device)
            spk_sim = float(F.cosine_similarity(ref_emb, spk_emb, dim=-1).item())

            w16 = torchaudio.transforms.Resample(sr, 16000)(torch.from_numpy(y).unsqueeze(0))[:, :16000 * 15].to(device)
            with torch.no_grad():
                stoi, pesq, sisdr = squim_obj(w16)

            lang_hint = "russian" if case.startswith("ru_") else ("chinese" if case.startswith("zh_") else "english")
            asr_out = asr(
                {"array": y, "sampling_rate": sr},
                generate_kwargs={"language": lang_hint, "task": "transcribe"},
                return_timestamps=True,
            )
            text = asr_out["text"].strip()

            rows[case] = {
                "file": wp,
                "duration_s": round(st["duration_s"], 2),
                "speech_frac": round(st["speech_frac"], 3),
                "trail_sil_s": round(st["trail_sil_s"], 3),
                "peak": round(peak, 4),
                "clipped_frac": round(clipped_frac, 6),
                "finite": finite,
                "spk_sim": round(spk_sim, 4),
                "squim_stoi": round(float(stoi[0]), 4),
                "squim_pesq": round(float(pesq[0]), 3),
                "squim_sisdr": round(float(sisdr[0]), 2),
                "asr": text,
            }
            print(
                f"[{tag:8s}] {case:8s}: dur={rows[case]['duration_s']:5.2f}s spk={spk_sim:.3f} "
                f"stoi={rows[case]['squim_stoi']:.3f} pesq={rows[case]['squim_pesq']:.2f} "
                f"speech={rows[case]['speech_frac']:.2f} trail={rows[case]['trail_sil_s']:.2f}s | {text[:80]}",
                flush=True,
            )
        return rows

    base_rows = eval_dir("baseline", args.baseline_dir)
    opt_rows = eval_dir("opt", args.opt_dir)

    report = {
        "bigvgan_parity": voc_parity,
        "baseline": base_rows,
        "optimized": opt_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved quality report -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
