#!/usr/bin/env python3
"""Convert an index-tts-training LoRA checkpoint into a PEFT adapter directory
that vLLM can load per-request (`vllm.lora.request.LoRARequest`).

The training checkpoint stores the un-merged adapter as 192 tensors

    gpt.h.<N>.{attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}.lora_{A,B}

with A of shape [r, in] and B of shape [out, r] — exactly the PEFT orientation,
so this is a pure key-rename + fp16 cast, NO transposes.  Verified against the
merged checkpoint: base + (alpha/r) * (B @ A)^T == merged, bit-exact in fp32
(the merge runs on HF Conv1D weights, which are [in, out]; vLLM transposes them
to [out, in] at load time, which is the orientation B @ A already has).

Usage:
    python tools/lora_to_peft.py \
        --ckpt  ../trained_ckpts_lora/model_step12000.pth \
        --out   ../vllm_models/lora_peft_step12000
"""
import argparse
import json
import os
import re

import torch
from safetensors.torch import save_file

# gpt.h.<N>.<module path>.lora_{A,B}
_KEY_RE = re.compile(r"^gpt\.h\.(\d+)\.(.+)\.lora_([AB])$")

TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]


def convert(ckpt_path, out_dir, r=128, lora_alpha=256):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    src = {k: v for k, v in sd.items() if ".lora_" in k}
    assert len(src) == 192, f"expected 192 lora tensors in the ckpt, got {len(src)}"

    out = {}
    for k, v in src.items():
        m = _KEY_RE.match(k)
        assert m, f"unexpected lora key: {k}"
        layer, mod, ab = m.group(1), m.group(2), m.group(3)
        assert mod.split(".")[-1] in TARGET_MODULES, f"unexpected module {mod}"
        new_k = f"base_model.model.transformer.h.{layer}.{mod}.lora_{ab}.weight"
        t = v.detach().to(torch.float16).contiguous()
        # orientation sanity: A is [r, in], B is [out, r]
        if ab == "A":
            assert t.shape[0] == r, f"{k}: lora_A first dim {t.shape[0]} != r={r}"
        else:
            assert t.shape[1] == r, f"{k}: lora_B second dim {t.shape[1]} != r={r}"
        assert new_k not in out
        out[new_k] = t

    assert len(out) == 192, f"expected 192 output tensors, got {len(out)}"
    n_modules = len({k.rsplit(".lora_", 1)[0] for k in out})
    assert n_modules == 96, f"expected 96 wrapped modules, got {n_modules}"

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "adapter_model.safetensors"))

    cfg = {
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": TARGET_MODULES,
        "peft_type": "LORA",
        "bias": "none",
        "use_rslora": False,
        "modules_to_save": None,
        "base_model_name_or_path": "",
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[lora_to_peft] wrote {len(out)} tensors ({n_modules} modules) -> {out_dir}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int, default=128)
    ap.add_argument("--alpha", type=int, default=256)
    ap.add_argument("--verify-merged", default=None,
                    help="optional merged gpt.pth to check base + scale*(B@A)^T == merged")
    ap.add_argument("--verify-base", default=None)
    args = ap.parse_args()

    out = convert(args.ckpt, args.out, r=args.r, lora_alpha=args.alpha)

    if args.verify_merged and args.verify_base:
        def _load(p):
            s = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(s, dict) and "model" in s and isinstance(s["model"], dict):
                s = s["model"]
            return s
        base, merged = _load(args.verify_base), _load(args.verify_merged)
        scale = args.alpha / args.r
        worst = 0.0
        for k in list(out):
            if not k.endswith("lora_A.weight"):
                continue
            stem = k[: -len(".lora_A.weight")]
            hf = stem.replace("base_model.model.transformer.h.", "gpt.h.") + ".weight"
            A = out[stem + ".lora_A.weight"].float()
            B = out[stem + ".lora_B.weight"].float()
            delta = (B @ A).t() * scale
            err = (base[hf].float() + delta - merged[hf].float()).abs().max().item()
            worst = max(worst, err)
        print(f"[lora_to_peft] merge check over 96 modules: max abs err = {worst:.3e}")


if __name__ == "__main__":
    main()
