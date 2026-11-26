export CUDA_DEVICE_ORDER=PCI_BUS_ID
# RTX 4090 = index 0 in PCI bus order (bus 01). vLLM int()-parses CUDA_VISIBLE_DEVICES,
# so a UUID crashes it; the 4060 Ti is for singing + mining
export CUDA_VISIBLE_DEVICES=0

# CUDA 13.x nvcc cannot parse gcc-16 libstdc++ headers; force JIT compiles
# (flashinfer, BigVGAN kernel) to use gcc-15 as host compiler.
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-15'

# flashinfer's get_cuda_path() only checks $CUDA_HOME then /usr/local/cuda,
# but on Arch CUDA lives in /opt/cuda.
export CUDA_HOME=/opt/cuda
export PATH=/opt/cuda/bin:$PATH

# reduce caching-allocator fragmentation in the multi-model main process
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS=/home/forsen/tts_models

# 2026-11-04 speedups (research/speedup_2026-11-03): compiled DiT, fused
# BigVGAN kernel; 18-step s2mel with custom TF32-FA2 + hoisted invariants.
# Rollback: S2MEL_STEPS=25 S2MEL_COMPILE=0 BIGVGAN_CUDA_KERNEL=0.
export S2MEL_STEPS="${S2MEL_STEPS:-18}"
export S2MEL_COMPILE="${S2MEL_COMPILE:-1}"
export BIGVGAN_CUDA_KERNEL="${BIGVGAN_CUDA_KERNEL:-1}"

# Custom sm_89 CUDA GPT-2 engine + hoisted/BF16 s2mel + dedicated streams.
# Instant rollback:
#   GPT_ENGINE=vllm TTS_TF32=0 S2MEL_OPT=0 S2MEL_BF16=0 TTS_STREAMS=0
export GPT_ENGINE="${GPT_ENGINE:-cuda}"
export TTS_TF32="${TTS_TF32:-1}"
export S2MEL_OPT="${S2MEL_OPT:-1}"
export S2MEL_BF16="${S2MEL_BF16:-1}"
export TTS_STREAMS="${TTS_STREAMS:-1}"

# --single_engine_lora is the production mode (UNPACK.md): ONE vLLM engine holds
#   the pristine base weights, the 16 adapter languages attach a per-request
#   LoRA, en/zh run bit-exact base. The two KV budgets are summed into one
#   3.5GB pool. Dropping --single_engine_lora/--lora_peft_dir falls back to the
#   two-engine mode: same API, ~+2.25GiB VRAM, faster adapter path.
# --mel_workers: overrides the measured handoff config (2). Prod has run 4 since
#   the streaming switch; drop back to 2 if the card gets tight.
MEL_WORKERS="${MEL_WORKERS:-4}"

LD_PRELOAD=$REPO/fake_dns.so \
    /home/forsen/miniconda3/envs/index-tts-vllm-fast/bin/python api_server_v2.py \
    --host 0.0.0.0 \
    --port 5113 \
    --base_model_dir "$MODELS/base_pristine" \
    --adapter_model_dir "$MODELS/lora_step12000" \
    --lid_model "$MODELS/glotlid/model.bin" \
    --single_engine_lora \
    --lora_peft_dir "$MODELS/lora_peft_step12000" \
    --gpu_memory_utilization 0.10 \
    --base_kv_cache_memory_bytes 1000000000 \
    --adapter_kv_cache_memory_bytes 2500000000 \
    --qwen_emo_mode disabled \
    --ref_cache_size 32 \
    --mel_workers "$MEL_WORKERS" \
    --code_aligner "$MODELS/code_aligner/run8.pt" \
    --code_aligner_threads 8
