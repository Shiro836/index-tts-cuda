#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GptEngine GptEngine;

// Model constants fixed for Index-TTS2 GPT-2 (24L, 1280D, 20H, 5120FF, 8194V)
#define GPT_N_LAYERS    24
#define GPT_D_MODEL     1280
#define GPT_N_HEADS     20
#define GPT_HEAD_DIM    64
#define GPT_D_FF        5120
#define GPT_VOCAB_SIZE  8194
#define GPT_VOCAB_PAD   8200
#define GPT_MAX_POS     1818
#define GPT_STOP_TOKEN  8193

GptEngine* gpt_engine_create(int max_batch, int max_seq_len, int num_models);
void gpt_engine_destroy(GptEngine* eng);
size_t gpt_engine_vram_bytes(GptEngine* eng);

// Load shared FP16 weights (device or host pointers; is_device=1 for GPU tensors).
// lm_head_w is [8194, 1280] FP16 row-major; lm_head_b is [8194] FP16.
// audio_emb is [8194, 1280] FP16; pos_emb is [1818, 1280] FP16 (row 0 zeroed by engine).
// ln_f_w/b, final_norm_w/b are [1280] FP16.
int gpt_engine_set_shared_top(
    GptEngine* eng,
    const void* audio_emb_fp16,
    const void* pos_emb_fp16,
    const void* ln_f_w_fp16,
    const void* ln_f_b_fp16,
    const void* final_norm_w_fp16,
    const void* final_norm_b_fp16,
    const void* lm_head_w_fp16,
    const void* lm_head_b_fp16,
    int is_device
);

// Load shared per-layer norms & biases (identical across base and adapter).
int gpt_engine_set_shared_layer(
    GptEngine* eng,
    int layer,
    const void* ln1_w_fp16,
    const void* ln1_b_fp16,
    const void* c_attn_b_fp16,
    const void* attn_proj_b_fp16,
    const void* ln2_w_fp16,
    const void* ln2_b_fp16,
    const void* c_fc_b_fp16,
    const void* mlp_proj_b_fp16,
    int is_device
);

// Load per-model (0=base, 1=adapter) linear weights, pre-transposed to row-major [N, K] FP16:
//   c_attn_w:   [3840, 1280]
//   attn_proj_w:[1280, 1280]
//   c_fc_w:     [5120, 1280]
//   mlp_proj_w: [1280, 5120]
int gpt_engine_set_model_layer(
    GptEngine* eng,
    int model_id,
    int layer,
    const void* c_attn_w_fp16,
    const void* attn_proj_w_fp16,
    const void* c_fc_w_fp16,
    const void* mlp_proj_w_fp16,
    int is_device
);

// Run full causal GPT-2 + ln_f + final_norm on GPU inputs_embeds [seq_len, 1280] FP16,
// writing [seq_len, 1280] FP32 to out_hidden_fp32 (GPU pointer).
// Used to replace UnifiedVoice.forward() (s2mel latent pass).
int gpt_engine_forward(
    GptEngine* eng,
    int model_id,
    const void* inputs_embeds_fp16_gpu,
    int seq_len,
    void* out_hidden_fp32_gpu,
    void* cuda_stream
);

// Run prefill on [seq_len, 1280] FP16 GPU tensor and write last-token [8194] FP32 logits (GPU ptr).
// Used for parity testing against PyTorch/vLLM.
int gpt_engine_prefill_logits(
    GptEngine* eng,
    int model_id,
    const void* inputs_embeds_fp16_gpu,
    int seq_len,
    void* out_logits_fp32_gpu,
    void* cuda_stream
);

// Batched prefill + CUDA-graphed autoregressive decode with in-GPU sampling.
// prompt_embeds_gpu_ptrs[b] points to [prompt_lens[b], 1280] FP16 on GPU.
// out_tokens_host must have capacity batch_size * max_new_tokens ints.
// out_lens_host must have capacity batch_size ints (includes stop_token if emitted).
// If temperature <= 0.0f, uses deterministic greedy argmax (still applies repetition_penalty).
// If penalize_placeholder_zero != 0, marks token 0 in seen_mask before decoding (matches vLLM).
int gpt_engine_generate_batch(
    GptEngine* eng,
    int model_id,
    int batch_size,
    const void* const* prompt_embeds_gpu_ptrs,
    const int* prompt_lens,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    float repetition_penalty,
    int stop_token_id,
    int penalize_placeholder_zero,
    const uint64_t* seeds,
    int* out_tokens_host,
    int* out_lens_host,
    void* cuda_stream
);

// Unmasked FP32-I/O TF32 Tensor-Core FlashAttention-2 (head_dim=64, sm_89+ compatible).
// q, k, v, out: contiguous [BH, T, 64] float32 device pointers.
int gpt_tf32_flash_attn_d64(
    const void* q_fp32_gpu,
    const void* k_fp32_gpu,
    const void* v_fp32_gpu,
    void* out_fp32_gpu,
    int bh,
    int seq_len,
    float scale,
    void* cuda_stream
);

#ifdef __cplusplus
}
#endif
