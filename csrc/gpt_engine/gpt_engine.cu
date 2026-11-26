#include "gpt_engine.h"
#include "gpt_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <unordered_map>

using namespace indextts_gpt;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[gpt_engine] CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[gpt_engine] cuBLAS error %d at %s:%d\n", (int)st, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

struct SharedLayerWeights {
    half* ln1_w = nullptr;
    half* ln1_b = nullptr;
    half* c_attn_b = nullptr;
    half* attn_proj_b = nullptr;
    half* ln2_w = nullptr;
    half* ln2_b = nullptr;
    half* c_fc_b = nullptr;
    half* mlp_proj_b = nullptr;
};

struct ModelLayerWeights {
    half* c_attn_w = nullptr;    // [3840, 1280]
    half* attn_proj_w = nullptr; // [1280, 1280]
    half* c_fc_w = nullptr;      // [5120, 1280]
    half* mlp_proj_w = nullptr;  // [1280, 5120]
};

struct DecodeGraphKey {
    int model_id;
    int batch_size;
    int unroll_steps;
    int stop_token_id;
    int top_k;
    uint32_t temp_bits;
    uint32_t topp_bits;
    uint32_t rep_bits;

    bool operator==(const DecodeGraphKey& o) const {
        return model_id == o.model_id && batch_size == o.batch_size &&
               unroll_steps == o.unroll_steps &&
               stop_token_id == o.stop_token_id && top_k == o.top_k &&
               temp_bits == o.temp_bits && topp_bits == o.topp_bits && rep_bits == o.rep_bits;
    }
};

struct DecodeGraphKeyHash {
    size_t operator()(const DecodeGraphKey& k) const {
        size_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix(k.model_id); mix(k.batch_size); mix(k.unroll_steps);
        mix(k.stop_token_id); mix(k.top_k);
        mix(k.temp_bits); mix(k.topp_bits); mix(k.rep_bits);
        return h;
    }
};

struct GptEngine {
    int max_batch = 6;
    int max_seq_len = 2048;
    int num_models = 2;
    size_t vram_bytes = 0;

    cublasHandle_t cublas = nullptr;
    cublasHandle_t fwd_cublas = nullptr;
    cudaStream_t own_stream = nullptr;

    // Shared top weights
    half* audio_emb = nullptr;    // [8194, 1280]
    half* pos_emb = nullptr;      // [1818, 1280]
    half* ln_f_w = nullptr;       // [1280]
    half* ln_f_b = nullptr;
    half* final_norm_w = nullptr; // [1280]
    half* final_norm_b = nullptr;
    half* lm_head_w = nullptr;    // [VOCAB_PAD, 1280]
    half* lm_head_b = nullptr;    // [VOCAB_PAD]

    SharedLayerWeights shared_layers[GPT_N_LAYERS];
    std::vector<ModelLayerWeights> model_layers; // [num_models * GPT_N_LAYERS]

    // Static KV cache: [max_batch, GPT_N_LAYERS, N_HEADS, max_seq_len, HEAD_DIM]
    half* k_cache = nullptr;
    half* v_cache = nullptr;

    // Decode scratch buffers (fixed addresses for CUDA Graphs!)
    half* d_x = nullptr;          // [max_batch, 1280]
    half* d_h = nullptr;          // [max_batch, 1280]
    half* d_q = nullptr;          // [max_batch, 1280]
    half* d_attn_out = nullptr;   // [max_batch, 1280]
    half* d_ff = nullptr;         // [max_batch, 5120]
    float* d_logits = nullptr;    // [max_batch, 8194]
    float* d_part_out = nullptr;  // [max_batch, 20, DECODE_SPLITS, 64]
    float* d_part_m = nullptr;    // [max_batch, 20, DECODE_SPLITS]
    float* d_part_l = nullptr;    // [max_batch, 20, DECODE_SPLITS]

    // Decode device state (fixed addresses for CUDA Graphs!)
    uint32_t* d_seen_mask = nullptr; // [max_batch, MASK_WORDS]
    uint64_t* d_rng_state = nullptr; // [max_batch]
    int* d_kv_pos = nullptr;         // [max_batch]
    int* d_gen_step = nullptr;       // [max_batch]
    int* d_finished = nullptr;       // [max_batch]
    int* d_out_tokens = nullptr;     // [max_batch, max_seq_len]
    int* d_out_lens = nullptr;       // [max_batch]
    int* d_max_new = nullptr;        // [1]
    int* h_finished_pin = nullptr;   // pinned host [max_batch]

    // Prefill workspace (owned by generate_batch / prefill_logits)
    int prefill_cap = 0;
    half* p_x = nullptr;       // [cap, 1280]
    half* p_h = nullptr;       // [cap, 1280]
    half* p_qkv = nullptr;     // [cap, 3840]
    half* p_q = nullptr;       // [20, cap, 64]
    half* p_k = nullptr;       // [20, cap, 64]
    half* p_v = nullptr;       // [20, cap, 64]
    half* p_attn = nullptr;    // [cap, 1280]
    half* p_ff = nullptr;      // [cap, 5120]

    // Dedicated gpt_forward workspace so forward() runs concurrently with generate_batch()
    int fwd_cap = 0;
    half* f_x = nullptr;
    half* f_h = nullptr;
    half* f_qkv = nullptr;
    half* f_q = nullptr;
    half* f_k = nullptr;
    half* f_v = nullptr;
    half* f_attn = nullptr;
    half* f_ff = nullptr;

    // Cached CUDA graphs keyed by (model_id, batch_size, unroll_steps, sampling config)
    std::unordered_map<DecodeGraphKey, cudaGraphExec_t, DecodeGraphKeyHash> graph_cache;

    void* alloc_gpu(size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        vram_bytes += bytes;
        return p;
    }

    void free_gpu(void*& p, size_t bytes) {
        if (p) {
            cudaFree(p);
            p = nullptr;
            if (vram_bytes >= bytes) vram_bytes -= bytes;
        }
    }

    void clear_graphs() {
        for (auto& kv : graph_cache) {
            cudaGraphExecDestroy(kv.second);
        }
        graph_cache.clear();
    }
};

static int ensure_workspace(
    GptEngine* eng, int seq_len, int& cap_ref,
    half*& w_x, half*& w_h, half*& w_qkv, half*& w_q, half*& w_k, half*& w_v, half*& w_attn, half*& w_ff
) {
    if (seq_len <= cap_ref) return 0;
    int cap = seq_len;
    if (cap < 512) cap = 512;
    if (cap > eng->max_seq_len) cap = ((seq_len + 255) / 256) * 256;

    if (cap_ref > 0) {
        size_t c = cap_ref;
        eng->free_gpu((void*&)w_x,    c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_h,    c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_qkv,  c * 3 * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_q,    c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_k,    c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_v,    c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_attn, c * D_MODEL * sizeof(half));
        eng->free_gpu((void*&)w_ff,   c * D_FF * sizeof(half));
        cap_ref = 0;
    }

    size_t c = cap;
    w_x    = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_h    = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_qkv  = (half*)eng->alloc_gpu(c * 3 * D_MODEL * sizeof(half));
    w_q    = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_k    = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_v    = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_attn = (half*)eng->alloc_gpu(c * D_MODEL * sizeof(half));
    w_ff   = (half*)eng->alloc_gpu(c * D_FF * sizeof(half));
    if (!w_x || !w_h || !w_qkv || !w_q || !w_k || !w_v || !w_attn || !w_ff) {
        return -1;
    }
    cap_ref = cap;
    return 0;
}

static int ensure_prefill_workspace(GptEngine* eng, int seq_len) {
    return ensure_workspace(eng, seq_len, eng->prefill_cap,
                            eng->p_x, eng->p_h, eng->p_qkv, eng->p_q, eng->p_k, eng->p_v, eng->p_attn, eng->p_ff);
}

static int ensure_fwd_workspace(GptEngine* eng, int seq_len) {
    return ensure_workspace(eng, seq_len, eng->fwd_cap,
                            eng->f_x, eng->f_h, eng->f_qkv, eng->f_q, eng->f_k, eng->f_v, eng->f_attn, eng->f_ff);
}

static int copy_tensor(void* dst, const void* src, size_t bytes, int is_device) {
    cudaMemcpyKind kind = is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
    return 0;
}

// Helper: cuBLAS FP16 GEMM computing C [M, N] = A [M, K] @ B^T [K, N]
// Row-major C[M, N] = A[M, K] @ W[N, K]^T  <==>  Col-major C^T[N, M] = W^T[N, K] @ A^T[K, M]
static int cublas_gemm_nt(
    cublasHandle_t handle,
    const half* A, // [M, K] row-major
    const half* W, // [N, K] row-major
    void* C,       // [M, N] row-major
    cudaDataType_t c_type,
    int M, int N, int K
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16F, K,
        A, CUDA_R_16F, K,
        &beta,
        C, c_type, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    return 0;
}

// Prefill one sequence [seq_len, 1280] into either:
//   slot >= 0: fills k_cache/v_cache[slot], writes final hidden of LAST token to d_x[slot]
//   slot <  0: forward-only (no KV cache write), writes [seq_len, 1280] FP32 to out_hidden_fp32
static int run_prefill_internal(
    GptEngine* eng,
    int model_id,
    const half* inputs_embeds_gpu,
    int seq_len,
    int slot,
    float* out_hidden_fp32,
    cudaStream_t stream
) {
    bool is_fwd = (slot < 0);
    if (is_fwd) {
        if (ensure_fwd_workspace(eng, seq_len) != 0) return -1;
    } else {
        if (ensure_prefill_workspace(eng, seq_len) != 0) return -1;
    }
    cublasHandle_t blas = is_fwd ? eng->fwd_cublas : eng->cublas;
    half* wx    = is_fwd ? eng->f_x    : eng->p_x;
    half* wh    = is_fwd ? eng->f_h    : eng->p_h;
    half* wqkv  = is_fwd ? eng->f_qkv  : eng->p_qkv;
    half* wq    = is_fwd ? eng->f_q    : eng->p_q;
    half* wk    = is_fwd ? eng->f_k    : eng->p_k;
    half* wv    = is_fwd ? eng->f_v    : eng->p_v;
    half* wattn = is_fwd ? eng->f_attn : eng->p_attn;
    half* wff   = is_fwd ? eng->f_ff   : eng->p_ff;

    CUBLAS_CHECK(cublasSetStream(blas, stream));

    CUDA_CHECK(cudaMemcpyAsync(wx, inputs_embeds_gpu, (size_t)seq_len * D_MODEL * sizeof(half), cudaMemcpyDeviceToDevice, stream));

    for (int l = 0; l < GPT_N_LAYERS; ++l) {
        const auto& sl = eng->shared_layers[l];
        const auto& ml = eng->model_layers[model_id * GPT_N_LAYERS + l];

        layernorm_1280_kernel<false><<<seq_len, 160, 0, stream>>>(
            wx, sl.ln1_w, sl.ln1_b, wh, seq_len, 1e-5f);

        if (cublas_gemm_nt(blas, wh, ml.c_attn_w, wqkv, CUDA_R_16F, seq_len, 3 * D_MODEL, D_MODEL) != 0)
            return -1;

        half* k_dst = wk;
        half* v_dst = wv;
        int kv_stride = seq_len;
        if (slot >= 0) {
            size_t slot_layer_off = ((size_t)slot * GPT_N_LAYERS + l) * N_HEADS * eng->max_seq_len * HEAD_DIM;
            k_dst = eng->k_cache + slot_layer_off;
            v_dst = eng->v_cache + slot_layer_off;
            kv_stride = eng->max_seq_len;
        }
        dim3 unpack_grid(seq_len, N_HEADS);
        unpack_qkv_prefill_kernel<<<unpack_grid, HEAD_DIM, 0, stream>>>(
            wqkv, sl.c_attn_b, wq, k_dst, v_dst, seq_len, kv_stride);

        dim3 attn_grid((seq_len + 3) / 4, N_HEADS);
        causal_flash_prefill_kernel<<<attn_grid, 128, 0, stream>>>(
            wq, k_dst, v_dst, wattn, seq_len, kv_stride);

        if (cublas_gemm_nt(blas, wattn, ml.attn_proj_w, wh, CUDA_R_16F, seq_len, D_MODEL, D_MODEL) != 0)
            return -1;
        int n_d = seq_len * D_MODEL;
        add_bias_resid_kernel<<<(n_d + 255) / 256, 256, 0, stream>>>(wx, wh, sl.attn_proj_b, seq_len);

        layernorm_1280_kernel<false><<<seq_len, 160, 0, stream>>>(
            wx, sl.ln2_w, sl.ln2_b, wh, seq_len, 1e-5f);

        if (cublas_gemm_nt(blas, wh, ml.c_fc_w, wff, CUDA_R_16F, seq_len, D_FF, D_MODEL) != 0)
            return -1;
        int n_ff = seq_len * D_FF;
        add_bias_gelu_kernel<<<(n_ff + 255) / 256, 256, 0, stream>>>(wff, sl.c_fc_b, seq_len);

        if (cublas_gemm_nt(blas, wff, ml.mlp_proj_w, wh, CUDA_R_16F, seq_len, D_MODEL, D_FF) != 0)
            return -1;
        add_bias_resid_kernel<<<(n_d + 255) / 256, 256, 0, stream>>>(wx, wh, sl.mlp_proj_b, seq_len);
    }

    if (slot < 0) {
        layernorm_1280_kernel<false><<<seq_len, 160, 0, stream>>>(
            wx, eng->ln_f_w, eng->ln_f_b, wh, seq_len, 1e-5f);
        layernorm_1280_kernel<true><<<seq_len, 160, 0, stream>>>(
            wh, eng->final_norm_w, eng->final_norm_b, out_hidden_fp32, seq_len, 1e-5f);
    } else {
        const half* last_x = wx + (size_t)(seq_len - 1) * D_MODEL;
        layernorm_1280_kernel<false><<<1, 160, 0, stream>>>(
            last_x, eng->ln_f_w, eng->ln_f_b, wh, 1, 1e-5f);
        layernorm_1280_kernel<false><<<1, 160, 0, stream>>>(
            wh, eng->final_norm_w, eng->final_norm_b, eng->d_h + (size_t)slot * D_MODEL, 1, 1e-5f);
    }
    return 0;
}

// Launch 1 decode step for batch size B (templated):
// 1-warp zero-smem LN, WPB=2/4 zero-smem GEMVs, 2-warp attn_proj, Split-K=16 64T Flash-Decode, 1024T sampler
template <int B>
static void launch_decode_step_t(
    GptEngine* eng,
    int model_id,
    float inv_temp,
    float top_p,
    int top_k,
    float rep_penalty,
    int stop_token_id,
    cudaStream_t stream
) {
    for (int l = 0; l < GPT_N_LAYERS; ++l) {
        const auto& sl = eng->shared_layers[l];
        const auto& ml = eng->model_layers[model_id * GPT_N_LAYERS + l];

        // 1. 1-warp zero-smem ln_1
        layernorm_1280_1warp_kernel<<<B, 32, 0, stream>>>(
            eng->d_x, sl.ln1_w, sl.ln1_b, eng->d_h);

        // 2. c_attn + KV cache write (WPB=4 -> 960 blocks x 128T)
        gemv_k1280_v7_kernel<B, 4, 4><<<(3 * D_MODEL + 3) / 4, 128, 0, stream>>>(
            ml.c_attn_w, sl.c_attn_b, eng->d_h, eng->d_q, 3 * D_MODEL,
            eng->k_cache, eng->v_cache, eng->d_kv_pos, l, eng->max_seq_len);

        // 3. Split-K=16 2-warp (64T) Flash-Decode Attention + reduce
        dim3 split_grid(N_HEADS, DECODE_SPLITS, B);
        flash_decode_split_v7_kernel<<<split_grid, 64, 0, stream>>>(
            eng->d_q, eng->k_cache, eng->v_cache, eng->d_kv_pos,
            eng->d_part_out, eng->d_part_m, eng->d_part_l, l, eng->max_seq_len);
        dim3 red_grid(N_HEADS, B);
        flash_decode_reduce_v7_kernel<<<red_grid, HEAD_DIM, 0, stream>>>(
            eng->d_part_out, eng->d_part_m, eng->d_part_l, eng->d_attn_out);

        // 4. Zero-smem WPB=2 attn.c_proj (640 blocks x 64T) + residual
        gemv_attn_proj_v7_kernel<B><<<D_MODEL / 2, 64, 0, stream>>>(
            ml.attn_proj_w, sl.attn_proj_b, eng->d_attn_out, eng->d_x);

        // 5. 1-warp zero-smem ln_2
        layernorm_1280_1warp_kernel<<<B, 32, 0, stream>>>(
            eng->d_x, sl.ln2_w, sl.ln2_b, eng->d_h);

        // 6. mlp.c_fc + gelu_new (WPB=4 -> 1280 blocks x 128T)
        gemv_k1280_v7_kernel<B, 4, 2><<<(D_FF + 3) / 4, 128, 0, stream>>>(
            ml.c_fc_w, sl.c_fc_b, eng->d_h, eng->d_ff, D_FF,
            nullptr, nullptr, nullptr, 0, 0);

        // 7. 5-warp mlp.c_proj + residual (1280 blocks x 160T)
        gemv_k5120_v7_resid_kernel<B><<<D_MODEL, 160, 0, stream>>>(
            ml.mlp_proj_w, sl.mlp_proj_b, eng->d_ff, eng->d_x);
    }

    // 1-warp fused double LayerNorm (ln_f -> final_norm)
    double_layernorm_1280_1warp_kernel<<<B, 32, 0, stream>>>(
        eng->d_x, eng->ln_f_w, eng->ln_f_b, eng->final_norm_w, eng->final_norm_b, eng->d_h);

    // Zero-smem lm_head -> FP32 logits [B, 8194]
    gemv_k1280_v7_kernel<B, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(
        eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD,
        nullptr, nullptr, nullptr, 0, 0);

    // 1024-thread 12-iter ballot binary-search sampler + next-token embed
    sample_and_embed_v7_kernel<<<B, 1024, 0, stream>>>(
        eng->d_logits, eng->audio_emb, eng->pos_emb,
        eng->d_seen_mask, eng->d_rng_state, eng->d_kv_pos, eng->d_gen_step,
        eng->d_finished, eng->d_out_tokens, eng->d_out_lens, eng->d_max_new,
        eng->d_x, eng->max_seq_len, inv_temp, top_p, top_k, rep_penalty, stop_token_id);
}

static void dispatch_decode_step(
    GptEngine* eng, int model_id, int B,
    float inv_temp, float top_p, int top_k, float rep_penalty, int stop_token_id,
    cudaStream_t stream
) {
    switch (B) {
        case 1: launch_decode_step_t<1>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        case 2: launch_decode_step_t<2>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        case 3: launch_decode_step_t<3>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        case 4: launch_decode_step_t<4>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        case 5: launch_decode_step_t<5>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        case 6: launch_decode_step_t<6>(eng, model_id, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream); break;
        default: break;
    }
}

static uint32_t float_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static cudaGraphExec_t get_or_capture_graph(
    GptEngine* eng,
    int model_id,
    int B,
    int unroll_steps,
    float inv_temp,
    float top_p,
    int top_k,
    float rep_penalty,
    int stop_token_id,
    cudaStream_t stream
) {
    DecodeGraphKey key{
        model_id, B, unroll_steps, stop_token_id, top_k,
        float_bits(inv_temp), float_bits(top_p), float_bits(rep_penalty)
    };
    auto it = eng->graph_cache.find(key);
    if (it != eng->graph_cache.end()) return it->second;

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpt_engine] BeginCapture failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    for (int u = 0; u < unroll_steps; ++u) {
        dispatch_decode_step(eng, model_id, B, inv_temp, top_p, top_k, rep_penalty, stop_token_id, stream);
    }
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpt_engine] EndCapture failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        fprintf(stderr, "[gpt_engine] GraphInstantiate failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    eng->graph_cache[key] = exec;
    return exec;
}

// ============================================================================
// C API Implementation
// ============================================================================
extern "C" {

GptEngine* gpt_engine_create(int max_batch, int max_seq_len, int num_models) {
    if (max_batch < 1 || max_batch > 6) {
        fprintf(stderr, "[gpt_engine] max_batch must be 1..6, got %d\n", max_batch);
        return nullptr;
    }
    auto* eng = new GptEngine();
    eng->max_batch = max_batch;
    eng->max_seq_len = max_seq_len;
    eng->num_models = num_models;

    if (cublasCreate(&eng->cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasCreate(&eng->fwd_cublas) != CUBLAS_STATUS_SUCCESS) {
        if (eng->cublas) cublasDestroy(eng->cublas);
        if (eng->fwd_cublas) cublasDestroy(eng->fwd_cublas);
        delete eng;
        return nullptr;
    }
    // Keep FP32 accumulation on Tensor Cores
    cublasSetMathMode(eng->cublas, CUBLAS_DEFAULT_MATH);
    cublasSetMathMode(eng->fwd_cublas, CUBLAS_DEFAULT_MATH);
    if (cudaStreamCreateWithFlags(&eng->own_stream, cudaStreamNonBlocking) != cudaSuccess) {
        cublasDestroy(eng->fwd_cublas);
        cublasDestroy(eng->cublas);
        delete eng;
        return nullptr;
    }

    // Shared top weights
    eng->audio_emb    = (half*)eng->alloc_gpu((size_t)VOCAB_SIZE * D_MODEL * sizeof(half));
    eng->pos_emb      = (half*)eng->alloc_gpu((size_t)MAX_POS * D_MODEL * sizeof(half));
    eng->ln_f_w       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
    eng->ln_f_b       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
    eng->final_norm_w = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
    eng->final_norm_b = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
    eng->lm_head_w    = (half*)eng->alloc_gpu((size_t)VOCAB_PAD * D_MODEL * sizeof(half));
    eng->lm_head_b    = (half*)eng->alloc_gpu((size_t)VOCAB_PAD * sizeof(half));
    cudaMemset(eng->lm_head_w, 0, (size_t)VOCAB_PAD * D_MODEL * sizeof(half));
    cudaMemset(eng->lm_head_b, 0, (size_t)VOCAB_PAD * sizeof(half));

    for (int l = 0; l < GPT_N_LAYERS; ++l) {
        auto& sl = eng->shared_layers[l];
        sl.ln1_w       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
        sl.ln1_b       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
        sl.c_attn_b    = (half*)eng->alloc_gpu(3 * D_MODEL * sizeof(half));
        sl.attn_proj_b = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
        sl.ln2_w       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
        sl.ln2_b       = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
        sl.c_fc_b      = (half*)eng->alloc_gpu(D_FF * sizeof(half));
        sl.mlp_proj_b  = (half*)eng->alloc_gpu(D_MODEL * sizeof(half));
    }

    eng->model_layers.resize(num_models * GPT_N_LAYERS);
    for (int m = 0; m < num_models; ++m) {
        for (int l = 0; l < GPT_N_LAYERS; ++l) {
            auto& ml = eng->model_layers[m * GPT_N_LAYERS + l];
            ml.c_attn_w    = (half*)eng->alloc_gpu((size_t)(3 * D_MODEL) * D_MODEL * sizeof(half));
            ml.attn_proj_w = (half*)eng->alloc_gpu((size_t)D_MODEL * D_MODEL * sizeof(half));
            ml.c_fc_w      = (half*)eng->alloc_gpu((size_t)D_FF * D_MODEL * sizeof(half));
            ml.mlp_proj_w  = (half*)eng->alloc_gpu((size_t)D_MODEL * D_FF * sizeof(half));
        }
    }

    // Static KV cache
    size_t kv_elems = (size_t)max_batch * GPT_N_LAYERS * N_HEADS * max_seq_len * HEAD_DIM;
    eng->k_cache = (half*)eng->alloc_gpu(kv_elems * sizeof(half));
    eng->v_cache = (half*)eng->alloc_gpu(kv_elems * sizeof(half));

    // Decode buffers
    eng->d_x        = (half*)eng->alloc_gpu((size_t)max_batch * D_MODEL * sizeof(half));
    eng->d_h        = (half*)eng->alloc_gpu((size_t)max_batch * D_MODEL * sizeof(half));
    eng->d_q        = (half*)eng->alloc_gpu((size_t)max_batch * D_MODEL * sizeof(half));
    eng->d_attn_out = (half*)eng->alloc_gpu((size_t)max_batch * D_MODEL * sizeof(half));
    eng->d_ff       = (half*)eng->alloc_gpu((size_t)max_batch * D_FF * sizeof(half));
    eng->d_logits   = (float*)eng->alloc_gpu((size_t)max_batch * VOCAB_SIZE * sizeof(float));
    eng->d_part_out  = (float*)eng->alloc_gpu((size_t)max_batch * N_HEADS * DECODE_SPLITS * HEAD_DIM * sizeof(float));
    eng->d_part_m    = (float*)eng->alloc_gpu((size_t)max_batch * N_HEADS * DECODE_SPLITS * sizeof(float));
    eng->d_part_l    = (float*)eng->alloc_gpu((size_t)max_batch * N_HEADS * DECODE_SPLITS * sizeof(float));

    // Maximize L1 cache carveout for zero-smem decode GEMVs
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<1, 4, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<1, 4, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<1, 4, 3>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_attn_proj_v7_kernel<1>,   cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<4, 4, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<4, 4, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_k1280_v7_kernel<4, 4, 3>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_attn_proj_v7_kernel<4>,   cudaFuncAttributePreferredSharedMemoryCarveout, 0);

    eng->d_seen_mask  = (uint32_t*)eng->alloc_gpu((size_t)max_batch * MASK_WORDS * sizeof(uint32_t));
    eng->d_rng_state  = (uint64_t*)eng->alloc_gpu((size_t)max_batch * sizeof(uint64_t));
    eng->d_kv_pos     = (int*)eng->alloc_gpu((size_t)max_batch * sizeof(int));
    eng->d_gen_step   = (int*)eng->alloc_gpu((size_t)max_batch * sizeof(int));
    eng->d_finished   = (int*)eng->alloc_gpu((size_t)max_batch * sizeof(int));
    eng->d_out_tokens = (int*)eng->alloc_gpu((size_t)max_batch * max_seq_len * sizeof(int));
    eng->d_out_lens   = (int*)eng->alloc_gpu((size_t)max_batch * sizeof(int));
    eng->d_max_new    = (int*)eng->alloc_gpu(sizeof(int));
    cudaMallocHost(&eng->h_finished_pin, (size_t)max_batch * sizeof(int));

    if (ensure_prefill_workspace(eng, 1024) != 0 || ensure_fwd_workspace(eng, 1024) != 0) {
        gpt_engine_destroy(eng);
        return nullptr;
    }

    return eng;
}

void gpt_engine_destroy(GptEngine* eng) {
    if (!eng) return;
    eng->clear_graphs();
    if (eng->h_finished_pin) cudaFreeHost(eng->h_finished_pin);
    if (eng->own_stream) cudaStreamDestroy(eng->own_stream);
    if (eng->fwd_cublas) cublasDestroy(eng->fwd_cublas);
    if (eng->cublas) cublasDestroy(eng->cublas);

    auto f = [](void* p) { if (p) cudaFree(p); };
    f(eng->audio_emb); f(eng->pos_emb); f(eng->ln_f_w); f(eng->ln_f_b);
    f(eng->final_norm_w); f(eng->final_norm_b); f(eng->lm_head_w); f(eng->lm_head_b);
    for (int l = 0; l < GPT_N_LAYERS; ++l) {
        auto& sl = eng->shared_layers[l];
        f(sl.ln1_w); f(sl.ln1_b); f(sl.c_attn_b); f(sl.attn_proj_b);
        f(sl.ln2_w); f(sl.ln2_b); f(sl.c_fc_b); f(sl.mlp_proj_b);
    }
    for (auto& ml : eng->model_layers) {
        f(ml.c_attn_w); f(ml.attn_proj_w); f(ml.c_fc_w); f(ml.mlp_proj_w);
    }
    f(eng->k_cache); f(eng->v_cache);
    f(eng->d_x); f(eng->d_h); f(eng->d_q); f(eng->d_attn_out); f(eng->d_ff);
    f(eng->d_logits); f(eng->d_part_out); f(eng->d_part_m); f(eng->d_part_l);
    f(eng->d_seen_mask); f(eng->d_rng_state); f(eng->d_kv_pos); f(eng->d_gen_step);
    f(eng->d_finished); f(eng->d_out_tokens); f(eng->d_out_lens); f(eng->d_max_new);
    f(eng->p_x); f(eng->p_h); f(eng->p_qkv); f(eng->p_q); f(eng->p_k);
    f(eng->p_v); f(eng->p_attn); f(eng->p_ff);
    f(eng->f_x); f(eng->f_h); f(eng->f_qkv); f(eng->f_q); f(eng->f_k);
    f(eng->f_v); f(eng->f_attn); f(eng->f_ff);
    delete eng;
}

size_t gpt_engine_vram_bytes(GptEngine* eng) {
    return eng ? eng->vram_bytes : 0;
}

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
) {
    if (copy_tensor(eng->audio_emb,    audio_emb_fp16,    (size_t)VOCAB_SIZE * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->pos_emb,      pos_emb_fp16,      (size_t)MAX_POS * D_MODEL * sizeof(half), is_device)) return -1;
    // Pin row 0 of pos_emb to zero (matches index_tts_gpt2_vllm_v1.py)
    CUDA_CHECK(cudaMemset(eng->pos_emb, 0, D_MODEL * sizeof(half)));
    if (copy_tensor(eng->ln_f_w,       ln_f_w_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->ln_f_b,       ln_f_b_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->final_norm_w, final_norm_w_fp16, D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->final_norm_b, final_norm_b_fp16, D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->lm_head_w,    lm_head_w_fp16,    (size_t)VOCAB_SIZE * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(eng->lm_head_b,    lm_head_b_fp16,    (size_t)VOCAB_SIZE * sizeof(half), is_device)) return -1;
    return 0;
}

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
) {
    if (layer < 0 || layer >= GPT_N_LAYERS) return -1;
    auto& sl = eng->shared_layers[layer];
    if (copy_tensor(sl.ln1_w,       ln1_w_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.ln1_b,       ln1_b_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.c_attn_b,    c_attn_b_fp16,    3 * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.attn_proj_b, attn_proj_b_fp16, D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.ln2_w,       ln2_w_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.ln2_b,       ln2_b_fp16,       D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.c_fc_b,      c_fc_b_fp16,      D_FF * sizeof(half), is_device)) return -1;
    if (copy_tensor(sl.mlp_proj_b,  mlp_proj_b_fp16,  D_MODEL * sizeof(half), is_device)) return -1;
    return 0;
}

int gpt_engine_set_model_layer(
    GptEngine* eng,
    int model_id,
    int layer,
    const void* c_attn_w_fp16,
    const void* attn_proj_w_fp16,
    const void* c_fc_w_fp16,
    const void* mlp_proj_w_fp16,
    int is_device
) {
    if (model_id < 0 || model_id >= eng->num_models || layer < 0 || layer >= GPT_N_LAYERS) return -1;
    auto& ml = eng->model_layers[model_id * GPT_N_LAYERS + layer];
    if (copy_tensor(ml.c_attn_w,    c_attn_w_fp16,    (size_t)(3 * D_MODEL) * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(ml.attn_proj_w, attn_proj_w_fp16, (size_t)D_MODEL * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(ml.c_fc_w,      c_fc_w_fp16,      (size_t)D_FF * D_MODEL * sizeof(half), is_device)) return -1;
    if (copy_tensor(ml.mlp_proj_w,  mlp_proj_w_fp16,  (size_t)D_MODEL * D_FF * sizeof(half), is_device)) return -1;
    return 0;
}

int gpt_engine_forward(
    GptEngine* eng,
    int model_id,
    const void* inputs_embeds_fp16_gpu,
    int seq_len,
    void* out_hidden_fp32_gpu,
    void* cuda_stream
) {
    if (!eng || model_id < 0 || model_id >= eng->num_models || seq_len <= 0) return -1;
    // Run on caller's exact stream (including default stream 0) so PyTorch producer/consumer ops stay ordered
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    return run_prefill_internal(eng, model_id, (const half*)inputs_embeds_fp16_gpu, seq_len, -1, (float*)out_hidden_fp32_gpu, stream);
}

int gpt_engine_prefill_logits(
    GptEngine* eng,
    int model_id,
    const void* inputs_embeds_fp16_gpu,
    int seq_len,
    void* out_logits_fp32_gpu,
    void* cuda_stream
) {
    if (!eng || model_id < 0 || model_id >= eng->num_models || seq_len <= 0) return -1;
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    if (run_prefill_internal(eng, model_id, (const half*)inputs_embeds_fp16_gpu, seq_len, 0, nullptr, stream) != 0)
        return -1;
    // eng->d_h[0] now holds final_norm(ln_f(last_hidden))
    gemv_k1280_v7_kernel<1, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(
        eng->lm_head_w, eng->lm_head_b, eng->d_h, out_logits_fp32_gpu, VOCAB_PAD,
        nullptr, nullptr, nullptr, 0, 0);
    return 0;
}

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
) {
    if (!eng || model_id < 0 || model_id >= eng->num_models) return -1;
    if (batch_size < 1 || batch_size > eng->max_batch) return -1;

    // Cap max_new_tokens so total seq never exceeds max_seq_len
    int max_p = 0;
    for (int b = 0; b < batch_size; ++b) {
        if (prompt_lens[b] > max_p) max_p = prompt_lens[b];
    }
    int avail = eng->max_seq_len - max_p;
    if (avail <= 0) return -1;
    int orig_max_new = max_new_tokens;
    if (max_new_tokens > avail) max_new_tokens = avail;

    cudaStream_t stream = cuda_stream ? (cudaStream_t)cuda_stream : eng->own_stream;

    // 1. Reset device decode state
    CUDA_CHECK(cudaMemsetAsync(eng->d_seen_mask, 0, (size_t)batch_size * MASK_WORDS * sizeof(uint32_t), stream));
    if (penalize_placeholder_zero) {
        uint32_t one = 1u;
        for (int b = 0; b < batch_size; ++b) {
            CUDA_CHECK(cudaMemcpyAsync(eng->d_seen_mask + (size_t)b * MASK_WORDS, &one, sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        }
    }
    int h_kv_pos[6], h_zero[6] = {0, 0, 0, 0, 0, 0};
    for (int b = 0; b < batch_size; ++b) h_kv_pos[b] = prompt_lens[b] - 1;
    CUDA_CHECK(cudaMemcpyAsync(eng->d_kv_pos,   h_kv_pos, batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(eng->d_gen_step, h_zero,   batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(eng->d_finished, h_zero,   batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(eng->d_out_lens, h_zero,   batch_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(eng->d_max_new,  &max_new_tokens, sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(eng->d_rng_state, seeds,   batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    // 2. Prefill each prompt into KV slot b (writes last hidden after ln_f+final_norm into eng->d_h[b])
    for (int b = 0; b < batch_size; ++b) {
        if (run_prefill_internal(eng, model_id, (const half*)prompt_embeds_gpu_ptrs[b], prompt_lens[b], b, nullptr, stream) != 0)
            return -1;
    }

    // 3. First token logits + sampling directly from prefill's last hidden (eng->d_h[0..B-1])
    float inv_temp = (temperature > 0.0f) ? (1.0f / temperature) : 0.0f;
    switch (batch_size) {
        case 1: gemv_k1280_v7_kernel<1, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
        case 2: gemv_k1280_v7_kernel<2, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
        case 3: gemv_k1280_v7_kernel<3, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
        case 4: gemv_k1280_v7_kernel<4, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
        case 5: gemv_k1280_v7_kernel<5, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
        case 6: gemv_k1280_v7_kernel<6, 4, 3><<<(VOCAB_PAD + 3) / 4, 128, 0, stream>>>(eng->lm_head_w, eng->lm_head_b, eng->d_h, eng->d_logits, VOCAB_PAD, nullptr, nullptr, nullptr, 0, 0); break;
    }
    sample_and_embed_v7_kernel<<<batch_size, 1024, 0, stream>>>(
        eng->d_logits, eng->audio_emb, eng->pos_emb,
        eng->d_seen_mask, eng->d_rng_state, eng->d_kv_pos, eng->d_gen_step,
        eng->d_finished, eng->d_out_tokens, eng->d_out_lens, eng->d_max_new,
        eng->d_x, eng->max_seq_len, inv_temp, top_p, top_k, repetition_penalty, stop_token_id);

    // 4. Autoregressive decode using unrolled CUDA Graphs (K_UNROLL=8 caps post-EOS overshoot to ~3.5ms)
    constexpr int K_UNROLL = 8;
    int steps_done = 1;
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraphExec_t graph_step1 = nullptr;
    if (max_new_tokens > K_UNROLL) {
        graph_exec = get_or_capture_graph(
            eng, model_id, batch_size, K_UNROLL,
            inv_temp, top_p, top_k, repetition_penalty, stop_token_id, stream);
    }
    if (max_new_tokens > 1) {
        graph_step1 = get_or_capture_graph(
            eng, model_id, batch_size, 1,
            inv_temp, top_p, top_k, repetition_penalty, stop_token_id, stream);
    }

    while (steps_done < max_new_tokens) {
        int rem = max_new_tokens - steps_done;
        if (graph_exec && rem >= K_UNROLL) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            steps_done += K_UNROLL;
        } else if (graph_step1) {
            for (int i = 0; i < rem; ++i) {
                CUDA_CHECK(cudaGraphLaunch(graph_step1, stream));
            }
            steps_done += rem;
        } else {
            for (int i = 0; i < rem; ++i) {
                dispatch_decode_step(eng, model_id, batch_size,
                                     inv_temp, top_p, top_k, repetition_penalty, stop_token_id, stream);
            }
            steps_done += rem;
        }

        // Check completion once every K_UNROLL (16) steps
        CUDA_CHECK(cudaMemcpyAsync(eng->h_finished_pin, eng->d_finished, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bool all_done = true;
        for (int b = 0; b < batch_size; ++b) {
            if (!eng->h_finished_pin[b]) { all_done = false; break; }
        }
        if (all_done) break;
    }

    // 5. Copy generated tokens & lengths back to host (per-slot using actual length!)
    CUDA_CHECK(cudaMemcpyAsync(out_lens_host, eng->d_out_lens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int b = 0; b < batch_size; ++b) {
        int len = out_lens_host[b];
        if (len > orig_max_new) len = orig_max_new;
        if (len > 0) {
            CUDA_CHECK(cudaMemcpyAsync(
                out_tokens_host + (size_t)b * orig_max_new,
                eng->d_out_tokens + (size_t)b * eng->max_seq_len,
                (size_t)len * sizeof(int),
                cudaMemcpyDeviceToHost, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

int gpt_tf32_flash_attn_d64(
    const void* q_fp32_gpu,
    const void* k_fp32_gpu,
    const void* v_fp32_gpu,
    void* out_fp32_gpu,
    int bh,
    int seq_len,
    float scale,
    void* cuda_stream
) {
    if (bh <= 0 || seq_len <= 0) return -1;
    dim3 grid((seq_len + 63) / 64, bh);
    float scale_log2e = scale * 1.4426950408889634f;
    tf32_flash_attn_d64_kernel<<<grid, 128, 0, (cudaStream_t)cuda_stream>>>(
        (const float*)q_fp32_gpu,
        (const float*)k_fp32_gpu,
        (const float*)v_fp32_gpu,
        (float*)out_fp32_gpu,
        seq_len,
        scale_log2e
    );
    return 0;
}

} // extern "C"
