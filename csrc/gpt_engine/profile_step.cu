#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "gpt_kernels.cuh"

using namespace indextts_gpt;

static float bench(void (*fn)(cudaStream_t), cudaStream_t s, int iters = 200) {
    for (int i = 0; i < 20; ++i) fn(s);
    cudaStreamSynchronize(s);
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; ++i) fn(s);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (ms * 1000.0f) / iters; // microseconds
}

// Globals for quick lambda-free benching
half *g_w_qkv, *g_w_proj, *g_w_fc, *g_w_mproj, *g_w_lm, *g_b, *g_x, *g_h, *g_q, *g_ff, *g_kc, *g_vc, *g_aemb, *g_pemb;
float *g_logits, *g_po, *g_pm, *g_pl;
int *g_kvpos, *g_gstep, *g_fin, *g_otok, *g_olen, *g_maxnew;
uint32_t *g_mask;
uint64_t *g_rng;

template<int B> void run_ln(cudaStream_t s) { layernorm_1280_kernel<false><<<B, 160, 0, s>>>(g_x, g_b, g_b, g_h, B, 1e-5f); }
template<int B> void run_cattn(cudaStream_t s) { gemv_k1280_kernel<B, 4, 4><<<960, 128, 0, s>>>(g_w_qkv, g_b, g_h, g_q, 3840, g_kc, g_vc, g_kvpos, 0, 1818); }
template<int B> void run_fsplit(cudaStream_t s) { flash_decode_split_kernel<<<dim3(20, 8, B), 128, 0, s>>>(g_q, g_kc, g_vc, g_kvpos, g_po, g_pm, g_pl, 0, 1818); }
template<int B> void run_fred(cudaStream_t s) { flash_decode_reduce_kernel<<<dim3(20, B), 64, 0, s>>>(g_po, g_pm, g_pl, g_h); }
template<int B> void run_aproj(cudaStream_t s) { gemv_k1280_kernel<B, 2, 1><<<640, 64, 0, s>>>(g_w_proj, g_b, g_h, g_x, 1280, nullptr, nullptr, nullptr, 0, 0); }
template<int B> void run_fc(cudaStream_t s) { gemv_k1280_kernel<B, 4, 2><<<1280, 128, 0, s>>>(g_w_fc, g_b, g_h, g_ff, 5120, nullptr, nullptr, nullptr, 0, 0); }
template<int B> void run_mproj(cudaStream_t s) { gemv_k5120_resid_kernel<B><<<1280, 128, 0, s>>>(g_w_mproj, g_b, g_ff, g_x); }
template<int B> void run_lm(cudaStream_t s) { gemv_k1280_kernel<B, 4, 3><<<2050, 128, 0, s>>>(g_w_lm, g_b, g_h, g_logits, 8200, nullptr, nullptr, nullptr, 0, 0); }
template<int B> void run_samp(cudaStream_t s) {
    sample_and_embed_kernel<<<B, 256, 0, s>>>(g_logits, g_aemb, g_pemb, g_mask, g_rng, g_kvpos, g_gstep, g_fin, g_otok, g_olen, g_maxnew, g_x, 1818, 1.0f, 0.8f, 30, 10.0f, 8193);
}

int main() {
    cudaStream_t s; cudaStreamCreate(&s);
    // Allocate 24-layer sized weights? Just benchmark 1 layer L2-cold by rotating 24 layers?
    // First measure single-layer (warm L2) AND 24-layer cold DRAM!
    constexpr int L = 24;
    cudaMalloc(&g_w_qkv,   (size_t)L * 3840 * 1280 * 2);
    cudaMalloc(&g_w_proj,  (size_t)L * 1280 * 1280 * 2);
    cudaMalloc(&g_w_fc,    (size_t)L * 5120 * 1280 * 2);
    cudaMalloc(&g_w_mproj, (size_t)L * 1280 * 5120 * 2);
    cudaMalloc(&g_w_lm,    (size_t)8200 * 1280 * 2);
    cudaMalloc(&g_b,       8200 * 2);
    cudaMalloc(&g_x,       6 * 1280 * 2);
    cudaMalloc(&g_h,       6 * 1280 * 2);
    cudaMalloc(&g_q,       6 * 1280 * 2);
    cudaMalloc(&g_ff,      6 * 5120 * 2);
    cudaMalloc(&g_kc,      (size_t)4 * L * 20 * 1818 * 64 * 2);
    cudaMalloc(&g_vc,      (size_t)4 * L * 20 * 1818 * 64 * 2);
    cudaMalloc(&g_aemb,    8194 * 1280 * 2);
    cudaMalloc(&g_pemb,    1818 * 1280 * 2);
    cudaMalloc(&g_logits,  6 * 8194 * 4);
    cudaMalloc(&g_po,      6 * 20 * 8 * 64 * 4);
    cudaMalloc(&g_pm,      6 * 20 * 8 * 4);
    cudaMalloc(&g_pl,      6 * 20 * 8 * 4);
    cudaMalloc(&g_kvpos,   6 * 4);
    cudaMalloc(&g_gstep,   6 * 4);
    cudaMalloc(&g_fin,     6 * 4);
    cudaMalloc(&g_otok,    6 * 1818 * 4);
    cudaMalloc(&g_olen,    6 * 4);
    cudaMalloc(&g_maxnew,  4);
    cudaMalloc(&g_mask,    6 * MASK_WORDS * 4);
    cudaMalloc(&g_rng,     6 * 8);

    int h_kv[6] = {300, 300, 300, 300, 300, 300};
    int h_z[6] = {0, 0, 0, 0, 0, 0};
    int h_mn = 1800;
    cudaMemcpy(g_kvpos, h_kv, 24, cudaMemcpyHostToDevice);
    cudaMemcpy(g_gstep, h_z, 24, cudaMemcpyHostToDevice);
    cudaMemcpy(g_fin,   h_z, 24, cudaMemcpyHostToDevice);
    cudaMemcpy(g_maxnew, &h_mn, 4, cudaMemcpyHostToDevice);
    cudaMemset(g_mask, 0, 6 * MASK_WORDS * 4);
    cudaMemset(g_x, 0, 6 * 1280 * 2);
    cudaMemset(g_h, 0, 6 * 1280 * 2);
    cudaMemset(g_q, 0, 6 * 1280 * 2);
    cudaMemset(g_ff, 0, 6 * 5120 * 2);
    cudaMemset(g_logits, 0, 6 * 8194 * 4);

    // Measure cold-DRAM 24-layer sweep for each kernel type at B=1 and B=4!
    auto bench24 = [&](auto fn, const char* name, double bytes_mb) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int i = 0; i < 5; ++i) for (int l = 0; l < L; ++l) fn(l);
        cudaStreamSynchronize(s);
        cudaEventRecord(a, s);
        for (int i = 0; i < 30; ++i) for (int l = 0; l < L; ++l) fn(l);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        float us_total = (ms * 1000.0f) / 30.0f; // 24 layers total per token
        double bw = bytes_mb > 0 ? (bytes_mb * 24.0 / 1000.0) / (us_total / 1e6) : 0.0;
        printf("  %-22s: %6.1f us/tok (%5.2f us/layer) | eff_BW = %6.1f GB/s\n", name, us_total, us_total/24.0f, bw);
        cudaEventDestroy(a); cudaEventDestroy(b);
        return us_total;
    };

    printf("=== COLD-DRAM 24-LAYER BREAKDOWN (B=1, seq=300) ===\n");
    float t = 0;
    t += bench24([&](int l){ layernorm_1280_kernel<false><<<1, 160, 0, s>>>(g_x, g_b, g_b, g_h, 1, 1e-5f); }, "ln_1 (x24)", 0);
    t += bench24([&](int l){ gemv_k1280_kernel<1, 4, 4><<<960, 128, 0, s>>>(g_w_qkv + (size_t)l*3840*1280, g_b, g_h, g_q, 3840, g_kc, g_vc, g_kvpos, l, 1818); }, "c_attn (3840x1280)", 9.8304);
    t += bench24([&](int l){ flash_decode_split_kernel<<<dim3(20, 8, 1), 128, 0, s>>>(g_q, g_kc, g_vc, g_kvpos, g_po, g_pm, g_pl, l, 1818); }, "attn_split (seq=300)", 0);
    t += bench24([&](int l){ flash_decode_reduce_kernel<<<dim3(20, 1), 64, 0, s>>>(g_po, g_pm, g_pl, g_h); }, "attn_reduce", 0);
    t += bench24([&](int l){ gemv_k1280_kernel<1, 2, 1><<<640, 64, 0, s>>>(g_w_proj + (size_t)l*1280*1280, g_b, g_h, g_x, 1280, nullptr, nullptr, nullptr, 0, 0); }, "attn_proj (1280x1280)", 3.2768);
    t += bench24([&](int l){ layernorm_1280_kernel<false><<<1, 160, 0, s>>>(g_x, g_b, g_b, g_h, 1, 1e-5f); }, "ln_2 (x24)", 0);
    t += bench24([&](int l){ gemv_k1280_kernel<1, 4, 2><<<1280, 128, 0, s>>>(g_w_fc + (size_t)l*5120*1280, g_b, g_h, g_ff, 5120, nullptr, nullptr, nullptr, 0, 0); }, "mlp_fc (5120x1280)", 13.1072);
    t += bench24([&](int l){ gemv_k5120_resid_kernel<1><<<1280, 128, 0, s>>>(g_w_mproj + (size_t)l*1280*5120, g_b, g_ff, g_x); }, "mlp_proj (1280x5120)", 13.1072);
    float t_lm = bench(run_lm<1>, s);
    float t_samp = bench(run_samp<1>, s);
    printf("  %-22s: %6.1f us/tok | eff_BW = %6.1f GB/s\n", "lm_head (8200x1280)", t_lm, (20.992/1000.0)/(t_lm/1e6));
    printf("  %-22s: %6.1f us/tok\n", "sample_and_embed", t_samp);
    printf("  TOTAL SUM (B=1)       : %6.1f us/tok (%.1f tok/s)\n\n", t + t_lm + t_samp, 1e6f / (t + t_lm + t_samp));

    printf("=== COLD-DRAM 24-LAYER BREAKDOWN (B=4, seq=300) ===\n");
    float t4 = 0;
    t4 += bench24([&](int l){ layernorm_1280_kernel<false><<<4, 160, 0, s>>>(g_x, g_b, g_b, g_h, 4, 1e-5f); }, "ln_1 (x24)", 0);
    t4 += bench24([&](int l){ gemv_k1280_kernel<4, 4, 4><<<960, 128, 0, s>>>(g_w_qkv + (size_t)l*3840*1280, g_b, g_h, g_q, 3840, g_kc, g_vc, g_kvpos, l, 1818); }, "c_attn (3840x1280)", 9.8304);
    t4 += bench24([&](int l){ flash_decode_split_kernel<<<dim3(20, 8, 4), 128, 0, s>>>(g_q, g_kc, g_vc, g_kvpos, g_po, g_pm, g_pl, l, 1818); }, "attn_split (seq=300)", 0);
    t4 += bench24([&](int l){ flash_decode_reduce_kernel<<<dim3(20, 4), 64, 0, s>>>(g_po, g_pm, g_pl, g_h); }, "attn_reduce", 0);
    t4 += bench24([&](int l){ gemv_k1280_kernel<4, 2, 1><<<640, 64, 0, s>>>(g_w_proj + (size_t)l*1280*1280, g_b, g_h, g_x, 1280, nullptr, nullptr, nullptr, 0, 0); }, "attn_proj (1280x1280)", 3.2768);
    t4 += bench24([&](int l){ layernorm_1280_kernel<false><<<4, 160, 0, s>>>(g_x, g_b, g_b, g_h, 4, 1e-5f); }, "ln_2 (x24)", 0);
    t4 += bench24([&](int l){ gemv_k1280_kernel<4, 4, 2><<<1280, 128, 0, s>>>(g_w_fc + (size_t)l*5120*1280, g_b, g_h, g_ff, 5120, nullptr, nullptr, nullptr, 0, 0); }, "mlp_fc (5120x1280)", 13.1072);
    t4 += bench24([&](int l){ gemv_k5120_resid_kernel<4><<<1280, 128, 0, s>>>(g_w_mproj + (size_t)l*1280*5120, g_b, g_ff, g_x); }, "mlp_proj (1280x5120)", 13.1072);
    float t4_lm = bench(run_lm<4>, s);
    float t4_samp = bench(run_samp<4>, s);
    printf("  %-22s: %6.1f us/tok | eff_BW = %6.1f GB/s\n", "lm_head (8200x1280)", t4_lm, (20.992/1000.0)/(t4_lm/1e6));
    printf("  %-22s: %6.1f us/tok\n", "sample_and_embed", t4_samp);
    printf("  TOTAL SUM (B=4)       : %6.1f us/tok (agg %.1f tok/s)\n", t4 + t4_lm + t4_samp, 4e6f / (t4 + t4_lm + t4_samp));
    return 0;
}
