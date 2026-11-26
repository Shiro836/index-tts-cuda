#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include "gpt_engine.h"

namespace indextts_gpt {

constexpr int D_MODEL = GPT_D_MODEL;       // 1280
constexpr int N_HEADS = GPT_N_HEADS;       // 20
constexpr int HEAD_DIM = GPT_HEAD_DIM;     // 64
constexpr int D_FF = GPT_D_FF;             // 5120
constexpr int VOCAB_SIZE = GPT_VOCAB_SIZE; // 8194
constexpr int VOCAB_PAD = GPT_VOCAB_PAD;   // 8200
constexpr int MAX_POS = GPT_MAX_POS;       // 1818
constexpr int MASK_WORDS = (VOCAB_SIZE + 31) / 32; // 257
constexpr int DECODE_SPLITS = 16;          // 20 heads * 16 = 320 blocks (~2 waves on 170 SMs)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float gelu_new_f(float x) {
    constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kBeta  = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

// ============================================================================
// 1. LayerNorm: 5-warp for prefill/gpt_forward, 1-warp zero-smem for decode!
// ============================================================================
template <bool OUT_FP32>
__global__ void __launch_bounds__(160) layernorm_1280_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ b,
    void* __restrict__ out,
    int rows,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* x_row = x + (size_t)row * D_MODEL;
    int tid = threadIdx.x; // 0..159
    uint4 xv4 = __ldg(reinterpret_cast<const uint4*>(x_row) + tid);
    const half2* xh2 = reinterpret_cast<const half2*>(&xv4);

    float vals[8];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(xh2[i]);
        vals[2 * i]     = f.x;
        vals[2 * i + 1] = f.y;
        sum += f.x + f.y;
        sum_sq += f.x * f.x + f.y * f.y;
    }

    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float s_sum[5];
    __shared__ float s_sq[5];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) { s_sum[warp] = sum; s_sq[warp] = sum_sq; }
    __syncthreads();

    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float total_sum = s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3] + s_sum[4];
        float total_sq  = s_sq[0]  + s_sq[1]  + s_sq[2]  + s_sq[3]  + s_sq[4];
        float mean = total_sum * (1.0f / (float)D_MODEL);
        float var = total_sq * (1.0f / (float)D_MODEL) - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(fmaxf(var, 0.0f) + eps);
    }
    __syncthreads();

    float mean = s_mean, rstd = s_rstd;
    uint4 wv4 = __ldg(reinterpret_cast<const uint4*>(w) + tid);
    uint4 bv4 = __ldg(reinterpret_cast<const uint4*>(b) + tid);
    const half2* wh2 = reinterpret_cast<const half2*>(&wv4);
    const half2* bh2 = reinterpret_cast<const half2*>(&bv4);

    if constexpr (OUT_FP32) {
        float* out_row = reinterpret_cast<float*>(out) + (size_t)row * D_MODEL + tid * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 wf = __half22float2(wh2[i]);
            float2 bf = __half22float2(bh2[i]);
            out_row[2 * i]     = (vals[2 * i]     - mean) * rstd * wf.x + bf.x;
            out_row[2 * i + 1] = (vals[2 * i + 1] - mean) * rstd * wf.y + bf.y;
        }
    } else {
        uint4 ov4;
        half2* oh2 = reinterpret_cast<half2*>(&ov4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 wf = __half22float2(wh2[i]);
            float2 bf = __half22float2(bh2[i]);
            oh2[i] = __floats2half2_rn(
                (vals[2 * i]     - mean) * rstd * wf.x + bf.x,
                (vals[2 * i + 1] - mean) * rstd * wf.y + bf.y
            );
        }
        reinterpret_cast<uint4*>(reinterpret_cast<half*>(out) + (size_t)row * D_MODEL)[tid] = ov4;
    }
}

// 1-warp zero-smem zero-barrier LayerNorm for decode (~1.02 us/call, 100% deterministic)
__global__ void __launch_bounds__(32) layernorm_1280_1warp_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ b,
    half* __restrict__ out
) {
    int bi = blockIdx.x;
    int lane = threadIdx.x; // 0..31
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)bi * D_MODEL);

    uint4 xv[5];
    float sum = 0.0f, sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        xv[it] = __ldg(x4 + lane + it * 32);
        const half2* xh = reinterpret_cast<const half2*>(&xv[it]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __half22float2(xh[i]);
            sum += f.x + f.y;
            sq  += f.x * f.x + f.y * f.y;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
        sq  += __shfl_xor_sync(0xffffffff, sq, off);
    }

    float mean = sum * (1.0f / (float)D_MODEL);
    float rstd = rsqrtf(fmaxf(sq * (1.0f / (float)D_MODEL) - mean * mean, 0.0f) + 1e-5f);
    const uint4* w4 = reinterpret_cast<const uint4*>(w);
    const uint4* b4 = reinterpret_cast<const uint4*>(b);
    uint4* o4 = reinterpret_cast<uint4*>(out + (size_t)bi * D_MODEL);

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int col = lane + it * 32;
        uint4 wv = __ldg(w4 + col);
        uint4 bv = __ldg(b4 + col);
        const half2* xh = reinterpret_cast<const half2*>(&xv[it]);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        const half2* bh = reinterpret_cast<const half2*>(&bv);
        uint4 ov;
        half2* oh = reinterpret_cast<half2*>(&ov);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 xf = __half22float2(xh[i]);
            float2 wf = __half22float2(wh[i]);
            float2 bf = __half22float2(bh[i]);
            oh[i] = __floats2half2_rn(
                (xf.x - mean) * rstd * wf.x + bf.x,
                (xf.y - mean) * rstd * wf.y + bf.y
            );
        }
        o4[col] = ov;
    }
}

// 1-warp fused double LayerNorm (ln_f -> final_norm)
__global__ void __launch_bounds__(32) double_layernorm_1280_1warp_kernel(
    const half* __restrict__ x,
    const half* __restrict__ ln1_w,
    const half* __restrict__ ln1_b,
    const half* __restrict__ ln2_w,
    const half* __restrict__ ln2_b,
    half* __restrict__ out
) {
    int bi = blockIdx.x;
    int lane = threadIdx.x;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)bi * D_MODEL);

    uint4 xv[5];
    float sum1 = 0.0f, sq1 = 0.0f;
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        xv[it] = __ldg(x4 + lane + it * 32);
        const half2* xh = reinterpret_cast<const half2*>(&xv[it]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __half22float2(xh[i]);
            sum1 += f.x + f.y;
            sq1  += f.x * f.x + f.y * f.y;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum1 += __shfl_xor_sync(0xffffffff, sum1, off);
        sq1  += __shfl_xor_sync(0xffffffff, sq1, off);
    }
    float m1 = sum1 * (1.0f / (float)D_MODEL);
    float r1 = rsqrtf(fmaxf(sq1 * (1.0f / (float)D_MODEL) - m1 * m1, 0.0f) + 1e-5f);

    const uint4* w1_4 = reinterpret_cast<const uint4*>(ln1_w);
    const uint4* b1_4 = reinterpret_cast<const uint4*>(ln1_b);
    float sum2 = 0.0f, sq2 = 0.0f;
    uint4 yv[5];
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int col = lane + it * 32;
        uint4 wv = __ldg(w1_4 + col), bv = __ldg(b1_4 + col);
        const half2* xh = reinterpret_cast<const half2*>(&xv[it]);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        const half2* bh = reinterpret_cast<const half2*>(&bv);
        half2* yh = reinterpret_cast<half2*>(&yv[it]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 xf = __half22float2(xh[i]);
            float2 wf = __half22float2(wh[i]);
            float2 bf = __half22float2(bh[i]);
            yh[i] = __floats2half2_rn(
                (xf.x - m1) * r1 * wf.x + bf.x,
                (xf.y - m1) * r1 * wf.y + bf.y
            );
            float2 yf = __half22float2(yh[i]);
            sum2 += yf.x + yf.y;
            sq2  += yf.x * yf.x + yf.y * yf.y;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum2 += __shfl_xor_sync(0xffffffff, sum2, off);
        sq2  += __shfl_xor_sync(0xffffffff, sq2, off);
    }
    float m2 = sum2 * (1.0f / (float)D_MODEL);
    float r2 = rsqrtf(fmaxf(sq2 * (1.0f / (float)D_MODEL) - m2 * m2, 0.0f) + 1e-5f);

    const uint4* w2_4 = reinterpret_cast<const uint4*>(ln2_w);
    const uint4* b2_4 = reinterpret_cast<const uint4*>(ln2_b);
    uint4* o4 = reinterpret_cast<uint4*>(out + (size_t)bi * D_MODEL);
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int col = lane + it * 32;
        uint4 wv = __ldg(w2_4 + col), bv = __ldg(b2_4 + col);
        const half2* yh = reinterpret_cast<const half2*>(&yv[it]);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        const half2* bh = reinterpret_cast<const half2*>(&bv);
        uint4 ov;
        half2* oh = reinterpret_cast<half2*>(&ov);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 yf = __half22float2(yh[i]);
            float2 wf = __half22float2(wh[i]);
            float2 bf = __half22float2(bh[i]);
            oh[i] = __floats2half2_rn(
                (yf.x - m2) * r2 * wf.x + bf.x,
                (yf.y - m2) * r2 * wf.y + bf.y
            );
        }
        o4[col] = ov;
    }
}

// ============================================================================
// 2. Zero-smem Multi-Vector FP16 GEMV for K=1280 (c_attn, mlp.c_fc, lm_head)
//    EPILOGUE:
//      2 = bias + gelu_new -> FP16 (mlp.c_fc)
//      3 = bias -> FP32 logits (lm_head)
//      4 = c_attn fused Q (pre-scaled by 0.125) + KV cache write
// ============================================================================
template <int B, int WARPS_PER_BLOCK, int EPILOGUE>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32) gemv_k1280_v7_kernel(
    const half* __restrict__ W,
    const half* __restrict__ bias,
    const half* __restrict__ X,
    void* __restrict__ out,
    int N,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const int* __restrict__ d_kv_pos,
    int layer_idx,
    int max_seq_len
) {
    constexpr int K_VEC = D_MODEL / 8; // 160 uint4s
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= N) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * D_MODEL);
    const uint4* x4    = reinterpret_cast<const uint4*>(X);

    half2 acc0[B], acc1[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        acc0[b] = __float2half2_rn(0.0f);
        acc1[b] = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int iter = 0; iter < 5; ++iter) {
        int col = lane + iter * 32;
        uint4 wv = __ldg(w_row + col);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            uint4 xv = __ldg(x4 + b * K_VEC + col);
            const half2* xh = reinterpret_cast<const half2*>(&xv);
            acc0[b] = __hfma2(wh[0], xh[0], acc0[b]);
            acc1[b] = __hfma2(wh[1], xh[1], acc1[b]);
            acc0[b] = __hfma2(wh[2], xh[2], acc0[b]);
            acc1[b] = __hfma2(wh[3], xh[3], acc1[b]);
        }
    }

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        float2 f0 = __half22float2(acc0[b]);
        float2 f1 = __half22float2(acc1[b]);
        acc[b] = warp_reduce_sum((f0.x + f0.y) + (f1.x + f1.y));
    }

    if (lane == 0) {
        float bv = __half2float(__ldg(bias + row));
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            float val = acc[b] + bv;
            if constexpr (EPILOGUE == 2) {
                reinterpret_cast<half*>(out)[(size_t)b * N + row] = __float2half(gelu_new_f(val));
            } else if constexpr (EPILOGUE == 3) {
                if (row < VOCAB_SIZE) {
                    reinterpret_cast<float*>(out)[(size_t)b * VOCAB_SIZE + row] = val;
                }
            } else if constexpr (EPILOGUE == 4) {
                if (row < D_MODEL) {
                    reinterpret_cast<half*>(out)[(size_t)b * D_MODEL + row] = __float2half(val);
                } else {
                    int pos = d_kv_pos[b];
                    if (pos < max_seq_len) {
                        int rem = row - D_MODEL;
                        bool is_v = (rem >= D_MODEL);
                        int off = is_v ? (rem - D_MODEL) : rem;
                        int h = off >> 6;
                        int d = off & 63;
                        size_t kv_idx = ((((size_t)b * GPT_N_LAYERS + layer_idx) * N_HEADS + h) * max_seq_len + pos) * HEAD_DIM + d;
                        if (is_v) v_cache[kv_idx] = __float2half(val);
                        else      k_cache[kv_idx] = __float2half(val);
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2b. Zero-smem WPB=2 attn.c_proj (1280x1280, 640 blocks x 64T, 0 barriers!)
// ============================================================================
template <int B>
__global__ void __launch_bounds__(64) gemv_attn_proj_v7_kernel(
    const half* __restrict__ W,
    const half* __restrict__ bias,
    const half* __restrict__ X,
    half* __restrict__ resid
) {
    constexpr int K_VEC = D_MODEL / 8; // 160 uint4s
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int row = blockIdx.x * 2 + warp;
    if (row >= D_MODEL) return;

    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * D_MODEL);
    const uint4* x4 = reinterpret_cast<const uint4*>(X);

    half2 acc0[B], acc1[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        acc0[b] = __float2half2_rn(0.0f);
        acc1[b] = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int col = lane + it * 32;
        uint4 wv = __ldg(w_row + col);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            uint4 xv = __ldg(x4 + b * K_VEC + col);
            const half2* xh = reinterpret_cast<const half2*>(&xv);
            acc0[b] = __hfma2(wh[0], xh[0], acc0[b]);
            acc1[b] = __hfma2(wh[1], xh[1], acc1[b]);
            acc0[b] = __hfma2(wh[2], xh[2], acc0[b]);
            acc1[b] = __hfma2(wh[3], xh[3], acc1[b]);
        }
    }

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        float2 f0 = __half22float2(acc0[b]);
        float2 f1 = __half22float2(acc1[b]);
        acc[b] = warp_reduce_sum((f0.x + f0.y) + (f1.x + f1.y));
    }

    if (lane == 0) {
        float bv = __half2float(__ldg(bias + row));
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            half* dst = resid + (size_t)b * D_MODEL + row;
            *dst = __float2half(acc[b] + bv + __half2float(*dst));
        }
    }
}

// ============================================================================
// 3. 5-Warp mlp.c_proj (K=5120, N=1280, 1280 blocks x 160T)
// ============================================================================
template <int B>
__global__ void __launch_bounds__(160) gemv_k5120_v7_resid_kernel(
    const half* __restrict__ W,
    const half* __restrict__ bias,
    const half* __restrict__ X,
    half* __restrict__ resid
) {
    constexpr int K_VEC = D_FF / 8; // 640 uint4s -> 4 iters of 160
    int row = blockIdx.x;
    if (row >= D_MODEL) return;

    int tid = threadIdx.x; // 0..159 (5 warps)
    int warp = tid >> 5;
    int lane = tid & 31;

    const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)row * D_FF);
    const uint4* x4 = reinterpret_cast<const uint4*>(X);

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int col = tid + it * 160;
        uint4 wv = __ldg(w_row + col);
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            uint4 xv = __ldg(x4 + b * K_VEC + col);
            const half2* xh = reinterpret_cast<const half2*>(&xv);
            half2 p0 = __hmul2(wh[0], xh[0]);
            half2 p1 = __hmul2(wh[1], xh[1]);
            p0 = __hfma2(wh[2], xh[2], p0);
            p1 = __hfma2(wh[3], xh[3], p1);
            float2 f0 = __half22float2(p0);
            float2 f1 = __half22float2(p1);
            acc[b] += (f0.x + f0.y) + (f1.x + f1.y);
        }
    }

    __shared__ float s_red[5][B];
    #pragma unroll
    for (int b = 0; b < B; ++b) {
        float v = warp_reduce_sum(acc[b]);
        if (lane == 0) s_red[warp][b] = v;
    }
    __syncthreads();

    if (tid == 0) {
        float bv = __half2float(__ldg(bias + row));
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            float total = s_red[0][b] + s_red[1][b] + s_red[2][b] + s_red[3][b] + s_red[4][b] + bv;
            half* dst = resid + (size_t)b * D_MODEL + row;
            *dst = __float2half(total + __half2float(*dst));
        }
    }
}

// ============================================================================
// 4. Split-K=16 2-Warp (64T) Flash-Decode Attention (320 blocks at B=1!)
// ============================================================================
__global__ void __launch_bounds__(64) flash_decode_split_v7_kernel(
    const half* __restrict__ q_buf,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ d_kv_pos,
    float* __restrict__ partial_out,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int layer_idx,
    int max_seq_len
) {
    int h = blockIdx.x;
    int s = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x; // 0..63 (2 warps = 8 groups of 8 lanes)
    int grp = tid >> 3;    // 0..7
    int d8  = tid & 7;     // 0..7

    int seq_len = d_kv_pos[b] + 1;
    if (seq_len > max_seq_len) seq_len = max_seq_len;

    int chunk = (seq_len + DECODE_SPLITS - 1) / DECODE_SPLITS;
    int start = s * chunk;
    int end = min(start + chunk, seq_len);

    int meta_idx = (b * N_HEADS + h) * DECODE_SPLITS + s;
    float* out_ptr = partial_out + (size_t)meta_idx * HEAD_DIM;

    if (start >= end) {
        if (tid == 0) { partial_m[meta_idx] = -INFINITY; partial_l[meta_idx] = 0.0f; }
        return;
    }

    const uint4* q_head = reinterpret_cast<const uint4*>(q_buf + (size_t)b * D_MODEL + h * HEAD_DIM);
    uint4 qv = __ldg(q_head + d8);
    const half2* qh_raw = reinterpret_cast<const half2*>(&qv);
    half2 sc_h = __float2half2_rn(0.125f);
    half2 qh[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qh[i] = __hmul2(qh_raw[i], sc_h);

    size_t head_base = (((size_t)b * GPT_N_LAYERS + layer_idx) * N_HEADS + h) * max_seq_len;
    const uint4* k_head = reinterpret_cast<const uint4*>(k_cache + head_base * HEAD_DIM);
    const uint4* v_head = reinterpret_cast<const uint4*>(v_cache + head_base * HEAD_DIM);

    float m_loc = -INFINITY, l_loc = 0.0f;
    float o_loc[8] = {0,0,0,0,0,0,0,0};
    unsigned sub_mask = 0xffu << (tid & 24);

    for (int pos = start + grp; pos < end; pos += 8) {
        uint4 kv = __ldg(k_head + (size_t)pos * 8 + d8);
        const half2* kh = reinterpret_cast<const half2*>(&kv);
        half2 d0 = __hmul2(qh[0], kh[0]);
        half2 d1 = __hmul2(qh[1], kh[1]);
        d0 = __hfma2(qh[2], kh[2], d0);
        d1 = __hfma2(qh[3], kh[3], d1);
        float2 f0 = __half22float2(d0), f1 = __half22float2(d1);
        float dot = (f0.x + f0.y) + (f1.x + f1.y);
        dot += __shfl_xor_sync(sub_mask, dot, 4);
        dot += __shfl_xor_sync(sub_mask, dot, 2);
        dot += __shfl_xor_sync(sub_mask, dot, 1);

        float m_new = fmaxf(m_loc, dot);
        float alpha = (m_loc == -INFINITY) ? 0.0f : __expf(m_loc - m_new);
        float p = __expf(dot - m_new);
        l_loc = l_loc * alpha + p;
        m_loc = m_new;

        uint4 vv = __ldg(v_head + (size_t)pos * 8 + d8);
        const half2* vh = reinterpret_cast<const half2*>(&vv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 vf = __half22float2(vh[i]);
            o_loc[2*i]   = o_loc[2*i]   * alpha + p * vf.x;
            o_loc[2*i+1] = o_loc[2*i+1] * alpha + p * vf.y;
        }
    }

    // In-warp reduction across 4 groups within each warp
    #pragma unroll
    for (int off = 16; off >= 8; off >>= 1) {
        float om = __shfl_xor_sync(0xffffffff, m_loc, off);
        float ol = __shfl_xor_sync(0xffffffff, l_loc, off);
        float nm = fmaxf(m_loc, om);
        float s1 = (m_loc == -INFINITY) ? 0.0f : __expf(m_loc - nm);
        float s2 = (om    == -INFINITY) ? 0.0f : __expf(om    - nm);
        l_loc = l_loc * s1 + ol * s2;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float oo = __shfl_xor_sync(0xffffffff, o_loc[i], off);
            o_loc[i] = o_loc[i] * s1 + oo * s2;
        }
        m_loc = nm;
    }

    // Combine the 2 warps in tiny shared memory (528 bytes)
    __shared__ float s_m[2], s_l[2], s_o[2][HEAD_DIM];
    int warp = tid >> 5;
    if ((tid & 31) < 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) s_o[warp][d8 * 8 + i] = o_loc[i];
        if (d8 == 0) { s_m[warp] = m_loc; s_l[warp] = l_loc; }
    }
    __syncthreads();

    float gm = fmaxf(s_m[0], s_m[1]);
    float sc0 = (s_l[0] > 0.0f) ? __expf(s_m[0] - gm) : 0.0f;
    float sc1 = (s_l[1] > 0.0f) ? __expf(s_m[1] - gm) : 0.0f;
    out_ptr[tid] = s_o[0][tid] * sc0 + s_o[1][tid] * sc1;
    if (tid == 0) {
        partial_m[meta_idx] = gm;
        partial_l[meta_idx] = s_l[0] * sc0 + s_l[1] * sc1;
    }
}

__global__ void __launch_bounds__(64) flash_decode_reduce_v7_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    half* __restrict__ attn_out
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int d = threadIdx.x; // 0..63

    int base = (b * N_HEADS + h) * DECODE_SPLITS;
    float gm = -INFINITY;
    #pragma unroll
    for (int s = 0; s < DECODE_SPLITS; ++s) gm = fmaxf(gm, partial_m[base + s]);
    float gl = 0.0f, go = 0.0f;
    #pragma unroll
    for (int s = 0; s < DECODE_SPLITS; ++s) {
        float ls = partial_l[base + s];
        if (ls > 0.0f) {
            float sc = __expf(partial_m[base + s] - gm);
            gl += ls * sc;
            go += partial_out[(size_t)(base + s) * HEAD_DIM + d] * sc;
        }
    }
    attn_out[(size_t)b * D_MODEL + h * HEAD_DIM + d] = __float2half(gl > 0.0f ? (go / gl) : 0.0f);
}

// ============================================================================
// 5. Prefill / gpt_forward helper kernels
// ============================================================================
__global__ void add_bias_gelu_kernel(half* __restrict__ x, const half* __restrict__ bias, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * D_FF;
    if (idx >= total) return;
    int col = idx % D_FF;
    float v = __half2float(x[idx]) + __half2float(bias[col]);
    x[idx] = __float2half(gelu_new_f(v));
}

__global__ void add_bias_resid_kernel(half* __restrict__ resid, const half* __restrict__ proj, const half* __restrict__ bias, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * D_MODEL;
    if (idx >= total) return;
    int col = idx % D_MODEL;
    float v = __half2float(resid[idx]) + __half2float(proj[idx]) + __half2float(bias[col]);
    resid[idx] = __float2half(v);
}

__global__ void unpack_qkv_prefill_kernel(
    const half* __restrict__ qkv, const half* __restrict__ bias,
    half* __restrict__ q_out, half* __restrict__ k_out, half* __restrict__ v_out,
    int T, int kv_stride_seq
) {
    int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (t >= T || h >= N_HEADS) return;
    int base_col = h * HEAD_DIM + d;
    const half* row = qkv + (size_t)t * (3 * D_MODEL);
    q_out[((size_t)h * T + t) * HEAD_DIM + d]             = __float2half(__half2float(row[base_col])               + __half2float(bias[base_col]));
    k_out[((size_t)h * kv_stride_seq + t) * HEAD_DIM + d] = __float2half(__half2float(row[D_MODEL + base_col])     + __half2float(bias[D_MODEL + base_col]));
    v_out[((size_t)h * kv_stride_seq + t) * HEAD_DIM + d] = __float2half(__half2float(row[2 * D_MODEL + base_col]) + __half2float(bias[2 * D_MODEL + base_col]));
}

__global__ void causal_flash_prefill_kernel(
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    half* __restrict__ O, int T, int kv_stride
) {
    constexpr int BR = 4, BC = 32;
    int q_tile = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;
    int q_pos = q_tile * BR + warp;
    bool q_valid = (q_pos < T);

    const half* q_head = Q + (size_t)h * T * HEAD_DIM;
    const half* k_head = K + (size_t)h * kv_stride * HEAD_DIM;
    const half* v_head = V + (size_t)h * kv_stride * HEAD_DIM;

    float q0 = 0.f, q1 = 0.f;
    if (q_valid) {
        float2 qf = __half22float2(reinterpret_cast<const half2*>(q_head + (size_t)q_pos * HEAD_DIM)[lane]);
        q0 = qf.x * 0.125f; q1 = qf.y * 0.125f;
    }
    float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
    __shared__ half2 s_K[BC][32], s_V[BC][32];

    int max_k = min(T, (q_tile + 1) * BR);
    for (int k_base = 0; k_base < max_k; k_base += BC) {
        #pragma unroll
        for (int idx = tid; idx < BC * 32; idx += 128) {
            int r = idx >> 5, c = idx & 31, kp = k_base + r;
            if (kp < max_k) {
                s_K[r][c] = reinterpret_cast<const half2*>(k_head + (size_t)kp * HEAD_DIM)[c];
                s_V[r][c] = reinterpret_cast<const half2*>(v_head + (size_t)kp * HEAD_DIM)[c];
            } else {
                s_K[r][c] = __float2half2_rn(0.0f);
                s_V[r][c] = __float2half2_rn(0.0f);
            }
        }
        __syncthreads();
        if (q_valid) {
            int k_limit = min(BC, q_pos - k_base + 1);
            for (int r = 0; r < k_limit; ++r) {
                float2 kf = __half22float2(s_K[r][lane]);
                float dot = warp_reduce_sum(q0 * kf.x + q1 * kf.y);
                dot = __shfl_sync(0xffffffff, dot, 0);
                float m_new = fmaxf(m, dot);
                float alpha = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
                float p = __expf(dot - m_new);
                l = l * alpha + p; m = m_new;
                float2 vf = __half22float2(s_V[r][lane]);
                o0 = o0 * alpha + p * vf.x;
                o1 = o1 * alpha + p * vf.y;
            }
        }
        __syncthreads();
    }
    if (q_valid) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        reinterpret_cast<half2*>(O + (size_t)q_pos * D_MODEL + h * HEAD_DIM)[lane] = __floats2half2_rn(o0 * inv_l, o1 * inv_l);
    }
}

// ============================================================================
// 6. 1024-Thread 12-Iter Register Binary-Search Top-K Sampler (~14 us!)
// ============================================================================
__device__ __forceinline__ uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void __launch_bounds__(1024) sample_and_embed_v7_kernel(
    const float* __restrict__ logits,
    const half* __restrict__ audio_emb,
    const half* __restrict__ pos_emb,
    uint32_t* __restrict__ seen_mask,
    uint64_t* __restrict__ rng_state,
    int* __restrict__ d_kv_pos,
    int* __restrict__ d_gen_step,
    int* __restrict__ d_finished,
    int* __restrict__ d_out_tokens,
    int* __restrict__ d_out_lens,
    const int* __restrict__ d_max_new,
    half* __restrict__ x_next,
    int out_stride,
    float inv_temperature,
    float top_p,
    int top_k,
    float rep_penalty,
    int stop_token_id
) {
    int b = blockIdx.x;
    int tid = threadIdx.x; // 0..1023
    if (d_finished[b]) return;

    __shared__ float s_logits[VOCAB_SIZE];
    __shared__ float s_warp_val[32];
    __shared__ int   s_warp_idx[32];
    __shared__ float s_warp_min[32];
    __shared__ int   s_warp_cnt[32];
    __shared__ float s_top_val[64];
    __shared__ int   s_top_idx[64];
    __shared__ float s_cand_val[256];
    __shared__ int   s_cand_idx[256];
    __shared__ int   s_cand_cnt;
    __shared__ float s_lo, s_hi;
    __shared__ int   s_chosen_tok;
    __shared__ int   s_new_pos;

    const float* row = logits + (size_t)b * VOCAB_SIZE;
    uint32_t* mask = seen_mask + (size_t)b * MASK_WORDS;

    float local_max = -INFINITY, local_min = INFINITY;
    int local_idx = -1;
    bool use_rep = (rep_penalty != 1.0f);
    bool greedy = (inv_temperature <= 0.0f);

    float r_val[9];
    int n_elem = 0;
    for (int i = tid; i < VOCAB_SIZE; i += 1024) {
        float v = __ldg(row + i);
        if (use_rep) {
            uint32_t word = mask[i >> 5];
            if ((word >> (i & 31)) & 1u) {
                v = (v > 0.0f) ? (v / rep_penalty) : (v * rep_penalty);
            }
        }
        if (!greedy) v *= inv_temperature;
        r_val[n_elem++] = v;
        if (v > local_max) { local_max = v; local_idx = i; }
        if (v < local_min) { local_min = v; }
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    int k = greedy ? 1 : max(1, min(top_k, 64));

    // Global max & min across 32 warps
    {
        float wv = local_max, wmn = local_min;
        int wi = local_idx;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, wv, off);
            int oi = __shfl_down_sync(0xffffffff, wi, off);
            float omn = __shfl_down_sync(0xffffffff, wmn, off);
            if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
            if (omn < wmn) wmn = omn;
        }
        if (lane == 0) { s_warp_val[warp] = wv; s_warp_idx[warp] = wi; s_warp_min[warp] = wmn; }
        __syncthreads();
        if (warp == 0) {
            float bv = s_warp_val[lane], bmn = s_warp_min[lane];
            int bi = s_warp_idx[lane];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, bv, off);
                int oi = __shfl_down_sync(0xffffffff, bi, off);
                float omn = __shfl_down_sync(0xffffffff, bmn, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                if (omn < bmn) bmn = omn;
            }
            if (lane == 0) {
                s_hi = bv; s_lo = bmn;
                s_top_val[0] = bv; s_top_idx[0] = bi;
                s_cand_cnt = 0;
            }
        }
        __syncthreads();
    }

    if (!greedy && k > 1) {
        float lo = s_lo, hi = s_hi;
        for (int iter = 0; iter < 12; ++iter) {
            float mid = 0.5f * (lo + hi);
            int cnt = 0;
            #pragma unroll
            for (int e = 0; e < 8; ++e) cnt += (r_val[e] >= mid);
            if (n_elem == 9) cnt += (r_val[8] >= mid);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
            if (lane == 0) s_warp_cnt[warp] = cnt;
            __syncthreads();
            if (warp == 0) {
                int tot = s_warp_cnt[lane];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) tot += __shfl_down_sync(0xffffffff, tot, off);
                if (lane == 0) s_warp_cnt[0] = tot;
            }
            __syncthreads();
            int total = s_warp_cnt[0];
            if (total >= k && total <= 160) { lo = mid; break; }
            if (total >= k) lo = mid;
            else hi = mid;
            __syncthreads();
        }

        for (int e = 0; e < n_elem; ++e) {
            if (r_val[e] >= lo) {
                int pos = atomicAdd(&s_cand_cnt, 1);
                if (pos < 256) {
                    s_cand_val[pos] = r_val[e];
                    s_cand_idx[pos] = tid + e * 1024;
                }
            }
        }
        __syncthreads();

        int cnt = s_cand_cnt;
        if (cnt >= k && cnt <= 256) {
            if (tid < 32) {
                for (int r = 0; r < k; ++r) {
                    float bv = -INFINITY; int bi = 0; int bpos = -1;
                    for (int c = tid; c < cnt; c += 32) {
                        float v = s_cand_val[c]; int idx = s_cand_idx[c];
                        if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; bpos = c; }
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        float ov = __shfl_down_sync(0xffffffff, bv, off);
                        int oi = __shfl_down_sync(0xffffffff, bi, off);
                        int op = __shfl_down_sync(0xffffffff, bpos, off);
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bpos = op; }
                    }
                    if (tid == 0) {
                        s_top_val[r] = bv; s_top_idx[r] = bi;
                        if (bpos >= 0) s_cand_val[bpos] = -INFINITY;
                    }
                    __syncwarp();
                }
            }
            __syncthreads();
        } else {
            // Exact fallback for >256 identical float ties
            for (int e = 0; e < n_elem; ++e) s_logits[tid + e * 1024] = r_val[e];
            __syncthreads();
            if (tid == 0) s_logits[s_top_idx[0]] = -INFINITY;
            __syncthreads();
            if ((s_top_idx[0] & 1023) == tid) {
                float lm = -INFINITY; int li = -1;
                for (int i = tid; i < VOCAB_SIZE; i += 1024) {
                    float v = s_logits[i]; if (v > lm) { lm = v; li = i; }
                }
                local_max = lm; local_idx = li;
            }
            for (int r = 1; r < k; ++r) {
                float wv = local_max; int wi = local_idx;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, wv, off);
                    int oi = __shfl_down_sync(0xffffffff, wi, off);
                    if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; }
                }
                if (lane == 0) { s_warp_val[warp] = wv; s_warp_idx[warp] = wi; }
                __syncthreads();
                if (warp == 0) {
                    float bv = s_warp_val[lane]; int bi = s_warp_idx[lane];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        float ov = __shfl_down_sync(0xffffffff, bv, off);
                        int oi = __shfl_down_sync(0xffffffff, bi, off);
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                    }
                    if (lane == 0) { s_top_val[r] = bv; s_top_idx[r] = bi; s_logits[bi] = -INFINITY; }
                }
                __syncthreads();
                if ((s_top_idx[r] & 1023) == tid && r + 1 < k) {
                    float lm = -INFINITY; int li = -1;
                    for (int i = tid; i < VOCAB_SIZE; i += 1024) {
                        float v = s_logits[i]; if (v > lm) { lm = v; li = i; }
                    }
                    local_max = lm; local_idx = li;
                }
            }
        }
    }

    if (tid == 0) {
        int chosen = s_top_idx[0];
        if (!greedy && k > 1) {
            float max_v = s_top_val[0];
            float probs[64];
            float sum_p = 0.0f;
            for (int r = 0; r < k; ++r) { probs[r] = __expf(s_top_val[r] - max_v); sum_p += probs[r]; }
            float inv_sum = 1.0f / sum_p;
            float cum = 0.0f; int keep = 0;
            for (int r = 0; r < k; ++r) { probs[r] *= inv_sum; cum += probs[r]; keep = r + 1; if (cum >= top_p) break; }
            uint64_t rnd = splitmix64_next(rng_state[b]);
            float u = (float)(rnd >> 40) * (1.0f / 16777216.0f) * cum;
            float acc = 0.0f;
            chosen = s_top_idx[keep - 1];
            for (int r = 0; r < keep; ++r) { acc += probs[r]; if (u < acc) { chosen = s_top_idx[r]; break; } }
        }
        s_chosen_tok = chosen;

        mask[chosen >> 5] |= (1u << (chosen & 31));
        int step = d_gen_step[b];
        d_out_tokens[(size_t)b * out_stride + step] = chosen;
        step += 1;
        d_gen_step[b] = step;
        d_out_lens[b] = step;
        d_kv_pos[b] += 1;
        s_new_pos = step;
        if (chosen == stop_token_id || step >= d_max_new[0]) d_finished[b] = 1;
    }
    __syncthreads();

    int tok = s_chosen_tok;
    int pos = s_new_pos;
    if (pos >= MAX_POS) pos = MAX_POS - 1;

    const half* tok_row = audio_emb + (size_t)tok * D_MODEL;
    const half* pos_row = pos_emb   + (size_t)pos * D_MODEL;
    half* dst_row = x_next + (size_t)b * D_MODEL;
    for (int d = tid; d < D_MODEL; d += 1024) {
        dst_row[d] = __float2half(__half2float(__ldg(tok_row + d)) + __half2float(__ldg(pos_row + d)));
    }
}

// ============================================================================
// 7. sm_89-compatible TF32 Tensor-Core FlashAttention-2 (FP32 I/O, D=64, unmasked)
//    Used by s2mel DiT to replace PyTorch's scalar-FP32 mem_efficient attention (~2.9x faster)
// ============================================================================
__device__ __forceinline__ uint32_t f32_to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__global__ void __launch_bounds__(128) tf32_flash_attn_d64_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T,
    float scale_log2e
) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int D = 64;
    constexpr int STRIDE = 68;

    int q_block = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int gid = lane >> 2;
    int tin = lane & 3;

    const float* q_base = Q + (size_t)bh * T * D;
    const float* k_base = K + (size_t)bh * T * D;
    const float* v_base = V + (size_t)bh * T * D;
    float*       o_base = O + (size_t)bh * T * D;

    int q_row0 = q_block * BM + warp * 16 + gid;
    int q_row1 = q_row0 + 8;

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        int c0 = kt * 8 + tin;
        int c1 = c0 + 4;
        float q00 = (q_row0 < T) ? q_base[(size_t)q_row0 * D + c0] * scale_log2e : 0.0f;
        float q10 = (q_row1 < T) ? q_base[(size_t)q_row1 * D + c0] * scale_log2e : 0.0f;
        float q01 = (q_row0 < T) ? q_base[(size_t)q_row0 * D + c1] * scale_log2e : 0.0f;
        float q11 = (q_row1 < T) ? q_base[(size_t)q_row1 * D + c1] * scale_log2e : 0.0f;
        q_reg[kt][0] = f32_to_tf32(q00);
        q_reg[kt][1] = f32_to_tf32(q10);
        q_reg[kt][2] = f32_to_tf32(q01);
        q_reg[kt][3] = f32_to_tf32(q11);
    }

    __shared__ uint32_t s_K[BN][STRIDE];
    __shared__ uint32_t s_V[BN][STRIDE];

    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float o_acc[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.0f;
    }

    int base_lane = gid * 4;
    int src_lo = base_lane + (tin >> 1);
    int src_hi = base_lane + 2 + (tin >> 1);
    bool odd_t = (tin & 1) != 0;

    for (int kv_start = 0; kv_start < T; kv_start += BN) {
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int f4_idx = tid + it * 128;
            int r = f4_idx >> 4;
            int c4 = (f4_idx & 15) * 4;
            int k_row = kv_start + r;
            if (k_row < T) {
                float4 kv = __ldg(reinterpret_cast<const float4*>(k_base + (size_t)k_row * D + c4));
                float4 vv = __ldg(reinterpret_cast<const float4*>(v_base + (size_t)k_row * D + c4));
                s_K[r][c4 + 0] = f32_to_tf32(kv.x); s_K[r][c4 + 1] = f32_to_tf32(kv.y);
                s_K[r][c4 + 2] = f32_to_tf32(kv.z); s_K[r][c4 + 3] = f32_to_tf32(kv.w);
                s_V[r][c4 + 0] = f32_to_tf32(vv.x); s_V[r][c4 + 1] = f32_to_tf32(vv.y);
                s_V[r][c4 + 2] = f32_to_tf32(vv.z); s_V[r][c4 + 3] = f32_to_tf32(vv.w);
            } else {
                s_K[r][c4 + 0] = s_K[r][c4 + 1] = s_K[r][c4 + 2] = s_K[r][c4 + 3] = 0u;
                s_V[r][c4 + 0] = s_V[r][c4 + 1] = s_V[r][c4 + 2] = s_V[r][c4 + 3] = 0u;
            }
        }
        __syncthreads();

        float s_reg[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s_reg[nt][0] = s_reg[nt][1] = s_reg[nt][2] = s_reg[nt][3] = 0.0f;
            int k_r = nt * 8 + gid;
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                uint32_t b0 = s_K[k_r][kt * 8 + tin];
                uint32_t b1 = s_K[k_r][kt * 8 + tin + 4];
                mma_m16n8k8_tf32(
                    s_reg[nt][0], s_reg[nt][1], s_reg[nt][2], s_reg[nt][3],
                    q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                    b0, b1
                );
            }
        }

        if (kv_start + BN > T) {
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int c0 = kv_start + nt * 8 + 2 * tin;
                int c1 = c0 + 1;
                if (c0 >= T) { s_reg[nt][0] = -INFINITY; s_reg[nt][2] = -INFINITY; }
                if (c1 >= T) { s_reg[nt][1] = -INFINITY; s_reg[nt][3] = -INFINITY; }
            }
        }

        float bm0 = -INFINITY, bm1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            bm0 = fmaxf(bm0, fmaxf(s_reg[nt][0], s_reg[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(s_reg[nt][2], s_reg[nt][3]));
        }
        bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffff, bm0, 2));
        bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffff, bm0, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffff, bm1, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffff, bm1, 1));

        float nm0 = fmaxf(m0, bm0);
        float nm1 = fmaxf(m1, bm1);
        float alpha0 = (m0 == -INFINITY) ? 0.0f : exp2f(m0 - nm0);
        float alpha1 = (m1 == -INFINITY) ? 0.0f : exp2f(m1 - nm1);

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            o_acc[nt][0] *= alpha0; o_acc[nt][1] *= alpha0;
            o_acc[nt][2] *= alpha1; o_acc[nt][3] *= alpha1;
        }
        l0 *= alpha0;
        l1 *= alpha1;

        float bl0 = 0.0f, bl1 = 0.0f;
        uint32_t p_reg[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float p00 = exp2f(s_reg[nt][0] - nm0);
            float p01 = exp2f(s_reg[nt][1] - nm0);
            float p10 = exp2f(s_reg[nt][2] - nm1);
            float p11 = exp2f(s_reg[nt][3] - nm1);
            bl0 += p00 + p01;
            bl1 += p10 + p11;

            uint32_t u00 = f32_to_tf32(p00), u01 = f32_to_tf32(p01);
            uint32_t u10 = f32_to_tf32(p10), u11 = f32_to_tf32(p11);
            uint32_t lo0 = __shfl_sync(0xffffffff, u00, src_lo), lo1 = __shfl_sync(0xffffffff, u01, src_lo);
            uint32_t hi0 = __shfl_sync(0xffffffff, u00, src_hi), hi1 = __shfl_sync(0xffffffff, u01, src_hi);
            p_reg[nt][0] = odd_t ? lo1 : lo0;
            p_reg[nt][2] = odd_t ? hi1 : hi0;
            uint32_t lo2 = __shfl_sync(0xffffffff, u10, src_lo), lo3 = __shfl_sync(0xffffffff, u11, src_lo);
            uint32_t hi2 = __shfl_sync(0xffffffff, u10, src_hi), hi3 = __shfl_sync(0xffffffff, u11, src_hi);
            p_reg[nt][1] = odd_t ? lo3 : lo2;
            p_reg[nt][3] = odd_t ? hi3 : hi2;
        }
        bl0 += __shfl_xor_sync(0xffffffff, bl0, 2); bl0 += __shfl_xor_sync(0xffffffff, bl0, 1);
        bl1 += __shfl_xor_sync(0xffffffff, bl1, 2); bl1 += __shfl_xor_sync(0xffffffff, bl1, 1);
        l0 += bl0; l1 += bl1;
        m0 = nm0;  m1 = nm1;

        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            int vr0 = kt * 8 + tin;
            int vr1 = vr0 + 4;
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                int vc = nt * 8 + gid;
                uint32_t b0 = s_V[vr0][vc];
                uint32_t b1 = s_V[vr1][vc];
                mma_m16n8k8_tf32(
                    o_acc[nt][0], o_acc[nt][1], o_acc[nt][2], o_acc[nt][3],
                    p_reg[kt][0], p_reg[kt][1], p_reg[kt][2], p_reg[kt][3],
                    b0, b1
                );
            }
        }
        __syncthreads();
    }

    float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        int c0 = nt * 8 + 2 * tin;
        int c1 = c0 + 1;
        if (q_row0 < T) {
            o_base[(size_t)q_row0 * D + c0] = o_acc[nt][0] * inv_l0;
            o_base[(size_t)q_row0 * D + c1] = o_acc[nt][1] * inv_l0;
        }
        if (q_row1 < T) {
            o_base[(size_t)q_row1 * D + c0] = o_acc[nt][2] * inv_l1;
            o_base[(size_t)q_row1 * D + c1] = o_acc[nt][3] * inv_l1;
        }
    }
}

} // namespace indextts_gpt
