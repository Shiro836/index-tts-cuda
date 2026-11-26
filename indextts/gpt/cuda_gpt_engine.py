"""Python wrapper and async micro-batching scheduler for the custom sm_89 CUDA GPT-2 engine.

Replaces vLLM AsyncLLM (inference_speech) and PyTorch GPT2Model (forward) with a single
in-process C++/CUDA library holding both base_pristine and merged lora_step12000 weights.
"""
import asyncio
import ctypes
import os
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


def _find_library() -> str:
    env_path = os.environ.get("INDEXTTS_GPT_CUDA_SO")
    if env_path and os.path.isfile(env_path):
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "csrc", "gpt_engine", "libindextts_gpt.so"),
        os.path.join(os.getcwd(), "csrc", "gpt_engine", "libindextts_gpt.so"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"libindextts_gpt.so not found; build it with csrc/gpt_engine/build.sh (checked {candidates})"
    )


class _LibWrapper:
    def __init__(self, so_path: str):
        lib = ctypes.CDLL(so_path)
        self.lib = lib

        lib.gpt_engine_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.gpt_engine_create.restype = ctypes.c_void_p

        lib.gpt_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.gpt_engine_destroy.restype = None

        lib.gpt_engine_vram_bytes.argtypes = [ctypes.c_void_p]
        lib.gpt_engine_vram_bytes.restype = ctypes.c_size_t

        lib.gpt_engine_set_shared_top.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.gpt_engine_set_shared_top.restype = ctypes.c_int

        lib.gpt_engine_set_shared_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.gpt_engine_set_shared_layer.restype = ctypes.c_int

        lib.gpt_engine_set_model_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.gpt_engine_set_model_layer.restype = ctypes.c_int

        lib.gpt_engine_forward.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.gpt_engine_forward.restype = ctypes.c_int

        lib.gpt_engine_prefill_logits.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.gpt_engine_prefill_logits.restype = ctypes.c_int

        lib.gpt_engine_generate_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p,
        ]
        lib.gpt_engine_generate_batch.restype = ctypes.c_int

        lib.gpt_tf32_flash_attn_d64.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
        ]
        lib.gpt_tf32_flash_attn_d64.restype = ctypes.c_int


_TF32_FA_LIB = None


def _get_tf32_fa_lib():
    global _TF32_FA_LIB
    if _TF32_FA_LIB is None:
        _TF32_FA_LIB = _LibWrapper(_find_library()).lib
    return _TF32_FA_LIB


@torch.library.custom_op("indextts::tf32_sdpa_d64", mutates_args=())
def tf32_sdpa_d64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unmasked FP32-I/O TF32 Tensor-Core FlashAttention-2 for [B, H, T, 64] tensors."""
    B, H, T, D = q.shape
    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()
    out = torch.empty((B, H, T, D), dtype=qc.dtype, device=qc.device)
    stream_ptr = torch.cuda.current_stream(qc.device).cuda_stream
    rc = _get_tf32_fa_lib().gpt_tf32_flash_attn_d64(
        qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), out.data_ptr(),
        int(B * H), int(T), float(D ** -0.5), ctypes.c_void_p(stream_ptr),
    )
    if rc != 0:
        raise RuntimeError("gpt_tf32_flash_attn_d64 failed")
    return out


@tf32_sdpa_d64.register_fake
def _tf32_sdpa_d64_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


class CudaGptEngine:
    """In-process CUDA GPT-2 engine supporting base (model_id=0) and adapter (model_id=1)."""

    def __init__(
        self,
        model_dirs: Dict[int, str],
        max_batch: int = 4,
        max_seq_len: int = 1818,
        device: str = "cuda:0",
    ):
        self.so_path = _find_library()
        self._w = _LibWrapper(self.so_path)
        self.lib = self._w.lib
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.num_models = len(model_dirs)
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)

        self._handle = self.lib.gpt_engine_create(max_batch, max_seq_len, self.num_models)
        if not self._handle:
            raise RuntimeError("gpt_engine_create failed (OOM or invalid config)")

        # Separate locks: _gpu_lock serializes KV-cache decode, _fwd_lock serializes gpt_forward
        # (separate scratch buffers & cuBLAS handle let forward_latents overlap with decode!)
        self._gpu_lock = threading.Lock()
        self._fwd_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._pending: Dict[int, List[tuple]] = {mid: [] for mid in range(self.num_models)}
        self._worker_running = False

        t0 = time.perf_counter()
        self._load_weights(model_dirs)
        vram_mib = self.vram_bytes / (1024 * 1024)
        logger.info(
            f">> CudaGptEngine ready ({self.num_models} models, max_batch={max_batch}, "
            f"max_seq={max_seq_len}) in {time.perf_counter() - t0:.2f}s, "
            f"VRAM={vram_mib:.1f} MiB ({self.so_path})"
        )

    @property
    def vram_bytes(self) -> int:
        if not self._handle:
            return 0
        return int(self.lib.gpt_engine_vram_bytes(self._handle))

    def _load_weights(self, model_dirs: Dict[int, str]) -> None:
        shared_loaded = False
        for mid in sorted(model_dirs.keys()):
            bin_path = os.path.join(model_dirs[mid], "gpt", "pytorch_model.bin")
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)

            def _fp16_contig(t: torch.Tensor) -> torch.Tensor:
                return t.to(dtype=torch.float16).contiguous()

            if not shared_loaded:
                audio_emb = _fp16_contig(sd["audio_emb.weight"])
                pos_emb = _fp16_contig(sd["text_pos_embedding.emb.weight"])
                ln_f_w = _fp16_contig(sd["transformer.ln_f.weight"])
                ln_f_b = _fp16_contig(sd["transformer.ln_f.bias"])
                final_w = _fp16_contig(sd["final_norm.weight"])
                final_b = _fp16_contig(sd["final_norm.bias"])
                lm_w = _fp16_contig(sd["lm_head.weight"])
                lm_b = _fp16_contig(sd["lm_head.bias"])
                rc = self.lib.gpt_engine_set_shared_top(
                    self._handle,
                    audio_emb.data_ptr(), pos_emb.data_ptr(),
                    ln_f_w.data_ptr(), ln_f_b.data_ptr(),
                    final_w.data_ptr(), final_b.data_ptr(),
                    lm_w.data_ptr(), lm_b.data_ptr(),
                    0,
                )
                if rc != 0:
                    raise RuntimeError("gpt_engine_set_shared_top failed")

                for l in range(24):
                    p = f"transformer.h.{l}."
                    t_ln1_w = _fp16_contig(sd[p + "ln_1.weight"])
                    t_ln1_b = _fp16_contig(sd[p + "ln_1.bias"])
                    t_catt_b = _fp16_contig(sd[p + "attn.c_attn.bias"])
                    t_aproj_b = _fp16_contig(sd[p + "attn.c_proj.bias"])
                    t_ln2_w = _fp16_contig(sd[p + "ln_2.weight"])
                    t_ln2_b = _fp16_contig(sd[p + "ln_2.bias"])
                    t_cfc_b = _fp16_contig(sd[p + "mlp.c_fc.bias"])
                    t_mproj_b = _fp16_contig(sd[p + "mlp.c_proj.bias"])
                    rc = self.lib.gpt_engine_set_shared_layer(
                        self._handle, l,
                        t_ln1_w.data_ptr(), t_ln1_b.data_ptr(),
                        t_catt_b.data_ptr(), t_aproj_b.data_ptr(),
                        t_ln2_w.data_ptr(), t_ln2_b.data_ptr(),
                        t_cfc_b.data_ptr(), t_mproj_b.data_ptr(),
                        0,
                    )
                    if rc != 0:
                        raise RuntimeError(f"gpt_engine_set_shared_layer({l}) failed")
                shared_loaded = True

            for l in range(24):
                p = f"transformer.h.{l}."
                # HF GPT-2 stores Conv1D as [in, out]; transpose to row-major [out, in]
                c_attn_w = _fp16_contig(sd[p + "attn.c_attn.weight"].t())
                attn_proj_w = _fp16_contig(sd[p + "attn.c_proj.weight"].t())
                c_fc_w = _fp16_contig(sd[p + "mlp.c_fc.weight"].t())
                mlp_proj_w = _fp16_contig(sd[p + "mlp.c_proj.weight"].t())
                rc = self.lib.gpt_engine_set_model_layer(
                    self._handle, mid, l,
                    c_attn_w.data_ptr(), attn_proj_w.data_ptr(),
                    c_fc_w.data_ptr(), mlp_proj_w.data_ptr(),
                    0,
                )
                if rc != 0:
                    raise RuntimeError(f"gpt_engine_set_model_layer({mid}, {l}) failed")
            del sd

    def forward_latents(self, model_id: int, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run 24-layer causal GPT-2 + ln_f + final_norm on [1, T, 1280] -> [1, T, 1280] FP32."""
        if inputs_embeds.ndim == 3:
            assert inputs_embeds.shape[0] == 1
            x = inputs_embeds.squeeze(0)
        else:
            x = inputs_embeds
        x_fp16 = x.to(device=self.device, dtype=torch.float16).contiguous()
        seq_len = int(x_fp16.shape[0])
        out = torch.empty((seq_len, 1280), device=self.device, dtype=torch.float32)
        cur_stream = torch.cuda.current_stream(self.device)
        with self._fwd_lock:
            rc = self.lib.gpt_engine_forward(
                self._handle, int(model_id),
                x_fp16.data_ptr(), seq_len,
                out.data_ptr(), ctypes.c_void_p(cur_stream.cuda_stream),
            )
            # Ensure forward scratch buffers are unused before another thread/stream acquires _fwd_lock
            cur_stream.synchronize()
        if rc != 0:
            raise RuntimeError("gpt_engine_forward failed")
        return out.unsqueeze(0)

    def prefill_logits(self, model_id: int, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run prefill on [1, T, 1280] or [T, 1280] and return last-token [8194] FP32 logits."""
        x = inputs_embeds.squeeze(0) if inputs_embeds.ndim == 3 else inputs_embeds
        x_fp16 = x.to(device=self.device, dtype=torch.float16).contiguous()
        out = torch.empty((8194,), device=self.device, dtype=torch.float32)
        cur_stream = torch.cuda.current_stream(self.device)
        with self._gpu_lock:
            rc = self.lib.gpt_engine_prefill_logits(
                self._handle, int(model_id),
                x_fp16.data_ptr(), int(x_fp16.shape[0]),
                out.data_ptr(), ctypes.c_void_p(cur_stream.cuda_stream),
            )
            cur_stream.synchronize()
        if rc != 0:
            raise RuntimeError("gpt_engine_prefill_logits failed")
        return out

    def generate_batch_sync(
        self,
        model_id: int,
        embeds_list: List[torch.Tensor],
        max_new_tokens: int = 1815,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        stop_token_id: int = 8193,
        penalize_placeholder_zero: bool = True,
        seeds: Optional[List[int]] = None,
    ) -> List[List[int]]:
        bsz = len(embeds_list)
        assert 1 <= bsz <= self.max_batch

        fp16_tensors = [
            (e.squeeze(0) if e.ndim == 3 else e).to(device=self.device, dtype=torch.float16).contiguous()
            for e in embeds_list
        ]
        # Ensure producer stream writes to fp16_tensors are visible to engine's own_stream
        torch.cuda.current_stream(self.device).synchronize()

        ptrs = (ctypes.c_void_p * bsz)(*[t.data_ptr() for t in fp16_tensors])
        lens = (ctypes.c_int * bsz)(*[int(t.shape[0]) for t in fp16_tensors])
        if seeds is None:
            seeds = [random.getrandbits(64) for _ in range(bsz)]
        seed_arr = (ctypes.c_uint64 * bsz)(*[int(s) & 0xFFFFFFFFFFFFFFFF for s in seeds])

        out_tokens = np.empty((bsz, max_new_tokens), dtype=np.int32)
        out_lens = np.empty((bsz,), dtype=np.int32)

        with self._gpu_lock:
            rc = self.lib.gpt_engine_generate_batch(
                self._handle,
                int(model_id),
                int(bsz),
                ptrs,
                lens,
                int(max_new_tokens),
                float(temperature),
                float(top_p),
                int(top_k),
                float(repetition_penalty),
                int(stop_token_id),
                1 if penalize_placeholder_zero else 0,
                seed_arr,
                out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                out_lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                None,
            )
        if rc != 0:
            raise RuntimeError("gpt_engine_generate_batch failed")

        return [out_tokens[b, : int(out_lens[b])].tolist() for b in range(bsz)]

    async def generate_async(
        self,
        model_id: int,
        inputs_embeds: torch.Tensor,
        max_new_tokens: int = 1815,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        stop_token_id: int = 8193,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        """Submit one request and coalesce concurrent same-model requests into one GPU batch."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        item = (
            inputs_embeds, max_new_tokens, temperature, top_p, top_k,
            repetition_penalty, stop_token_id,
            seed if seed is not None else random.getrandbits(64),
            fut,
        )
        should_launch = False
        with self._queue_lock:
            self._pending[model_id].append(item)
            if not self._worker_running:
                self._worker_running = True
                should_launch = True

        if should_launch:
            loop.create_task(self._drain_loop(loop))

        return await fut

    async def _drain_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            while True:
                # Yield once so sibling asyncio.create_task() sentences from the same
                # request (or concurrent burst requests) coalesce into one batch!
                await asyncio.sleep(0)

                batch_mid = None
                batch_items = None
                with self._queue_lock:
                    # Pick the model_id with the largest pending batch (ties -> lower id)
                    best_mid = None
                    best_n = 0
                    for mid, q in self._pending.items():
                        if len(q) > best_n:
                            best_n = len(q)
                            best_mid = mid
                    if best_mid is None:
                        self._worker_running = False
                        return
                    batch_mid = best_mid
                    batch_items = self._pending[best_mid][: self.max_batch]
                    del self._pending[best_mid][: len(batch_items)]

                embeds = [it[0] for it in batch_items]
                # Use sampling params from first item (all prod requests share identical params)
                _, max_new, temp, top_p, top_k, rep_pen, stop_id, _, _ = batch_items[0]
                seeds = [it[7] for it in batch_items]

                def _run():
                    t0 = time.perf_counter()
                    toks = self.generate_batch_sync(
                        batch_mid, embeds,
                        max_new_tokens=max_new,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=rep_pen,
                        stop_token_id=stop_id,
                        seeds=seeds,
                    )
                    dt = time.perf_counter() - t0
                    return toks, dt

                try:
                    toks_list, dt = await loop.run_in_executor(None, _run)
                    for it, toks in zip(batch_items, toks_list):
                        fut = it[8]
                        if not fut.done():
                            fut.set_result((toks, dt))
                except Exception as ex:
                    for it in batch_items:
                        fut = it[8]
                        if not fut.done():
                            fut.set_exception(ex)
        except Exception:
            with self._queue_lock:
                self._worker_running = False
            raise

    def shutdown(self) -> None:
        if self._handle:
            self.lib.gpt_engine_destroy(self._handle)
            self._handle = None
