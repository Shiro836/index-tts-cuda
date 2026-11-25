import os
import asyncio
import threading
import base64
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import json
import time
import soundfile as sf
from typing import List, Optional, Union

from loguru import logger
logger.add("logs/api_server_v2.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

from indextts.infer_vllm_v2 import IndexTTS2

try:
    from indextts.lang_router import ADAPTER_LANGS, BASE_LANGS
    SUPPORTED_LANGUAGES = frozenset(ADAPTER_LANGS) | frozenset(BASE_LANGS)
except ImportError:
    # single-engine deployments do not ship the router; the code list is fixed
    # (docs/ROUTING.md §4) so validation still works
    SUPPORTED_LANGUAGES = frozenset(
        "en zh ja ru uk be ba tt de fr es it pt pl tr nl cs sv".split()
    )

tts = None

# A device-side assert (or any lost-context CUDA error) is unrecoverable in-process:
# the context stays poisoned, every later request 500s, and /health keeps answering
# 200 because it never touches the GPU. Dying is the only way back — the unit is
# Restart=on-failure with no start limit.
_CUDA_CONTEXT_LOST = (
    "device-side assert triggered",
    "an illegal memory access",
    "unspecified launch failure",
    "misaligned address",
)


def _exit_if_context_lost(tb_str):
    if not any(marker in tb_str for marker in _CUDA_CONTEXT_LOST):
        return
    logger.critical("CUDA context lost; exiting so systemd restarts the server")
    # delayed so the caller's 500 still reaches the client; os._exit because a
    # poisoned context makes an orderly vLLM shutdown hang
    threading.Timer(1.0, lambda: os._exit(70)).start()


def _parse_language(data):
    """None/"auto" -> auto-detect; a supported two-letter code -> forced. Anything
    else is a client error (docs/ROUTING.md §4)."""
    language = data.get("language", None)
    if language is None:
        return None
    if not isinstance(language, str):
        raise ValueError("language must be a string")
    language = language.strip().lower()
    if language in ("", "auto"):
        return None
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"unsupported language {language!r}; expected 'auto' or one of "
            f"{sorted(SUPPORTED_LANGUAGES)}"
        )
    return language


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
        offload_device=args.offload_device,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        qwen_emo_mode=args.qwen_emo_mode,
        ref_device=args.ref_device,
        ref_cache_size=args.ref_cache_size,
        mel_workers=args.mel_workers,
        base_model_dir=args.base_model_dir,
        adapter_model_dir=args.adapter_model_dir,
        lid_model_path=args.lid_model,
        base_kv_cache_memory_bytes=args.base_kv_cache_memory_bytes,
        adapter_kv_cache_memory_bytes=args.adapter_kv_cache_memory_bytes,
        single_engine_lora=args.single_engine_lora,
        lora_peft_dir=args.lora_peft_dir,
        code_aligner_path=args.code_aligner,
        code_aligner_threads=args.code_aligner_threads,
    )
    try:
        yield
    finally:
        # without this the EngineCore subprocesses outlive the parent and keep
        # holding their KV cache
        engine = tts
        tts = None
        if engine is not None:
            engine.shutdown()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    status = tts.engine_status()
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time(),
            **status,
        }
    )


def _parse_tts_params(data):
    """Shared request parsing for the synthesis endpoints (mirrors /tts_url)."""
    emo_control_method = data.get("emo_control_method", 0)
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value

    emo_ref_path = data.get("emo_ref_path", None)
    emo_weight = data.get("emo_weight", 1.0)

    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0

    if emo_control_method == 2:
        vec = data.get("emo_vec", [0] * 8)
        if sum(vec) > 1.5:
            raise ValueError("情感向量之和不能超过1.5，请调整后重试。")
    else:
        vec = None

    return dict(
        spk_audio_prompt=data["spk_audio_path"],
        text=data["text"],
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method == 3),
        emo_text=data.get("emo_text", None),
        use_random=data.get("emo_random", False),
        max_text_tokens_per_sentence=int(data.get("max_text_tokens_per_sentence", 120)),
        language=_parse_language(data),
    )


@app.post("/tts_stream")
async def tts_api_stream(request: Request):
    """Stream synthesis as NDJSON: one line per finished sentence chunk

        {"text", "start", "end", "sampling_rate", "audio": "<b64 standalone wav>"}

    followed by a final {"done": true} line ({"error": "..."} on mid-stream
    failure). Chunks include the inter-sentence silence, so decoding and
    concatenating all chunks reproduces the /tts_url output exactly. The first
    line arrives as soon as the first sentence is synthesized, while later
    sentences are still decoding.
    """
    try:
        data = await request.json()
        params = _parse_tts_params(data)
        # guards run here, not inside the generator: once the StreamingResponse has
        # started there is no status code left to change
        tts.validate_request(
            params["text"],
            language=params["language"],
            max_text_tokens_per_sentence=params["max_text_tokens_per_sentence"],
        )
    except Exception as ex:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": str(ex)},
        )

    async def ndjson():
        try:
            async for segment, sr, chunk in tts.infer_stream(**params):
                wav_data = chunk.numpy().astype("int16").T
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, wav_data, sr, format="WAV")
                    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")
                segment.pop("_codes", None)  # private to infer(); not part of the wire format
                yield json.dumps({**segment, "sampling_rate": sr, "audio": audio_b64}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as ex:
            tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
            logger.error(f"tts_stream failed: {tb_str}")
            _exit_if_context_lost(tb_str)
            yield json.dumps({"error": str(ex)}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)
        return_segments = bool(data.get("return_segments", False))
        language = _parse_language(data)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        # logger.info(f"Emo control mode:{emo_control_method}, vec:{vec}")
        tts.validate_request(text, language=language,
                             max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        result = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                        language=language,
                        return_segments=return_segments)
        if return_segments:
            sr, wav, segments = result
        else:
            sr, wav = result
            segments = None

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        if return_segments:
            # JSON response with sentence-level time marks for frontend sync
            return JSONResponse(content={
                "audio": base64.b64encode(wav_bytes).decode("ascii"),
                "sampling_rate": sr,
                "segments": segments,
            })

        return Response(content=wav_bytes, media_type="audio/wav")

    except ValueError as ex:
        # input guards and unknown language codes are client errors
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": str(ex)
            }
        )
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        logger.error(f"tts_url failed: {tb_str}")
        _exit_if_context_lost(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM", help="Model checkpoints directory (single-engine mode)")
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Pristine base model dir serving en/zh. Enables dual-engine routing together with --adapter_model_dir; without both, --model_dir runs single-engine as before")
    parser.add_argument("--adapter_model_dir", type=str, default=None,
                        help="LoRA-merged model dir serving the 16 new languages")
    parser.add_argument("--lid_model", type=str, default=None,
                        help="GlotLID model.bin used for per-sentence language detection; required for dual-engine routing")
    parser.add_argument("--base_kv_cache_memory_bytes", type=int, default=None,
                        help="KV cache budget for the base (en/zh) engine in bytes; falls back to --kv_cache_memory_bytes")
    parser.add_argument("--adapter_kv_cache_memory_bytes", type=int, default=None,
                        help="KV cache budget for the adapter engine in bytes (the busy path — give it the bigger share); falls back to --kv_cache_memory_bytes")
    parser.add_argument("--single_engine_lora", action="store_true", default=False,
                        help="Serve BOTH views from one LoRA-enabled engine on the base weights: adapter languages attach a per-request LoRA, en/zh run bit-exact base. Saves ~1.7 GiB and unifies the KV pool (base+adapter budgets are summed); costs ~+120 ms per audio-second on adapter languages at low concurrency (STATE 15-J measurements)")
    parser.add_argument("--lora_peft_dir", type=str, default=None,
                        help="PEFT-format adapter dir (from tools/lora_to_peft.py); required with --single_engine_lora")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.10)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    parser.add_argument("--offload_device", type=str, default=None, help="Device to offload non-GPT models to (e.g. cuda:1)")
    parser.add_argument("--kv_cache_memory_bytes", type=int, default=None,
                        help="Explicit KV cache budget for the GPT engine in bytes; overrides gpu_memory_utilization and makes boot deterministic")
    parser.add_argument("--qwen_emo_mode", type=str, default="lazy", choices=["lazy", "eager", "disabled"],
                        help="When to build the Qwen emotion engine (~3GB VRAM, only used by emo_control_method=3)")
    parser.add_argument("--ref_device", type=str, default=None,
                        help="Device for reference-audio-only models (w2v-bert, campplus), e.g. 'cpu' to save ~2.5GB VRAM")
    parser.add_argument("--ref_cache_size", type=int, default=8,
                        help="Number of reference voices whose conditioning stays cached (tens of MB each)")
    parser.add_argument("--code_aligner", type=str, default=None,
                        help="checkpoint of the code-domain CTC aligner (indextts/utils/code_aligner.py); when set, "
                             "every segment carries word timings in segment['words'] (StyleTTS2 shape)")
    parser.add_argument("--code_aligner_threads", type=int, default=8,
                        help="torch intra-op threads for the code aligner (process-wide); 8 is the knee on serv2")
    parser.add_argument("--mel_workers", type=int, default=1,
                        help="Threads for the synchronous GPU tail (GPT forward/s2mel/BigVGAN); 1 already keeps the event loop responsive, >1 overlaps concurrent requests at the cost of activation VRAM per worker")
    args = parser.parse_args()
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port)