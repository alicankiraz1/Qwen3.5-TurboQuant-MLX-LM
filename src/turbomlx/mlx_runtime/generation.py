"""Generation helpers for baseline, mlx_quant, and TurboMLX backends."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal, Optional

from turbomlx.exceptions import UnsupportedConfigurationError
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime, mlx_runtime_available
from turbomlx.mlx_runtime.cache import TurboQuantKVCache
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig
from turbomlx.mlx_runtime.metrics import MemoryMetrics, recursive_nbytes
from turbomlx.mlx_runtime.patching import patch_attention_dispatch

Backend = Literal["baseline", "mlx_quant", "turbomlx"]


@dataclass(slots=True)
class GenerationStats:
    prompt_tokens: int
    generation_tokens: int
    timed_generation_tokens: int
    prompt_tps: float
    generation_tps: float
    generation_wall_time_s: float
    peak_memory_gb: float
    allocated_cache_bytes: int
    used_state_bytes: int
    key_path_bytes: int
    value_path_bytes: int
    total_kv_bytes: int
    native_working_set_bytes: int
    backend: str
    scorer_route: str

    def as_dict(self) -> dict:
        return asdict(self)


def convert_prompt_cache(prompt_cache, config: TurboQuantConfig, backend: Backend = "turbomlx"):
    if backend == "baseline":
        return prompt_cache
    mx, _base, cache_mod = ensure_mlx_runtime()
    KVCache = cache_mod.KVCache
    RotatingKVCache = getattr(cache_mod, "RotatingKVCache", None)
    ArraysCache = getattr(cache_mod, "ArraysCache", None)
    for index, cache in enumerate(prompt_cache):
        if backend == "turbomlx" and isinstance(cache, TurboQuantKVCache):
            continue
        if backend == "turbomlx" and not isinstance(cache, KVCache):
            if ArraysCache is not None and isinstance(cache, ArraysCache):
                continue
            if RotatingKVCache is not None and isinstance(cache, RotatingKVCache):
                raise UnsupportedConfigurationError(
                    "TurboMLX preview supports only the default non-rotating KVCache path; "
                    "RotatingKVCache is not supported."
                )
            raise UnsupportedConfigurationError(
                "TurboMLX preview supports only supported KVCache families; "
                f"got {type(cache).__name__}."
            )
        if not isinstance(cache, KVCache) or cache.offset < config.quantized_kv_start:
            continue
        if backend == "mlx_quant":
            prompt_cache[index] = cache.to_quantized(bits=config.bits_total)
        else:
            prompt_cache[index] = TurboQuantKVCache.from_kvcache(cache, config)
    return prompt_cache


def _safe_state(cache):
    try:
        return getattr(cache, "state", None)
    except Exception:
        return None


def _safe_nbytes(cache) -> int:
    try:
        return int(getattr(cache, "nbytes"))
    except Exception:
        return recursive_nbytes(_safe_state(cache))


def _cache_metrics(prompt_cache) -> MemoryMetrics:
    allocated = 0
    used_state = 0
    key_bytes = 0
    value_bytes = 0
    native_working_set_bytes = 0
    for cache in prompt_cache:
        state = _safe_state(cache)
        if hasattr(cache, "memory_metrics"):
            try:
                metrics = cache.memory_metrics()
            except Exception:
                pass
            else:
                allocated += metrics.allocated_cache_bytes
                used_state += metrics.used_state_bytes
                key_bytes += metrics.key_path_bytes
                value_bytes += metrics.value_path_bytes
                native_working_set_bytes += metrics.native_working_set_bytes
                continue
        allocated += _safe_nbytes(cache)
        state_bytes = recursive_nbytes(state)
        used_state += state_bytes
        key_bytes += state_bytes
    return MemoryMetrics(
        allocated_cache_bytes=allocated,
        used_state_bytes=used_state,
        key_path_bytes=key_bytes,
        value_path_bytes=value_bytes,
        total_kv_bytes=key_bytes + value_bytes,
        native_working_set_bytes=native_working_set_bytes,
    )


def _effective_scorer_route(prompt_cache, backend: Backend, config: TurboQuantConfig) -> str:
    if backend == "baseline":
        return "baseline"
    if backend == "mlx_quant":
        return "mlx_quant"
    routes = {
        route
        for route in (getattr(cache, "last_scorer_route", None) for cache in prompt_cache)
        if route is not None
    }
    if "native_mlx_fallback" in routes:
        return "native_mlx_fallback"
    if ScorerMode.NATIVE_MLX.value in routes:
        return ScorerMode.NATIVE_MLX.value
    if ScorerMode.ORACLE_PREVIEW.value in routes:
        return ScorerMode.ORACLE_PREVIEW.value
    return config.scorer_mode.value


def generate_with_backend(
    model,
    prompt_tokens,
    *,
    max_tokens: int = 16,
    backend: Backend = "baseline",
    config: Optional[TurboQuantConfig] = None,
    prefill_step_size: int = 2048,
):
    if not mlx_runtime_available():
        raise RuntimeError("MLX runtime is unavailable.")
    mx, _base, cache_mod = ensure_mlx_runtime()
    patch_attention_dispatch()
    config = config or TurboQuantConfig(bits_total=4)
    prompt_cache = cache_mod.make_prompt_cache(model)

    prompt_size = int(prompt_tokens.size)
    processed = 0
    total_start = time.perf_counter()
    prompt_start = time.perf_counter()

    while prompt_size - processed > 1:
        remaining = (prompt_size - processed) - 1
        n_to_process = min(prefill_step_size, remaining)
        model(prompt_tokens[processed : processed + n_to_process][None], cache=prompt_cache)
        convert_prompt_cache(prompt_cache, config, backend=backend)
        mx.eval([c.state for c in prompt_cache])
        processed += n_to_process

    logits = model(prompt_tokens[processed:][None], cache=prompt_cache)[:, -1, :]
    convert_prompt_cache(prompt_cache, config, backend=backend)
    prompt_elapsed = max(time.perf_counter() - prompt_start, 1e-6)
    prompt_tps = prompt_size / prompt_elapsed if prompt_size > 0 else 0.0
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    next_token = mx.argmax(logprobs, axis=-1)
    generated = [int(next_token.item())]

    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(next_token[None], cache=prompt_cache)[:, -1, :]
        convert_prompt_cache(prompt_cache, config, backend=backend)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        next_token = mx.argmax(logprobs, axis=-1)
        generated.append(int(next_token.item()))

    decode_elapsed = max(time.perf_counter() - decode_start, 1e-6)
    timed_generation_tokens = max(len(generated) - 1, 0)
    metrics = _cache_metrics(prompt_cache)
    scorer_route = _effective_scorer_route(prompt_cache, backend, config)
    stats = GenerationStats(
        prompt_tokens=prompt_size,
        generation_tokens=len(generated),
        timed_generation_tokens=timed_generation_tokens,
        prompt_tps=prompt_tps,
        generation_tps=(timed_generation_tokens / decode_elapsed) if timed_generation_tokens > 0 else 0.0,
        generation_wall_time_s=time.perf_counter() - total_start,
        peak_memory_gb=mx.get_peak_memory() / 1e9,
        allocated_cache_bytes=metrics.allocated_cache_bytes,
        used_state_bytes=metrics.used_state_bytes,
        key_path_bytes=metrics.key_path_bytes,
        value_path_bytes=metrics.value_path_bytes,
        total_kv_bytes=metrics.total_kv_bytes,
        native_working_set_bytes=metrics.native_working_set_bytes,
        backend=backend,
        scorer_route=scorer_route,
    )
    return mx.array(generated), logprobs.squeeze(0), stats


def score_tokens_with_backend(
    model,
    prompt_tokens,
    *,
    backend: Backend = "baseline",
    config: Optional[TurboQuantConfig] = None,
    chunk_size: int = 512,
):
    """Compute per-token log probabilities using a backend-specific KV cache.

    The previous implementation issued one ``model(token)`` call per token,
    which both serialized the autoregressive forward passes and re-ran
    backend conversion on every step. The chunked path here keeps the
    backend semantics intact (the cache is converted between chunks) while
    collapsing per-token Python overhead. ``chunk_size`` bounds the peak
    logits memory footprint to ``chunk_size * vocab_size`` floats.
    """
    if not mlx_runtime_available():
        raise RuntimeError("MLX runtime is unavailable.")
    mx, _base, cache_mod = ensure_mlx_runtime()
    patch_attention_dispatch()
    config = config or TurboQuantConfig(bits_total=4)
    prompt_cache = cache_mod.make_prompt_cache(model)

    prompt_size = int(prompt_tokens.size)
    if prompt_size < 2:
        return mx.zeros((0,), dtype=mx.float32), _cache_metrics(prompt_cache)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    target_tokens = prompt_tokens[1:]
    inputs = prompt_tokens[: prompt_size - 1]
    collected_logprobs: list = []
    processed = 0
    while processed < inputs.size:
        end = min(processed + chunk_size, int(inputs.size))
        chunk_inputs = inputs[processed:end]
        chunk_targets = target_tokens[processed:end]
        chunk_logits = model(chunk_inputs[None], cache=prompt_cache)
        convert_prompt_cache(prompt_cache, config, backend=backend)
        chunk_logprobs = chunk_logits - mx.logsumexp(chunk_logits, axis=-1, keepdims=True)
        selected = mx.take_along_axis(
            chunk_logprobs[0],
            chunk_targets.astype(mx.int32)[:, None],
            axis=-1,
        ).squeeze(-1)
        collected_logprobs.append(selected.astype(mx.float32))
        processed = end

    token_logprobs = (
        mx.concatenate(collected_logprobs, axis=0)
        if len(collected_logprobs) > 1
        else collected_logprobs[0]
    )
    return token_logprobs, _cache_metrics(prompt_cache)
