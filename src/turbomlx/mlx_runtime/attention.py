"""TurboMLX attention dispatch."""

from __future__ import annotations

import warnings

from turbomlx._logging import get_logger
from turbomlx.exceptions import MissingDependencyError
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime, mlx_runtime_available
from turbomlx.mlx_runtime.config import ScorerMode
from turbomlx.mlx_runtime.qwen_native import qwen_grouped_native_attention_output

_LOGGER = get_logger(__name__)

_MX_RUNTIME_READY = mlx_runtime_available()
mx = None
TurboQuantKVCache = None

if _MX_RUNTIME_READY:  # pragma: no cover - exercised only in MLX-enabled environments
    mx, _base_mod, _cache_mod = ensure_mlx_runtime()
    from .cache import TurboQuantKVCache


_NATIVE_FALLBACK_WARNED_KEYS: set[str] = set()


def _require_mlx_runtime():
    if not _MX_RUNTIME_READY or mx is None:
        raise MissingDependencyError("MLX runtime dependencies are missing.")


def _is_turboquant_cache(cache) -> bool:
    return TurboQuantKVCache is not None and isinstance(cache, TurboQuantKVCache)


def _set_cache_scorer_route(cache, route: str):
    if hasattr(cache, "last_scorer_route"):
        cache.last_scorer_route = route


def _warn_native_mlx_fallback_once(reason_key: str, message: str) -> None:
    if reason_key in _NATIVE_FALLBACK_WARNED_KEYS:
        _LOGGER.debug("Suppressed duplicate native_mlx fallback warning for %s", reason_key)
        return
    _NATIVE_FALLBACK_WARNED_KEYS.add(reason_key)
    _LOGGER.info("native_mlx fallback engaged (%s): %s", reason_key, message)
    warnings.warn(message, UserWarning, stacklevel=3)


def _scaled_masked_softmax(scores, mask):
    _require_mlx_runtime()
    if mask is None:
        return mx.softmax(scores, axis=-1, precise=True)
    if isinstance(mask, str):
        q_len, k_len = scores.shape[-2:]
        mask = mx.arange(k_len - q_len, k_len)[:, None] >= mx.arange(k_len)[None]
    if mask.dtype == mx.bool_:
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
    else:
        scores = scores + mask
    return mx.softmax(scores, axis=-1, precise=True)


def uses_qwen_grouped_attention(queries, dense_keys) -> bool:
    q_shape = getattr(queries, "shape", ())
    k_shape = getattr(dense_keys, "shape", ())
    if len(q_shape) != 4 or len(k_shape) != 4:
        return False
    if int(q_shape[0]) != int(k_shape[0]):
        raise ValueError("TurboMLX preview requires matching batch dimensions for queries and keys.")
    if int(q_shape[-1]) != int(k_shape[-1]):
        raise ValueError("TurboMLX preview requires matching head dimensions for queries and keys.")
    query_heads = int(q_shape[1])
    kv_heads = int(k_shape[1])
    if query_heads == kv_heads or kv_heads <= 0:
        return False
    if query_heads % kv_heads != 0:
        raise ValueError(
            "TurboMLX preview requires query head count to be divisible by KV head count; "
            f"got {query_heads} and {kv_heads}."
        )
    return True


def qwen_grouped_preview_attention(queries, dense_keys, value_state, *, scale, mask):
    _require_mlx_runtime()
    return mx.fast.scaled_dot_product_attention(
        queries,
        dense_keys,
        value_state,
        scale=scale,
        mask=mask,
    )


def qwen_grouped_native_attention(queries, value_state, cache, *, scale, mask):
    _require_mlx_runtime()
    return qwen_grouped_native_attention_output(
        queries,
        value_state,
        rotation_matrix=cache.rotation_matrix,
        score_keys=cache.native_qwen_score_keys(dtype=queries.dtype),
        scale=scale,
        mask=mask,
    )


def turboquant_scaled_dot_product_attention(queries, _keys, value_state, cache, scale, mask):
    _require_mlx_runtime()
    if not _is_turboquant_cache(cache):
        raise TypeError("cache must be TurboQuantKVCache")
    key_shape_hint = None
    if getattr(cache, "num_kv_heads", None) is not None and getattr(cache, "head_dim", None) is not None:
        key_shape_hint = type(
            "_DenseKeyShapeHint",
            (),
            {"shape": (queries.shape[0], cache.num_kv_heads, max(int(cache.offset), 0), cache.head_dim)},
        )()
    if key_shape_hint is not None and uses_qwen_grouped_attention(queries, key_shape_hint):
        if cache.config.scorer_mode == ScorerMode.NATIVE_MLX:
            reason = cache.native_qwen_support_reason(queries, value_state)
            if reason is None:
                _set_cache_scorer_route(cache, ScorerMode.NATIVE_MLX.value)
                out = qwen_grouped_native_attention(
                    queries,
                    value_state,
                    cache,
                    scale=scale,
                    mask=mask,
                )
                return out.astype(queries.dtype)
            _warn_native_mlx_fallback_once(
                reason[0],
                f"{reason[1]} Falling back to preview grouped attention scoring.",
            )
            _set_cache_scorer_route(cache, "native_mlx_fallback")
        else:
            _set_cache_scorer_route(cache, ScorerMode.ORACLE_PREVIEW.value)
        dense_keys = cache.materialize_keys(dtype=queries.dtype)
        if cache.config.values_mode_name == "affine" and isinstance(value_state, tuple):
            raise NotImplementedError(
                "TurboMLX Qwen preview dense-key fallback does not support affine value quantization yet."
            )
        out = qwen_grouped_preview_attention(
            queries,
            dense_keys,
            value_state,
            scale=scale,
            mask=mask,
        )
        return out.astype(queries.dtype)

    if cache.config.scorer_mode == ScorerMode.NATIVE_MLX:
        _warn_native_mlx_fallback_once(
            "generic_preview_route",
            "TurboMLX native_mlx scorer currently supports only the narrow Qwen grouped-query preview contract. "
            "Falling back to the preview oracle scorer.",
        )
        _set_cache_scorer_route(cache, "native_mlx_fallback")
    else:
        _set_cache_scorer_route(cache, ScorerMode.ORACLE_PREVIEW.value)
    q = queries * scale
    scores = cache.score_queries(q)
    weights = _scaled_masked_softmax(scores, mask)
    if cache.config.values_mode_name == "affine" and isinstance(value_state, tuple):
        out = mx.quantized_matmul(
            weights,
            *value_state,
            transpose=False,
            group_size=cache.value_group_size,
            bits=cache.config.value_bits,
        )
    else:
        out = mx.matmul(weights, value_state)
    return out.astype(queries.dtype)


def dispatch_attention(previous):
    if not _MX_RUNTIME_READY:
        raise MissingDependencyError("MLX runtime dependencies are missing.")

    def _patched(queries, keys, values, cache, scale, mask, sinks=None):
        if _is_turboquant_cache(cache):
            if sinks is not None:
                raise ValueError("TurboQuantKVCache does not support attention sinks.")
            return turboquant_scaled_dot_product_attention(queries, keys, values, cache, scale, mask)
        return previous(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)

    return _patched
