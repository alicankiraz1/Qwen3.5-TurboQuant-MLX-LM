"""Qwen-first native MLX scorer helpers for narrow preview support."""

from __future__ import annotations

from turbomlx.exceptions import MissingDependencyError
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime, mlx_runtime_available
from turbomlx.mlx_runtime.config import TurboQuantMode, ValuesMode

_MX_RUNTIME_READY = mlx_runtime_available()
mx = None

if _MX_RUNTIME_READY:  # pragma: no cover - exercised in MLX-enabled environments
    mx, _base_mod, _cache_mod = ensure_mlx_runtime()


def _require_mlx_runtime():
    if not _MX_RUNTIME_READY or mx is None:
        raise MissingDependencyError("MLX runtime dependencies are missing.")


def qwen_group_size(query_heads: int, kv_heads: int) -> int:
    if kv_heads <= 0:
        raise ValueError("TurboMLX Qwen native scorer requires kv_heads > 0.")
    if query_heads % kv_heads != 0:
        raise ValueError(
            "TurboMLX Qwen native scorer requires query head count to be divisible by KV head count; "
            f"got {query_heads} and {kv_heads}."
        )
    return query_heads // kv_heads


def reshape_grouped_queries(queries, kv_heads: int):
    _require_mlx_runtime()
    batch, query_heads, query_len, head_dim = (int(dim) for dim in queries.shape)
    group_size = qwen_group_size(query_heads, kv_heads)
    return queries.reshape(batch, kv_heads, group_size, query_len, head_dim)


def native_qwen_support_reason(config, cache, queries, value_state):
    query_shape = getattr(queries, "shape", ())
    if len(query_shape) != 4:
        return ("query_rank", "TurboMLX native_mlx Qwen scorer requires queries with rank 4.")
    if config.mode != TurboQuantMode.MSE:
        return ("mode", "TurboMLX native_mlx Qwen scorer currently supports only mode=mse.")
    if int(config.bits_total) != 4:
        return ("bits_total", "TurboMLX native_mlx Qwen scorer currently supports only bits_total=4.")
    if config.values_mode != ValuesMode.DENSE or isinstance(value_state, tuple):
        return (
            "values_mode",
            "TurboMLX native_mlx Qwen scorer currently supports only values_mode=dense.",
        )
    if config.mixed_precision is not None:
        return (
            "mixed_precision",
            "TurboMLX native_mlx Qwen scorer does not support mixed precision yet.",
        )

    kv_heads = getattr(cache, "num_kv_heads", None)
    head_dim = getattr(cache, "head_dim", None)
    if kv_heads is None or head_dim is None:
        return ("cache_shape", "TurboMLX native_mlx Qwen scorer requires initialized KV head metadata.")
    if int(query_shape[-1]) != int(head_dim):
        return (
            "head_dim",
            "TurboMLX native_mlx Qwen scorer requires query head_dim to match cache head_dim.",
        )

    try:
        group_size = qwen_group_size(int(query_shape[1]), int(kv_heads))
    except ValueError as exc:
        return ("head_grouping", str(exc))
    if group_size <= 1:
        return (
            "non_grouped",
            "TurboMLX native_mlx Qwen scorer currently targets grouped-query Qwen attention only.",
        )

    if getattr(cache, "rotation_matrix", None) is None or getattr(cache, "centroids", None) is None:
        return (
            "artifacts",
            "TurboMLX native_mlx Qwen scorer requires MLX rotation and centroid artifacts.",
        )

    regular_payload = getattr(cache, "regular_payload", None)
    if regular_payload is None or getattr(regular_payload, "packed_indices", None) is None:
        return ("payload", "TurboMLX native_mlx Qwen scorer requires a quantized regular key payload.")
    if getattr(cache, "outlier_mask", None) is not None:
        return (
            "mixed_payload",
            "TurboMLX native_mlx Qwen scorer currently supports only plain MSE payloads.",
        )
    return None


def supports_qwen_native_mlx(config, cache, queries, value_state) -> bool:
    return native_qwen_support_reason(config, cache, queries, value_state) is None


def unpack_packed_indices_4bit_mlx(packed_indices, *, head_dim: int):
    _require_mlx_runtime()
    low = packed_indices & 0x0F
    high = (packed_indices >> 4) & 0x0F
    indices = mx.stack([low, high], axis=-1).reshape(*packed_indices.shape[:-1], packed_indices.shape[-1] * 2)
    return indices[..., :head_dim].astype(mx.int32)


def gather_mse_centroids_mlx(indices, centroids):
    _require_mlx_runtime()
    return mx.take(centroids.astype(mx.float32), indices.astype(mx.int32), axis=0)


def broadcast_key_norms_mlx(norms, *, group_size: int | None = None):
    """Insert grouped-query-friendly axes around the cached key norms.

    The returned tensor has shape ``[batch, kv_heads, 1, key_len, 1]`` so it
    can broadcast against grouped rotated values shaped
    ``[batch, kv_heads, group_size, key_len, head_dim]`` for any
    ``group_size`` value. ``group_size`` is accepted as an explicit reminder
    of the expected broadcast contract but is intentionally not used when
    materializing the tensor; the broadcast happens during downstream matmul.
    """
    _require_mlx_runtime()
    if group_size is not None and group_size < 1:
        raise ValueError("group_size must be >= 1 when provided")
    squeezed = mx.squeeze(norms.astype(mx.float32), axis=-1)
    return squeezed[:, :, None, :, None]


def rotate_queries_mlx(queries, rotation_matrix):
    _require_mlx_runtime()
    return mx.matmul(queries.astype(mx.float32), mx.transpose(rotation_matrix.astype(mx.float32), (1, 0)))


def build_qwen_mse_score_keys_mlx(rotated_values, norms, *, dtype=None):
    _require_mlx_runtime()
    target_dtype = dtype or mx.float32
    return (rotated_values.astype(target_dtype) * norms.astype(target_dtype)).astype(target_dtype)


def qwen_grouped_native_attention_output(queries, value_state, *, rotation_matrix, score_keys, scale, mask):
    _require_mlx_runtime()
    rotated_queries = rotate_queries_mlx(queries, rotation_matrix).astype(queries.dtype)
    return mx.fast.scaled_dot_product_attention(
        rotated_queries,
        score_keys.astype(queries.dtype),
        value_state,
        scale=scale,
        mask=mask,
    )


def qwen_grouped_mse_scores_mlx(
    queries,
    *,
    packed_indices,
    norms,
    centroids,
    rotation_matrix,
    kv_heads: int,
    head_dim: int,
    rotated_values=None,
    score_keys=None,
):
    _require_mlx_runtime()
    grouped_queries = reshape_grouped_queries(queries, kv_heads)
    rotated_queries = rotate_queries_mlx(grouped_queries, rotation_matrix)

    if score_keys is None:
        if rotated_values is None:
            indices = unpack_packed_indices_4bit_mlx(packed_indices, head_dim=head_dim)
            rotated_values = gather_mse_centroids_mlx(indices, centroids)
        else:
            rotated_values = rotated_values.astype(mx.float32)
        score_keys = build_qwen_mse_score_keys_mlx(
            rotated_values[:, :, None, :, :],
            broadcast_key_norms_mlx(norms, group_size=int(grouped_queries.shape[2])),
            dtype=mx.float32,
        )
    else:
        score_keys = score_keys[:, :, None, :, :].astype(mx.float32)

    scores = mx.matmul(rotated_queries, mx.swapaxes(score_keys, -1, -2))
    batch, _kv_heads, group_size, query_len, key_len = (int(dim) for dim in scores.shape)
    return scores.reshape(batch, _kv_heads * group_size, query_len, key_len)
