from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turbomlx.core_ref.packing import unpack_bits
from turbomlx.core_ref.quantizers import MSEPayload, TurboQuantMSERef
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig, TurboQuantMode, ValuesMode
from turbomlx.mlx_runtime.qwen_native import (
    gather_mse_centroids_mlx,
    native_qwen_support_reason,
    qwen_group_size,
    qwen_grouped_mse_scores_mlx,
    reshape_grouped_queries,
    supports_qwen_native_mlx,
    unpack_packed_indices_4bit_mlx,
)


def _require_mlx():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    return mx


def test_supports_qwen_native_mlx_accepts_supported_preview_contract():
    config = TurboQuantConfig(
        bits_total=4,
        mode=TurboQuantMode.MSE,
        values_mode=ValuesMode.DENSE,
        scorer_mode=ScorerMode.NATIVE_MLX,
    )
    cache = SimpleNamespace(
        num_kv_heads=4,
        head_dim=8,
        rotation_matrix=np.eye(8, dtype=np.float32),
        centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32),
        regular_payload=SimpleNamespace(packed_indices=np.zeros((1, 4, 5, 4), dtype=np.uint8)),
        outlier_mask=None,
    )
    queries = np.zeros((1, 16, 2, 8), dtype=np.float32)
    values = np.zeros((1, 4, 5, 8), dtype=np.float32)
    assert supports_qwen_native_mlx(config, cache, queries, values) is True


@pytest.mark.parametrize(
    ("config", "cache", "queries", "values", "reason_key"),
    [
        (
            TurboQuantConfig(bits_total=4, mode=TurboQuantMode.PROD, scorer_mode=ScorerMode.NATIVE_MLX),
            SimpleNamespace(
                num_kv_heads=4,
                head_dim=8,
                rotation_matrix=np.eye(8, dtype=np.float32),
                centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                regular_payload=SimpleNamespace(packed_indices=np.zeros((1, 4, 5, 4), dtype=np.uint8)),
                outlier_mask=None,
            ),
            np.zeros((1, 16, 2, 8), dtype=np.float32),
            np.zeros((1, 4, 5, 8), dtype=np.float32),
            "mode",
        ),
        (
            TurboQuantConfig(bits_total=3, mode=TurboQuantMode.MSE, scorer_mode=ScorerMode.NATIVE_MLX),
            SimpleNamespace(
                num_kv_heads=4,
                head_dim=8,
                rotation_matrix=np.eye(8, dtype=np.float32),
                centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                regular_payload=SimpleNamespace(packed_indices=np.zeros((1, 4, 5, 4), dtype=np.uint8)),
                outlier_mask=None,
            ),
            np.zeros((1, 16, 2, 8), dtype=np.float32),
            np.zeros((1, 4, 5, 8), dtype=np.float32),
            "bits_total",
        ),
        (
            TurboQuantConfig(
                bits_total=4,
                mode=TurboQuantMode.MSE,
                values_mode=ValuesMode.AFFINE,
                scorer_mode=ScorerMode.NATIVE_MLX,
            ),
            SimpleNamespace(
                num_kv_heads=4,
                head_dim=8,
                rotation_matrix=np.eye(8, dtype=np.float32),
                centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                regular_payload=SimpleNamespace(packed_indices=np.zeros((1, 4, 5, 4), dtype=np.uint8)),
                outlier_mask=None,
            ),
            np.zeros((1, 16, 2, 8), dtype=np.float32),
            (np.zeros((1, 4, 5, 8), dtype=np.float32),),
            "values_mode",
        ),
        (
            TurboQuantConfig(bits_total=4, mode=TurboQuantMode.MSE, scorer_mode=ScorerMode.NATIVE_MLX),
            SimpleNamespace(
                num_kv_heads=4,
                head_dim=8,
                rotation_matrix=np.eye(8, dtype=np.float32),
                centroids=np.linspace(-1.0, 1.0, 16, dtype=np.float32),
                regular_payload=SimpleNamespace(packed_indices=np.zeros((1, 4, 5, 4), dtype=np.uint8)),
                outlier_mask=None,
            ),
            np.zeros((1, 4, 2, 8), dtype=np.float32),
            np.zeros((1, 4, 5, 8), dtype=np.float32),
            "non_grouped",
        ),
    ],
)
def test_native_qwen_support_reason_rejects_unsupported_configs(config, cache, queries, values, reason_key):
    reason = native_qwen_support_reason(config, cache, queries, values)
    assert reason is not None
    assert reason[0] == reason_key


def test_qwen_group_size_requires_divisible_grouping():
    assert qwen_group_size(16, 4) == 4
    with pytest.raises(ValueError):
        qwen_group_size(10, 4)


def test_unpack_packed_indices_4bit_mlx_matches_reference_unpacking():
    mx = _require_mlx()
    packed = np.array([[[[0x21, 0x43]]]], dtype=np.uint8)
    unpacked = unpack_packed_indices_4bit_mlx(mx.array(packed), head_dim=4)
    np.testing.assert_array_equal(np.array(unpacked), unpack_bits(packed, 4, 4))


def test_gather_mse_centroids_mlx_matches_reference_values():
    mx = _require_mlx()
    centroids = mx.array(np.linspace(-1.0, 1.0, 16, dtype=np.float32))
    indices = mx.array([[[[1, 2, 3, 4]]]], dtype=mx.int32)
    gathered = gather_mse_centroids_mlx(indices, centroids)
    np.testing.assert_allclose(
        np.array(gathered),
        np.linspace(-1.0, 1.0, 16, dtype=np.float32)[[1, 2, 3, 4]][None, None, None, :],
        atol=1e-6,
    )


def test_qwen_grouped_mse_scores_mlx_matches_reference_grouped_score_matrix():
    mx = _require_mlx()
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((1, 4, 5, 8), dtype=np.float32)
    queries = rng.standard_normal((1, 16, 2, 8), dtype=np.float32)
    quantizer = TurboQuantMSERef(8, 4)
    payload = quantizer.quantize(keys)

    scores = qwen_grouped_mse_scores_mlx(
        mx.array(queries),
        packed_indices=mx.array(payload.packed_indices),
        norms=mx.array(payload.norms),
        centroids=mx.array(quantizer.centroids),
        rotation_matrix=mx.array(quantizer.rotation.matrix),
        kv_heads=4,
        head_dim=8,
    )

    expected = []
    for kv_head in range(4):
        kv_payload = MSEPayload(
            packed_indices=payload.packed_indices[:, kv_head : kv_head + 1, :, :],
            norms=payload.norms[:, kv_head : kv_head + 1, :, :],
            head_dim=payload.head_dim,
            bits_total=payload.bits_total,
            codebook_id=payload.codebook_id,
        )
        group_queries = queries[:, kv_head * 4 : (kv_head + 1) * 4, :, :]
        expected.append(quantizer.score_matrix(group_queries, kv_payload))
    expected_scores = np.concatenate(expected, axis=1)

    np.testing.assert_allclose(np.array(scores), expected_scores, atol=2e-2)
    assert tuple(scores.shape) == (1, 16, 2, 5)


def test_reshape_grouped_queries_preserves_qwen_head_order():
    mx = _require_mlx()
    queries = mx.arange(1 * 8 * 2 * 4, dtype=mx.float32).reshape(1, 8, 2, 4)
    grouped = reshape_grouped_queries(queries, 2)
    assert tuple(grouped.shape) == (1, 2, 4, 2, 4)
    np.testing.assert_array_equal(np.array(grouped[0, 0, 0]), np.array(queries[0, 0]))
    np.testing.assert_array_equal(np.array(grouped[0, 1, 0]), np.array(queries[0, 4]))
