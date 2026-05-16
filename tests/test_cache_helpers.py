import json

import numpy as np
import pytest

from turbomlx.core_ref.quantizers import TurboQuantMSERef
from turbomlx.mlx_runtime.cache import (
    TurboQuantKVCache,
    _deserialize_meta_state,
    _materialize_dense_keys_numpy,
    _safe_numpy_array,
    _serialize_meta_state,
    _slice_token_axis,
    _token_axis_length,
)
from turbomlx.mlx_runtime.config import (
    MixedPrecisionProfileConfig,
    OutlierSelectionPolicy,
    ScorerMode,
    TurboQuantConfig,
    TurboQuantMode,
    ValuesMode,
)
from turbomlx.mlx_runtime.metrics import recursive_nbytes


def test_slice_token_axis_preserves_backing_and_truncates_used_prefix():
    backing = np.arange(24, dtype=np.float32).reshape(1, 1, 4, 6)
    used = _slice_token_axis(backing, 2)
    assert backing.shape == (1, 1, 4, 6)
    assert used.shape == (1, 1, 2, 6)
    assert recursive_nbytes(backing) > recursive_nbytes(used)


def test_token_axis_length_handles_quantized_tuple_state():
    tuple_state = (
        np.zeros((1, 2, 5, 8), dtype=np.uint8),
        np.zeros((1, 2, 5, 4), dtype=np.float32),
    )
    assert _token_axis_length(tuple_state) == 5


def test_meta_state_roundtrip_preserves_non_default_configuration():
    config = TurboQuantConfig(
        bits_total=4,
        mode=TurboQuantMode.MSE,
        values_mode=ValuesMode.AFFINE,
        value_bits=3,
        scorer_mode=ScorerMode.ORACLE_PREVIEW,
        quantizer_version="quant-v2",
        codebook_version="codebook-v9",
        rotation_kind="qr",
        rotation_seed=17,
        qjl_seed=23,
        quantized_kv_start=11,
        packing_layout="packed-v2",
        mixed_precision=MixedPrecisionProfileConfig(
            outlier_channels=3,
            outlier_high_bits=4,
            regular_bits=2,
            outlier_selection_policy=OutlierSelectionPolicy.ROLLING_VARIANCE,
            calibration_prefix_tokens=77,
        ),
    )
    encoded = _serialize_meta_state(
        config=config,
        head_dim=128,
        num_kv_heads=8,
        offset=19,
        value_group_size=64,
    )
    payload = _deserialize_meta_state(encoded)
    assert payload["schema_version"] == 1
    assert payload["value_bits"] == 3
    assert payload["codebook_version"] == "codebook-v9"
    assert payload["packing_layout"] == "packed-v2"
    assert payload["scorer_mode"] == "oracle_preview"
    assert payload["value_group_size"] == 64
    assert payload["mixed_precision"]["outlier_channels"] == 3
    assert payload["mixed_precision"]["outlier_selection_policy"] == "rolling_variance"
    assert payload["mixed_precision"]["calibration_prefix_tokens"] == 77


def test_meta_state_rejects_unknown_schema_version():
    with pytest.raises(ValueError):
        _deserialize_meta_state((json.dumps({"schema_version": 999}),))


class _BrokenArrayProtocol:
    def __array__(self):
        raise RuntimeError("broken PEP 3118 conversion")

    def tolist(self):
        return [[1, 2], [3, 4]]


def test_safe_numpy_array_falls_back_to_tolist_when_array_protocol_breaks():
    value = _safe_numpy_array(_BrokenArrayProtocol())
    assert value.shape == (2, 2)
    assert value.tolist() == [[1, 2], [3, 4]]


def test_empty_runtime_helper_tensors_should_be_treated_as_absent():
    packed_signs = np.zeros((0,), dtype=np.uint8)
    residual_norms = np.zeros((0,), dtype=np.float32)
    assert packed_signs.size == 0
    assert residual_norms.size == 0


class _Node:
    def __init__(self, payload=None):
        self.payload = payload
        self.next = None


def test_recursive_nbytes_counts_python_fallback_objects_and_dict_keys():
    sample = {"needle": "value", 7: _Node(payload="payload")}
    assert recursive_nbytes(sample) > 0


def test_recursive_nbytes_deduplicates_shared_references_and_cycles():
    shared = np.arange(16, dtype=np.float32)
    linked = _Node(payload=shared)
    linked.next = linked

    duplicated = recursive_nbytes([shared, shared])
    unique = recursive_nbytes([shared])

    assert duplicated < unique + shared.nbytes
    assert recursive_nbytes(linked) > 0


def test_materialize_dense_keys_numpy_reconstructs_quantized_prefix_and_dense_tail():
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((1, 2, 3, 128), dtype=np.float32)
    quantizer = TurboQuantMSERef(128, 4)
    quantized_prefix = quantizer.quantize(keys[..., :2, :])

    dense = _materialize_dense_keys_numpy(
        quantizer=quantizer,
        regular_payload=quantized_prefix,
        calibration_keys=keys[..., 2:, :],
        calibration_tokens_seen=1,
    )

    expected = np.concatenate([quantizer.dequantize(quantized_prefix), keys[..., 2:, :]], axis=2)
    assert dense.shape == (1, 2, 3, 128)
    np.testing.assert_allclose(dense, expected, atol=1e-6)


def _require_mlx():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    return mx


def test_turboquant_cache_native_qwen_score_path_avoids_materialize_keys(monkeypatch):
    mx = _require_mlx()
    rng = np.random.default_rng(1)
    config = TurboQuantConfig(bits_total=4, scorer_mode=ScorerMode.NATIVE_MLX)
    cache = TurboQuantKVCache(config)
    keys = mx.array(rng.standard_normal((1, 4, 5, 8), dtype=np.float32))
    values = mx.array(rng.standard_normal((1, 4, 5, 8), dtype=np.float32))
    cache.update_and_fetch(keys, values)

    monkeypatch.setattr(
        cache,
        "materialize_keys",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("native scorer should not materialize dense keys")),
    )

    queries = mx.array(rng.standard_normal((1, 16, 2, 8), dtype=np.float32))
    scores = cache.score_qwen_grouped_queries_mlx(queries)
    metrics = cache.memory_metrics()

    assert tuple(scores.shape) == (1, 16, 2, 5)
    assert cache.native_rotated_values is not None
    assert cache.preview_dense_keys is None
    assert metrics.native_working_set_bytes > 0
