from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from turbomlx.exceptions import UnsupportedConfigurationError
from turbomlx.mlx_runtime import generation
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig
from turbomlx.mlx_runtime.generation import _cache_metrics, _effective_scorer_route, convert_prompt_cache
from turbomlx.mlx_runtime.metrics import MemoryMetrics


class _FakeKVCache:
    def __init__(self, offset=4):
        self.offset = offset


class _FakeRotatingKVCache:
    def __init__(self, offset=4):
        self.offset = offset


class _FakeArraysCache:
    def __init__(self, offset=4):
        self.offset = offset


class _BrokenNbytesCache:
    def __init__(self):
        self.state = (np.arange(8, dtype=np.uint8),)

    @property
    def nbytes(self):
        raise NameError("tree_reduce is not defined")


class _MetricsAwareCache:
    state = ()

    def memory_metrics(self):
        return MemoryMetrics(
            allocated_cache_bytes=64,
            used_state_bytes=48,
            key_path_bytes=32,
            value_path_bytes=8,
            total_kv_bytes=40,
            native_working_set_bytes=12,
        )


class _FakeMx:
    int32 = np.int32
    float32 = np.float32

    def logsumexp(self, logits, axis=-1, keepdims=True):
        return np.log(np.sum(np.exp(logits), axis=axis, keepdims=keepdims))

    def argmax(self, logits, axis=-1):
        return np.argmax(logits, axis=axis)

    def get_peak_memory(self):
        return 0

    def eval(self, _values):
        return None

    def array(self, value, dtype=None):
        return np.array(value, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float32)


def test_convert_prompt_cache_rejects_unsupported_rotating_cache(monkeypatch):
    cache_mod = SimpleNamespace(
        KVCache=_FakeKVCache,
        RotatingKVCache=_FakeRotatingKVCache,
        ArraysCache=_FakeArraysCache,
    )
    monkeypatch.setattr(
        "turbomlx.mlx_runtime.generation.ensure_mlx_runtime",
        lambda: (object(), object(), cache_mod),
    )

    with pytest.raises(UnsupportedConfigurationError):
        convert_prompt_cache([_FakeRotatingKVCache()], TurboQuantConfig(bits_total=4), backend="turbomlx")


def test_convert_prompt_cache_allows_arrays_cache_passthrough(monkeypatch):
    cache_mod = SimpleNamespace(
        KVCache=_FakeKVCache,
        RotatingKVCache=_FakeRotatingKVCache,
        ArraysCache=_FakeArraysCache,
    )
    monkeypatch.setattr(
        "turbomlx.mlx_runtime.generation.ensure_mlx_runtime",
        lambda: (object(), object(), cache_mod),
    )

    arrays_cache = _FakeArraysCache()
    prompt_cache = [arrays_cache]
    assert convert_prompt_cache(prompt_cache, TurboQuantConfig(bits_total=4), backend="turbomlx")[0] is arrays_cache


def test_convert_prompt_cache_is_idempotent_for_already_converted_turbomlx_cache(monkeypatch):
    cache_mod = SimpleNamespace(
        KVCache=_FakeKVCache,
        RotatingKVCache=_FakeRotatingKVCache,
        ArraysCache=_FakeArraysCache,
    )

    class _FakeTurboQuantKVCache:
        pass

    monkeypatch.setattr(
        "turbomlx.mlx_runtime.generation.ensure_mlx_runtime",
        lambda: (object(), object(), cache_mod),
    )
    monkeypatch.setattr(generation, "TurboQuantKVCache", _FakeTurboQuantKVCache)

    converted = _FakeTurboQuantKVCache()
    prompt_cache = [converted]
    assert convert_prompt_cache(prompt_cache, TurboQuantConfig(bits_total=4), backend="turbomlx")[0] is converted


def test_cache_metrics_falls_back_when_upstream_nbytes_property_is_broken():
    metrics = _cache_metrics([_BrokenNbytesCache()])
    assert metrics.allocated_cache_bytes > 0
    assert metrics.used_state_bytes > 0
    assert metrics.native_working_set_bytes == 0


def test_cache_metrics_preserves_native_working_set_from_structured_runtime_metrics():
    metrics = _cache_metrics([_MetricsAwareCache()])
    assert metrics.allocated_cache_bytes == 64
    assert metrics.used_state_bytes == 48
    assert metrics.key_path_bytes == 32
    assert metrics.total_kv_bytes == 40
    assert metrics.native_working_set_bytes == 12


def test_effective_scorer_route_prefers_native_fallback_when_any_layer_falls_back():
    prompt_cache = [
        SimpleNamespace(last_scorer_route=ScorerMode.NATIVE_MLX.value),
        SimpleNamespace(last_scorer_route="native_mlx_fallback"),
    ]
    route = _effective_scorer_route(prompt_cache, "turbomlx", TurboQuantConfig(bits_total=4))
    assert route == "native_mlx_fallback"


def test_effective_scorer_route_reports_native_when_no_fallback_occurred():
    prompt_cache = [SimpleNamespace(last_scorer_route=ScorerMode.NATIVE_MLX.value)]
    route = _effective_scorer_route(
        prompt_cache,
        "turbomlx",
        TurboQuantConfig(bits_total=4, scorer_mode=ScorerMode.NATIVE_MLX),
    )
    assert route == "native_mlx"


def test_score_tokens_with_backend_uses_single_chunked_forward_pass(monkeypatch):
    """Batched scoring should issue O(prompt/chunk_size) forward passes, not one per token."""
    fake_mx = _FakeMx()
    cache_mod = SimpleNamespace(make_prompt_cache=lambda _model: [SimpleNamespace(state=())])

    forward_calls: list[int] = []

    def fake_take_along_axis(values, indices, axis=-1):
        gathered = np.take_along_axis(values, indices, axis=axis)
        return gathered

    fake_mx.take_along_axis = fake_take_along_axis
    fake_mx.concatenate = lambda items, axis=0: np.concatenate(list(items), axis=axis)

    class _DeterministicModel:
        def __call__(self, inputs, cache=None):
            forward_calls.append(int(inputs.shape[1]))
            seq_len = int(inputs.shape[1])
            return np.tile(np.arange(4, dtype=np.float32)[None, None, :], (1, seq_len, 1))

    monkeypatch.setattr(generation, "mlx_runtime_available", lambda: True)
    monkeypatch.setattr(generation, "ensure_mlx_runtime", lambda: (fake_mx, object(), cache_mod))
    monkeypatch.setattr(generation, "patch_attention_dispatch", lambda: None)
    monkeypatch.setattr(generation, "convert_prompt_cache", lambda prompt_cache, config, backend="turbomlx": prompt_cache)

    prompt_tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    token_logprobs, _metrics = generation.score_tokens_with_backend(
        _DeterministicModel(),
        prompt_tokens,
        backend="baseline",
        config=TurboQuantConfig(bits_total=4),
        chunk_size=4,
    )

    assert token_logprobs.shape == (7,)
    assert forward_calls == [4, 3]
    assert len(forward_calls) < prompt_tokens.size - 1


def test_generate_with_backend_counts_full_prompt_elapsed_and_only_timed_decode_tokens(monkeypatch):
    fake_mx = _FakeMx()
    cache_mod = SimpleNamespace(make_prompt_cache=lambda _model: [SimpleNamespace(state=np.zeros((1,), dtype=np.uint8))])

    class _FakeModel:
        def __call__(self, inputs, cache=None):
            seq_len = int(inputs.shape[1])
            return np.zeros((1, seq_len, 4), dtype=np.float32)

    perf_values = iter([0.0, 0.0, 1.0, 1.0, 3.0, 4.0])

    monkeypatch.setattr(generation, "mlx_runtime_available", lambda: True)
    monkeypatch.setattr(generation, "ensure_mlx_runtime", lambda: (fake_mx, object(), cache_mod))
    monkeypatch.setattr(generation, "patch_attention_dispatch", lambda: None)
    monkeypatch.setattr(generation, "convert_prompt_cache", lambda prompt_cache, config, backend="turbomlx": prompt_cache)
    monkeypatch.setattr(generation.time, "perf_counter", lambda: next(perf_values))

    prompt_tokens = np.array([1, 2, 3, 4], dtype=np.int32)
    _generated, _logprobs, stats = generation.generate_with_backend(
        _FakeModel(),
        prompt_tokens,
        max_tokens=3,
        backend="baseline",
        config=TurboQuantConfig(bits_total=4),
        prefill_step_size=2,
    )

    assert stats.prompt_tokens == 4
    assert stats.generation_tokens == 3
    assert stats.timed_generation_tokens == 2
    assert stats.prompt_tps == pytest.approx(4.0)
    assert stats.generation_tps == pytest.approx(1.0)


def test_stream_with_backend_emits_one_event_per_generated_token(monkeypatch):
    """The streaming variant should yield exactly ``max_tokens`` events.

    The prefill phase makes one model call per chunk plus a final
    single-token call for the position the model needs to score, after
    which the decode phase issues ``max_tokens - 1`` autoregressive
    forward passes. ``prefill_step_size`` is chosen so the prefill never
    needs to chunk in the middle of the prompt and the single-token
    prefill tail is observable below.
    """
    fake_mx = _FakeMx()
    cache_mod = SimpleNamespace(make_prompt_cache=lambda _model: [SimpleNamespace(state=np.zeros((1,), dtype=np.uint8))])

    forward_history: list[tuple[int, int]] = []

    class _FakeModel:
        def __call__(self, inputs, cache=None):
            seq_len = int(inputs.shape[1])
            forward_history.append((int(inputs.shape[0]), seq_len))
            logits = np.zeros((1, seq_len, 4), dtype=np.float32)
            logits[:, :, 2] = 5.0
            return logits

    monkeypatch.setattr(generation, "mlx_runtime_available", lambda: True)
    monkeypatch.setattr(generation, "ensure_mlx_runtime", lambda: (fake_mx, object(), cache_mod))
    monkeypatch.setattr(generation, "patch_attention_dispatch", lambda: None)
    monkeypatch.setattr(generation, "convert_prompt_cache", lambda prompt_cache, config, backend="turbomlx": prompt_cache)

    prompt_tokens = np.array([1, 2, 3, 4], dtype=np.int32)
    stream = generation.stream_with_backend(
        _FakeModel(),
        prompt_tokens,
        max_tokens=3,
        backend="baseline",
        config=TurboQuantConfig(bits_total=4),
        prefill_step_size=64,
    )

    events = list(stream)
    assert [event.token for event in events] == [2, 2, 2]
    assert [event.position for event in events] == [0, 1, 2]
    assert all(event.logprob <= 0.0 for event in events)
    prefill_calls = [call for call in forward_history if call[1] > 1]
    decode_calls = [call for call in forward_history if call[1] == 1]
    assert len(prefill_calls) == 1, "single-chunk prefill should issue exactly one multi-token forward pass"
    assert len(decode_calls) == len(events) - 1 + 1, (
        "decode count = (max_tokens - 1) + the single-token prefill tail"
    )
