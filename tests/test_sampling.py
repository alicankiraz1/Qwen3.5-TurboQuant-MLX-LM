from __future__ import annotations

import math

import numpy as np
import pytest

from turbomlx.mlx_runtime.sampling import SamplerConfig, make_sampler


class _NumpyMxModule:
    """Minimal numpy-backed stand-in for ``mlx.core`` used in pure-Python tests."""

    int32 = np.int32
    float32 = np.float32

    def array(self, value, dtype=None):
        return np.asarray(value, dtype=dtype)

    def argmax(self, value, axis=-1):
        return np.argmax(value, axis=axis)

    def sort(self, value, axis=-1):
        return np.sort(value, axis=axis)

    def cumsum(self, value, axis=-1):
        return np.cumsum(value, axis=axis)

    def exp(self, value):
        return np.exp(value)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def ones_like(self, value):
        return np.ones_like(value)

    def concatenate(self, items, axis=-1):
        return np.concatenate(list(items), axis=axis)

    def max(self, value, axis=-1, keepdims=False):
        return np.max(value, axis=axis, keepdims=keepdims)

    def logsumexp(self, value, axis=-1, keepdims=True):
        return np.log(np.sum(np.exp(value), axis=axis, keepdims=keepdims))


class _NumpyMxModuleNoRandom(_NumpyMxModule):
    """Variant without ``random.categorical`` to exercise the numpy fallback path."""


def test_sampler_config_rejects_invalid_temperature_and_probabilities():
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        SamplerConfig(temperature=-0.1)
    with pytest.raises(ValueError, match="top_k must be >= 0"):
        SamplerConfig(top_k=-3)
    with pytest.raises(ValueError, match="top_p must be within"):
        SamplerConfig(top_p=1.5)
    with pytest.raises(ValueError, match="min_p must be within"):
        SamplerConfig(min_p=2.0)
    with pytest.raises(ValueError, match="temperature must be finite"):
        SamplerConfig(temperature=math.nan)


def test_greedy_sampler_returns_argmax_when_temperature_is_zero():
    sampler = make_sampler(temperature=0.0, mx_module=_NumpyMxModule())
    logits = np.array([[1.0, 7.0, 3.0]], dtype=np.float32)
    assert int(sampler(logits)[0]) == 1


def _first_sample(sampler, logits):
    """Extract the first sampled token regardless of leading batch shape."""
    result = sampler(logits)
    flat = np.asarray(result).reshape(-1)
    return int(flat[0])


def test_top_k_filter_only_keeps_highest_k_logprobs():
    mx_module = _NumpyMxModuleNoRandom()
    sampler = make_sampler(
        temperature=1.0,
        top_k=2,
        seed=0,
        mx_module=mx_module,
    )
    logits = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    samples = [_first_sample(sampler, logits) for _ in range(50)]
    assert set(samples).issubset({2, 3}), "top_k=2 must restrict samples to the highest-2 tokens"


def test_min_p_threshold_drops_low_probability_tokens():
    mx_module = _NumpyMxModuleNoRandom()
    sampler = make_sampler(
        temperature=1.0,
        min_p=0.5,
        seed=1,
        mx_module=mx_module,
    )
    logits = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    for _ in range(20):
        assert _first_sample(sampler, logits) == 2


def test_top_p_nucleus_filter_keeps_at_least_top_token():
    mx_module = _NumpyMxModuleNoRandom()
    sampler = make_sampler(
        temperature=1.0,
        top_p=0.1,
        seed=2,
        mx_module=mx_module,
    )
    logits = np.array([[0.1, 0.1, 0.1, 5.0]], dtype=np.float32)
    for _ in range(20):
        assert _first_sample(sampler, logits) == 3


def test_seeded_sampler_is_deterministic_across_invocations():
    mx_module = _NumpyMxModuleNoRandom()
    sampler_a = make_sampler(temperature=1.0, top_k=4, seed=42, mx_module=mx_module)
    sampler_b = make_sampler(temperature=1.0, top_k=4, seed=42, mx_module=mx_module)
    logits = np.linspace(0.0, 4.0, 4, dtype=np.float32)[None]
    assert _first_sample(sampler_a, logits) == _first_sample(sampler_b, logits)


def test_sampler_handles_two_dimensional_logits_with_batch_dimension():
    mx_module = _NumpyMxModuleNoRandom()
    sampler = make_sampler(temperature=0.0, mx_module=mx_module)
    logits = np.array(
        [
            [1.0, 9.0, 2.0],
            [3.0, 1.0, 4.0],
        ],
        dtype=np.float32,
    )
    result = sampler(logits)
    assert result.tolist() == [1, 2]
