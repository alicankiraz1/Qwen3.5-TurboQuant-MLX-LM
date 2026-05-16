"""Token sampling strategies for TurboMLX generation.

The samplers are written against the MLX functional surface but kept in a
self-contained module so they can be exercised with a NumPy stand-in in
unit tests. Each sampler accepts a logits tensor shaped ``[..., vocab]``
and returns a token tensor shaped ``[...]`` along the same leading axes,
so they slot into both single-token decode and batched scoring paths
without bespoke reshaping.

The constructor signature intentionally mirrors the ``mlx_lm`` sampler
factory so users moving between the two stacks do not have to re-learn
parameter names. The filtering chain follows the conventional order
"temperature → top-k → min-p → top-p" applied to log-probabilities so the
numerical behavior is independent of the underlying tensor framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from turbomlx._logging import get_logger
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime, mlx_runtime_available

_LOGGER = get_logger(__name__)

_MX_RUNTIME_READY = mlx_runtime_available()
mx: Any = None
if _MX_RUNTIME_READY:  # pragma: no cover - exercised only in MLX-enabled environments
    mx, _base_mod, _cache_mod = ensure_mlx_runtime()


def _require_mlx_runtime() -> None:
    if not _MX_RUNTIME_READY or mx is None:
        from turbomlx.exceptions import MissingDependencyError

        raise MissingDependencyError("MLX runtime dependencies are missing.")


class _Sampler(Protocol):
    def __call__(self, logits: Any) -> Any: ...


@dataclass(slots=True)
class SamplerConfig:
    """Declarative description of the sampling chain."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    min_p: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be within [0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be within [0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


class _GreedySampler:
    """Argmax sampler. Equivalent to ``temperature == 0``."""

    def __init__(self, mx_module: Any | None = None) -> None:
        self._mx = mx_module if mx_module is not None else mx

    def __call__(self, logits: Any) -> Any:
        return self._mx.argmax(logits, axis=-1)


def _apply_temperature(mx_module: Any, logits: Any, temperature: float) -> Any:
    if temperature == 1.0:
        return logits
    return logits / mx_module.array(temperature, dtype=logits.dtype)


def _neg_inf_like(mx_module: Any, reference: Any) -> Any:
    return mx_module.array(-float("inf"), dtype=reference.dtype)


def _apply_top_k(mx_module: Any, logprobs: Any, top_k: int) -> Any:
    """Mask everything below the kth-largest log-probability per row."""
    if top_k <= 0:
        return logprobs
    vocab = int(logprobs.shape[-1])
    k = min(top_k, vocab)
    sorted_logprobs = mx_module.sort(logprobs, axis=-1)
    threshold = sorted_logprobs[..., vocab - k:vocab - k + 1]
    return mx_module.where(logprobs >= threshold, logprobs, _neg_inf_like(mx_module, logprobs))


def _apply_min_p(mx_module: Any, logprobs: Any, min_p: float) -> Any:
    """Drop tokens whose probability is less than ``min_p * max_probability``."""
    if min_p <= 0.0:
        return logprobs
    log_min_p = math.log(min_p)
    max_logprob = mx_module.max(logprobs, axis=-1, keepdims=True)
    threshold = max_logprob + mx_module.array(log_min_p, dtype=logprobs.dtype)
    return mx_module.where(logprobs >= threshold, logprobs, _neg_inf_like(mx_module, logprobs))


def _apply_top_p(mx_module: Any, logprobs: Any, top_p: float) -> Any:
    """Nucleus filtering without an inverse-permutation scatter.

    Rather than reorder filtered log-probabilities back to vocabulary
    order, this implementation finds the smallest log-probability that
    falls inside the top-p nucleus and applies that threshold directly to
    the original logprob tensor. The result is mathematically equivalent
    to the canonical "sort + mask + unsort" recipe but only requires
    ``sort``, ``cumsum``, and ``where``, all of which are supported on
    MLX, the NumPy stand-ins used in tests, and any future runtime.
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logprobs
    sorted_logprobs = mx_module.sort(logprobs, axis=-1)
    sorted_probs = mx_module.exp(sorted_logprobs)
    sorted_probs_desc = sorted_probs[..., ::-1]
    cumulative_from_top_desc = mx_module.cumsum(sorted_probs_desc, axis=-1)
    keep_desc = cumulative_from_top_desc <= mx_module.array(top_p, dtype=cumulative_from_top_desc.dtype)
    keep_largest = mx_module.ones_like(keep_desc[..., :1])
    keep_desc = mx_module.concatenate([keep_largest, keep_desc[..., :-1]], axis=-1)
    keep_asc = keep_desc[..., ::-1]
    plus_inf = mx_module.array(float("inf"), dtype=sorted_logprobs.dtype)
    kept_logprobs = mx_module.where(keep_asc, sorted_logprobs, plus_inf)
    threshold = kept_logprobs.min(axis=-1, keepdims=True)
    return mx_module.where(logprobs >= threshold, logprobs, _neg_inf_like(mx_module, logprobs))


class _CategoricalSampler:
    """Configurable temperature / top-k / top-p / min-p sampler."""

    def __init__(self, config: SamplerConfig, mx_module: Any | None = None) -> None:
        self._config = config
        self._mx = mx_module if mx_module is not None else mx
        self._rng = None
        if (
            config.seed is not None
            and hasattr(self._mx, "random")
            and hasattr(self._mx.random, "key")
        ):
            self._rng = self._mx.random.key(int(config.seed))

    def __call__(self, logits: Any) -> Any:
        config = self._config
        mx_module = self._mx
        scaled = _apply_temperature(mx_module, logits, config.temperature)
        logprobs = scaled - mx_module.logsumexp(scaled, axis=-1, keepdims=True)
        logprobs = _apply_top_k(mx_module, logprobs, config.top_k)
        logprobs = _apply_min_p(mx_module, logprobs, config.min_p)
        logprobs = _apply_top_p(mx_module, logprobs, config.top_p)
        if hasattr(mx_module, "random") and hasattr(mx_module.random, "categorical"):
            kwargs = {"axis": -1}
            if self._rng is not None:
                kwargs["key"] = self._rng
            return mx_module.random.categorical(logprobs, **kwargs)
        return _categorical_numpy_fallback(mx_module, logprobs, seed=config.seed)


def _categorical_numpy_fallback(mx_module: Any, logprobs: Any, *, seed: int | None) -> Any:
    """Categorical sampling for environments without ``mlx.core.random``.

    This branch only runs under the test stand-in and bears the cost of an
    MLX → NumPy roundtrip per call. It is preserved so the sampler API can
    be unit-tested without the MLX runtime.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    probabilities = np.exp(np.asarray(logprobs))
    probabilities = probabilities / np.maximum(probabilities.sum(axis=-1, keepdims=True), 1e-12)
    if probabilities.ndim == 1:
        return mx_module.array(int(rng.choice(len(probabilities), p=probabilities)))
    leading = probabilities.shape[:-1]
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    samples = np.array([rng.choice(flat.shape[-1], p=row) for row in flat], dtype=np.int64)
    return mx_module.array(samples.reshape(leading))


def make_sampler(
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    seed: int | None = None,
    mx_module: Any | None = None,
) -> _Sampler:
    """Build a sampler callable from a declarative configuration.

    The returned object exposes ``__call__(logits)`` and adheres to the
    contract documented at the module level. Greedy sampling is selected
    automatically when ``temperature == 0`` so callers do not have to
    branch on the parameter.
    """
    config = SamplerConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        seed=seed,
    )
    if mx_module is None:
        _require_mlx_runtime()
    chosen_mx = mx_module if mx_module is not None else mx
    if config.is_greedy:
        return _GreedySampler(chosen_mx)
    return _CategoricalSampler(config, mx_module=chosen_mx)


__all__ = ["SamplerConfig", "make_sampler"]
