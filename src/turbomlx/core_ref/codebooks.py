"""Dimension-keyed Lloyd-Max codebooks for TurboQuant."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from math import lgamma, log, pi
from pathlib import Path

import numpy as np

CODEBOOK_VERSION = "v1-lloyd-max-beta"
_EPS = 1e-8


def _xdg_cache_home() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".cache"


def default_cache_dir() -> Path:
    """Return the persistent root cache directory for codebooks and rotations.

    Precedence:
      1. ``TURBOMLX_CACHE_DIR`` environment variable (explicit user override).
      2. ``${XDG_CACHE_HOME}/turbomlx`` for XDG-compliant environments.
      3. ``~/.cache/turbomlx`` as the cross-platform persistent default.
    """
    configured = os.environ.get("TURBOMLX_CACHE_DIR")
    if configured:
        return Path(configured)
    return _xdg_cache_home() / "turbomlx"


@dataclass(frozen=True, slots=True)
class CodebookKey:
    head_dim: int
    bits: int
    version: str = CODEBOOK_VERSION

    @property
    def identifier(self) -> str:
        return f"{self.version}-d{self.head_dim}-b{self.bits}"


@dataclass(slots=True)
class CodebookEntry:
    key: CodebookKey
    centroids: np.ndarray
    boundaries: np.ndarray
    mse: float

    @property
    def levels(self) -> int:
        return int(self.centroids.shape[0])


def beta_pdf(x: np.ndarray, head_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    clipped = np.clip(x, -1.0 + _EPS, 1.0 - _EPS)
    log_norm = lgamma(head_dim / 2.0) - 0.5 * log(pi) - lgamma((head_dim - 1.0) / 2.0)
    log_pdf = log_norm + ((head_dim - 3.0) / 2.0) * np.log1p(-(clipped * clipped))
    return np.exp(log_pdf)


def _normalized_beta_grid(head_dim: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-1.0 + _EPS, 1.0 - _EPS, grid_size, dtype=np.float64)
    pdf = beta_pdf(grid, head_dim)
    pdf = pdf / np.trapezoid(pdf, grid)
    return grid, pdf


def _masked_trapz(values: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.trapezoid(values * weights, grid))


def generate_lloyd_max_codebook(
    head_dim: int,
    bits: int,
    *,
    max_iter: int = 256,
    tol: float = 1e-12,
    grid_size: int = 32769,
) -> CodebookEntry:
    if head_dim < 2:
        raise ValueError("head_dim must be >= 2")
    if bits < 1:
        raise ValueError("bits must be >= 1")

    key = CodebookKey(head_dim=head_dim, bits=bits)
    levels = 1 << bits
    grid, pdf = _normalized_beta_grid(head_dim, grid_size)

    quantiles = np.linspace(0.0, 1.0, levels + 2, dtype=np.float64)[1:-1]
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * np.diff(grid) * 0.5)
    cdf = np.concatenate([[0.0], cdf])
    cdf[-1] = 1.0
    centroids = np.interp(quantiles, cdf, grid)

    for _ in range(max_iter):
        boundaries = (centroids[:-1] + centroids[1:]) * 0.5
        edges = np.concatenate(([-1.0], boundaries, [1.0]))
        updated = np.empty_like(centroids)

        for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if index == levels - 1:
                mask = (grid >= lo) & (grid <= hi)
            else:
                mask = (grid >= lo) & (grid < hi)
            if not np.any(mask):
                updated[index] = (lo + hi) * 0.5
                continue
            segment_grid = grid[mask]
            segment_pdf = pdf[mask]
            denom = _masked_trapz(np.ones_like(segment_grid), segment_pdf, segment_grid)
            if denom <= 0.0:
                updated[index] = (lo + hi) * 0.5
                continue
            numer = _masked_trapz(segment_grid, segment_pdf, segment_grid)
            updated[index] = numer / denom

        if np.max(np.abs(updated - centroids)) <= tol:
            centroids = updated
            break
        centroids = updated

    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    edges = np.concatenate(([-1.0], boundaries, [1.0]))
    mse = 0.0
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == levels - 1:
            mask = (grid >= lo) & (grid <= hi)
        else:
            mask = (grid >= lo) & (grid < hi)
        if not np.any(mask):
            continue
        segment_grid = grid[mask]
        segment_pdf = pdf[mask]
        mse += _masked_trapz((segment_grid - centroids[index]) ** 2, segment_pdf, segment_grid)

    return CodebookEntry(
        key=key,
        centroids=centroids.astype(np.float32),
        boundaries=boundaries.astype(np.float32),
        mse=float(mse),
    )


class CodebookStore:
    """Persistent codebook store keyed by `(head_dim, bits, version)`."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = (cache_dir or default_cache_dir()) / "codebooks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: CodebookKey) -> Path:
        return self.cache_dir / f"{key.identifier}.npz"

    def load(self, key: CodebookKey) -> CodebookEntry:
        payload = np.load(self.path_for(key))
        return CodebookEntry(
            key=key,
            centroids=payload["centroids"].astype(np.float32),
            boundaries=payload["boundaries"].astype(np.float32),
            mse=float(payload["mse"]),
        )

    def save(self, entry: CodebookEntry) -> CodebookEntry:
        np.savez_compressed(
            self.path_for(entry.key),
            centroids=entry.centroids.astype(np.float32),
            boundaries=entry.boundaries.astype(np.float32),
            mse=np.array(entry.mse, dtype=np.float64),
        )
        return entry

    def get_or_create(self, key: CodebookKey) -> CodebookEntry:
        path = self.path_for(key)
        if path.exists():
            return self.load(key)
        return self.save(generate_lloyd_max_codebook(key.head_dim, key.bits))

    def generate_many(self, keys: Iterable[CodebookKey]) -> list[CodebookEntry]:
        return [self.get_or_create(key) for key in keys]
