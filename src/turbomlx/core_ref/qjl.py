"""Shared QJL artefacts keyed by head dimension."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codebooks import default_cache_dir


@dataclass(slots=True)
class QJLSpec:
    head_dim: int
    seed: int
    matrix: np.ndarray

    @property
    def identifier(self) -> str:
        return f"qjl-d{self.head_dim}-s{self.seed}"


def _qjl_cache_dir(cache_dir: Path | None = None) -> Path:
    path = (cache_dir or default_cache_dir()) / "qjl"
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_qjl(head_dim: int, *, seed: int = 0) -> QJLSpec:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((head_dim, head_dim), dtype=np.float64).astype(np.float32)
    return QJLSpec(head_dim=head_dim, seed=seed, matrix=matrix)


def load_or_create_qjl(
    head_dim: int,
    *,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> QJLSpec:
    path = _qjl_cache_dir(cache_dir) / f"qjl-d{head_dim}-s{seed}.npz"
    if path.exists():
        payload = np.load(path)
        return QJLSpec(
            head_dim=int(payload["head_dim"]),
            seed=int(payload["seed"]),
            matrix=payload["matrix"].astype(np.float32),
        )
    spec = generate_qjl(head_dim, seed=seed)
    np.savez_compressed(
        path,
        head_dim=np.array(spec.head_dim, dtype=np.int32),
        seed=np.array(spec.seed, dtype=np.int32),
        matrix=spec.matrix.astype(np.float32),
    )
    return spec


def qjl_quantize_signs(vector: np.ndarray, spec: QJLSpec) -> np.ndarray:
    projected = np.asarray(vector, dtype=np.float32) @ spec.matrix.T
    signs = np.sign(projected).astype(np.float32)
    signs[signs == 0.0] = 1.0
    return signs.astype(np.int8)


def qjl_dequantize(signs: np.ndarray, gamma: np.ndarray | float, spec: QJLSpec) -> np.ndarray:
    signs_f = np.asarray(signs, dtype=np.float32)
    gamma_f = np.asarray(gamma, dtype=np.float32)
    factor = np.float32(np.sqrt(np.pi / 2.0) / float(spec.head_dim))
    return gamma_f[..., None] * factor * (signs_f @ spec.matrix)


def qjl_score_correction(
    query: np.ndarray,
    signs: np.ndarray,
    gamma: np.ndarray | float,
    spec: QJLSpec,
) -> np.ndarray:
    query_f = np.asarray(query, dtype=np.float32)
    signs_f = np.asarray(signs, dtype=np.float32)
    gamma_f = np.asarray(gamma, dtype=np.float32)
    factor = np.float32(np.sqrt(np.pi / 2.0) / float(spec.head_dim))
    projected_query = query_f @ spec.matrix.T
    return gamma_f * factor * np.sum(projected_query * signs_f, axis=-1)
