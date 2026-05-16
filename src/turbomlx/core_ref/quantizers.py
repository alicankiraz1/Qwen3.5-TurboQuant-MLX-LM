"""Reference TurboQuantMSE / TurboQuantProd implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import resolve_shared_artifacts
from .codebooks import CODEBOOK_VERSION
from .packing import pack_bits, pack_sign_bits, unpack_bits, unpack_sign_bits
from .qjl import QJLSpec, qjl_dequantize, qjl_quantize_signs, qjl_score_correction
from .rotation import apply_inverse_rotation, apply_rotation

_EPS = 1e-8


@dataclass(slots=True)
class MSEPayload:
    packed_indices: np.ndarray
    norms: np.ndarray
    head_dim: int
    bits_total: int
    codebook_id: str


@dataclass(slots=True)
class ProdPayload:
    packed_indices: np.ndarray
    packed_signs: np.ndarray
    norms: np.ndarray
    residual_norms: np.ndarray
    head_dim: int
    bits_total: int
    bits_mse: int
    codebook_id: str


def _vector_norms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(x, dtype=np.float32), axis=-1, keepdims=True)).astype(np.float32)


class TurboQuantMSERef:
    """Reference implementation of key-path TurboQuantmse."""

    def __init__(
        self,
        head_dim: int,
        bits_total: int,
        *,
        codebook_version: str = CODEBOOK_VERSION,
        rotation_kind: str = "qr",
        rotation_seed: int = 0,
    ):
        self.head_dim = head_dim
        self.bits_total = bits_total
        artifacts = resolve_shared_artifacts(
            head_dim,
            bits_total,
            codebook_version=codebook_version,
            rotation_kind=rotation_kind,
            rotation_seed=rotation_seed,
        )
        self.codebook = artifacts.codebook
        self.rotation = artifacts.rotation
        self.centroids = self.codebook.centroids
        self.boundaries = self.codebook.boundaries
        self._max_index = np.uint8(self.centroids.shape[0] - 1)

    @property
    def codebook_id(self) -> str:
        return self.codebook.key.identifier

    def validate_payload(self, payload: MSEPayload) -> None:
        if payload.head_dim != self.head_dim:
            raise ValueError(f"Payload head_dim={payload.head_dim} does not match quantizer head_dim={self.head_dim}")
        if payload.bits_total != self.bits_total:
            raise ValueError(
                f"Payload bits_total={payload.bits_total} does not match quantizer bits_total={self.bits_total}"
            )
        if payload.codebook_id != self.codebook_id:
            raise ValueError(
                f"Payload codebook_id={payload.codebook_id} does not match quantizer codebook_id={self.codebook_id}"
            )

    def _normalize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norms = _vector_norms(x)
        return x.astype(np.float32) / np.maximum(norms, _EPS), norms

    def _quantize_unit(self, x_unit: np.ndarray) -> np.ndarray:
        """Map rotated entries to their nearest centroid.

        Lloyd-Max boundaries are the optimal midpoints between adjacent
        centroids on the post-rotation distribution. ``np.searchsorted``
        therefore finds the nearest centroid in ``O(N log K)`` time without
        materializing the broadcast ``[..., head_dim, K]`` difference
        tensor used by the previous ``argmin`` implementation, which could
        allocate several gigabytes for long-context Qwen layers.
        """
        rotated = apply_rotation(x_unit, self.rotation)
        indices = np.searchsorted(self.boundaries, rotated, side="right")
        if indices.dtype != np.uint8:
            indices = np.clip(indices, 0, int(self._max_index)).astype(np.uint8)
        else:
            np.clip(indices, 0, self._max_index, out=indices)
        return indices

    def _dequantize_unit(self, packed_indices: np.ndarray, leading_shape: tuple[int, ...]) -> np.ndarray:
        indices = unpack_bits(packed_indices, self.bits_total, self.head_dim).reshape(leading_shape + (self.head_dim,))
        rotated = self.centroids[indices]
        return apply_inverse_rotation(rotated, self.rotation)

    def quantize(self, x: np.ndarray) -> MSEPayload:
        x = np.asarray(x, dtype=np.float32)
        x_unit, norms = self._normalize(x)
        indices = self._quantize_unit(x_unit)
        packed = pack_bits(indices, self.bits_total)
        return MSEPayload(
            packed_indices=packed,
            norms=norms.astype(np.float32),
            head_dim=self.head_dim,
            bits_total=self.bits_total,
            codebook_id=self.codebook_id,
        )

    def dequantize(self, payload: MSEPayload) -> np.ndarray:
        self.validate_payload(payload)
        unit = self._dequantize_unit(payload.packed_indices, payload.norms.shape[:-1])
        return unit * payload.norms

    def score_query(self, query: np.ndarray, payload: MSEPayload) -> np.ndarray:
        self.validate_payload(payload)
        leading_shape = payload.norms.shape[:-1]
        indices = unpack_bits(payload.packed_indices, self.bits_total, self.head_dim).reshape(leading_shape + (self.head_dim,))
        rotated_query = apply_rotation(np.asarray(query, dtype=np.float32), self.rotation)
        rotated_values = self.centroids[indices]
        main_scores = np.sum(rotated_query * rotated_values, axis=-1)
        return main_scores * np.squeeze(payload.norms, axis=-1)

    def score_matrix(self, queries: np.ndarray, payload: MSEPayload) -> np.ndarray:
        self.validate_payload(payload)
        leading_shape = payload.norms.shape[:-1]
        indices = unpack_bits(payload.packed_indices, self.bits_total, self.head_dim).reshape(leading_shape + (self.head_dim,))
        rotated_queries = apply_rotation(np.asarray(queries, dtype=np.float32), self.rotation)
        rotated_values = self.centroids[indices]
        scores = np.einsum("...qd,...kd->...qk", rotated_queries, rotated_values, optimize=True)
        return scores * np.squeeze(payload.norms, axis=-1)[..., None, :]

    def nbytes(self, payload: MSEPayload) -> int:
        return int(payload.packed_indices.nbytes + payload.norms.nbytes)


class TurboQuantProdRef:
    """Reference implementation of key-path TurboQuantprod."""

    def __init__(
        self,
        head_dim: int,
        bits_total: int,
        *,
        codebook_version: str = CODEBOOK_VERSION,
        rotation_kind: str = "qr",
        rotation_seed: int = 0,
        qjl_seed: int = 1,
    ):
        if bits_total < 2:
            raise ValueError("TurboQuantprod requires bits_total >= 2")
        self.head_dim = head_dim
        self.bits_total = bits_total
        self.bits_mse = bits_total - 1
        self.mse = TurboQuantMSERef(
            head_dim,
            self.bits_mse,
            codebook_version=codebook_version,
            rotation_kind=rotation_kind,
            rotation_seed=rotation_seed,
        )
        self.qjl: QJLSpec = resolve_shared_artifacts(
            head_dim,
            self.bits_mse,
            codebook_version=codebook_version,
            rotation_kind=rotation_kind,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
        ).qjl

    @property
    def codebook_id(self) -> str:
        return self.mse.codebook_id

    def validate_payload(self, payload: ProdPayload) -> None:
        if payload.head_dim != self.head_dim:
            raise ValueError(f"Payload head_dim={payload.head_dim} does not match quantizer head_dim={self.head_dim}")
        if payload.bits_total != self.bits_total:
            raise ValueError(
                f"Payload bits_total={payload.bits_total} does not match quantizer bits_total={self.bits_total}"
            )
        if payload.bits_mse != self.bits_mse:
            raise ValueError(
                f"Payload bits_mse={payload.bits_mse} does not match quantizer bits_mse={self.bits_mse}"
            )
        if payload.codebook_id != self.codebook_id:
            raise ValueError(
                f"Payload codebook_id={payload.codebook_id} does not match quantizer codebook_id={self.codebook_id}"
            )

    def quantize(self, x: np.ndarray) -> ProdPayload:
        x = np.asarray(x, dtype=np.float32)
        norms = _vector_norms(x)
        x_unit = x / np.maximum(norms, _EPS)
        mse_payload = self.mse.quantize(x_unit)
        approx_unit = self.mse.dequantize(
            MSEPayload(
                packed_indices=mse_payload.packed_indices,
                norms=np.ones_like(norms, dtype=np.float32),
                head_dim=self.head_dim,
                bits_total=self.bits_mse,
                codebook_id=mse_payload.codebook_id,
            )
        )
        residual = x_unit - approx_unit
        residual_norms = _vector_norms(residual)
        residual_unit = residual / np.maximum(residual_norms, _EPS)
        signs = qjl_quantize_signs(residual_unit, self.qjl)
        packed_signs = pack_sign_bits(signs)
        return ProdPayload(
            packed_indices=mse_payload.packed_indices,
            packed_signs=packed_signs,
            norms=norms.astype(np.float32),
            residual_norms=np.squeeze(residual_norms, axis=-1).astype(np.float32),
            head_dim=self.head_dim,
            bits_total=self.bits_total,
            bits_mse=self.bits_mse,
            codebook_id=self.codebook_id,
        )

    def dequantize(self, payload: ProdPayload) -> np.ndarray:
        self.validate_payload(payload)
        leading_shape = payload.norms.shape[:-1]
        unit_main = self.mse._dequantize_unit(payload.packed_indices, leading_shape)
        signs = unpack_sign_bits(payload.packed_signs, self.head_dim).reshape(leading_shape + (self.head_dim,))
        residual = qjl_dequantize(signs, payload.residual_norms, self.qjl)
        return (unit_main + residual) * payload.norms

    def score_query(self, query: np.ndarray, payload: ProdPayload) -> np.ndarray:
        self.validate_payload(payload)
        main_scores = self.mse.score_query(
            query,
            MSEPayload(
                packed_indices=payload.packed_indices,
                norms=payload.norms,
                head_dim=self.head_dim,
                bits_total=self.bits_mse,
                codebook_id=payload.codebook_id,
            ),
        )
        signs = unpack_sign_bits(payload.packed_signs, self.head_dim).reshape(payload.norms.shape[:-1] + (self.head_dim,))
        gamma = np.squeeze(payload.norms, axis=-1) * payload.residual_norms
        correction = qjl_score_correction(np.asarray(query, dtype=np.float32), signs, gamma, self.qjl)
        return main_scores + correction

    def score_matrix(self, queries: np.ndarray, payload: ProdPayload) -> np.ndarray:
        self.validate_payload(payload)
        main_scores = self.mse.score_matrix(
            queries,
            MSEPayload(
                packed_indices=payload.packed_indices,
                norms=payload.norms,
                head_dim=self.head_dim,
                bits_total=self.bits_mse,
                codebook_id=payload.codebook_id,
            ),
        )
        signs = unpack_sign_bits(payload.packed_signs, self.head_dim).reshape(payload.norms.shape[:-1] + (self.head_dim,))
        projected_queries = np.asarray(queries, dtype=np.float32) @ self.qjl.matrix.T
        factor = np.float32(np.sqrt(np.pi / 2.0) / float(self.head_dim))
        correction = np.einsum("...qd,...kd->...qk", projected_queries, signs, optimize=True)
        gamma = (np.squeeze(payload.norms, axis=-1) * payload.residual_norms)[..., None, :]
        return main_scores + (correction * gamma * factor)

    def nbytes(self, payload: ProdPayload) -> int:
        return int(
            payload.packed_indices.nbytes
            + payload.packed_signs.nbytes
            + payload.norms.nbytes
            + payload.residual_norms.nbytes
        )
