"""Reference outlier-aware mixed-precision key-path quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantizers import MSEPayload, ProdPayload, TurboQuantMSERef, TurboQuantProdRef


@dataclass(slots=True)
class MixedPrecisionProfile:
    """User-facing mixed-precision quantization profile.

    ``dim`` is the head dimension and defaults to ``0`` so callers may build
    a partial profile (e.g. from a CLI flag set) and have the surrounding
    runtime fill it in with the model's head dimension before quantization.
    Quantizers reject profiles with ``dim == 0`` so the placeholder default
    is never silently accepted at runtime.
    """

    outlier_channels: int
    outlier_high_bits: int
    regular_bits: int
    selection_policy: str = "calibration_variance"
    mode: str = "mse"
    dim: int = 0

    @property
    def effective_bits(self) -> float:
        if self.dim <= 0:
            raise ValueError("MixedPrecisionProfile.dim must be set before reading effective_bits")
        if self.outlier_channels >= self.dim:
            raise ValueError("outlier_channels must be strictly less than dim")
        total = (self.outlier_channels * self.outlier_high_bits) + (
            self.regular_bits * (self.dim - self.outlier_channels)
        )
        return total / float(self.dim)


@dataclass(slots=True)
class MixedPrecisionPayload:
    outlier_mask: np.ndarray
    regular_payload: MSEPayload | ProdPayload
    outlier_payload: MSEPayload | ProdPayload
    head_dim: int
    profile_name: str


class MixedPrecisionKeyPathRef:
    """Split channels into outlier/non-outlier groups and quantize separately."""

    def __init__(
        self,
        head_dim: int,
        profile: MixedPrecisionProfile,
        *,
        codebook_version: str = "v1-lloyd-max-beta",
        rotation_kind: str = "qr",
        rotation_seed: int = 0,
        qjl_seed: int = 1,
    ):
        if profile.outlier_channels <= 1 or profile.outlier_channels >= head_dim - 1:
            raise ValueError("outlier_channels must leave both regular and outlier partitions with dim >= 2")
        self.head_dim = head_dim
        self.profile = MixedPrecisionProfile(
            outlier_channels=profile.outlier_channels,
            outlier_high_bits=profile.outlier_high_bits,
            regular_bits=profile.regular_bits,
            selection_policy=profile.selection_policy,
            mode=profile.mode,
            dim=head_dim,
        )
        self.codebook_version = codebook_version
        self.rotation_kind = rotation_kind
        self.rotation_seed = rotation_seed
        self.qjl_seed = qjl_seed
        self._frozen_outlier_mask: np.ndarray | None = None
        self._rolling_count = 0
        self._rolling_sum: np.ndarray | None = None
        self._rolling_sq_sum: np.ndarray | None = None
        quantizer_cls = TurboQuantProdRef if profile.mode == "prod" else TurboQuantMSERef
        quantizer_kwargs = {
            "codebook_version": codebook_version,
            "rotation_kind": rotation_kind,
            "rotation_seed": rotation_seed,
        }
        if quantizer_cls is TurboQuantProdRef:
            quantizer_kwargs["qjl_seed"] = qjl_seed
        self.regular = quantizer_cls(
            head_dim - profile.outlier_channels,
            profile.regular_bits,
            **quantizer_kwargs,
        )
        self.outlier = quantizer_cls(
            profile.outlier_channels,
            profile.outlier_high_bits,
            **quantizer_kwargs,
        )

    @property
    def frozen_outlier_mask(self) -> np.ndarray | None:
        if self._frozen_outlier_mask is None:
            return None
        return self._frozen_outlier_mask.copy()

    def _variance_from_rolling_state(self) -> np.ndarray:
        if self._rolling_count <= 0 or self._rolling_sum is None or self._rolling_sq_sum is None:
            raise ValueError("rolling variance requested before any calibration vectors were observed")
        mean = self._rolling_sum / max(self._rolling_count, 1)
        variance = (self._rolling_sq_sum / max(self._rolling_count, 1)) - np.square(mean)
        return variance.astype(np.float32)

    def _select_from_variance(self, variances: np.ndarray) -> np.ndarray:
        indices = np.argsort(variances)[-self.profile.outlier_channels :]
        mask = np.zeros((self.head_dim,), dtype=bool)
        mask[indices] = True
        return mask

    def _channel_variance(self, calibration_vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(calibration_vectors, dtype=np.float32).reshape(-1, self.head_dim)
        if self.profile.selection_policy == "rolling_variance":
            if vectors.size == 0:
                return self._variance_from_rolling_state()
            batch_sum = np.sum(vectors, axis=0, dtype=np.float64)
            batch_sq_sum = np.sum(np.square(vectors, dtype=np.float64), axis=0, dtype=np.float64)
            batch_count = int(vectors.shape[0])
            self._rolling_sum = batch_sum if self._rolling_sum is None else self._rolling_sum + batch_sum
            self._rolling_sq_sum = (
                batch_sq_sum if self._rolling_sq_sum is None else self._rolling_sq_sum + batch_sq_sum
            )
            self._rolling_count += batch_count
            return self._variance_from_rolling_state()
        return np.var(vectors, axis=0, dtype=np.float32)

    def select_outliers(self, calibration_vectors: np.ndarray) -> np.ndarray:
        variances = self._channel_variance(calibration_vectors)
        return self._select_from_variance(variances)

    def calibrate(self, calibration_vectors: np.ndarray) -> np.ndarray:
        if self._frozen_outlier_mask is not None:
            return self._frozen_outlier_mask.copy()
        return self.select_outliers(calibration_vectors)

    def freeze(self, outlier_mask: np.ndarray | None = None, calibration_vectors: np.ndarray | None = None) -> np.ndarray:
        if self._frozen_outlier_mask is not None:
            return self._frozen_outlier_mask.copy()
        if outlier_mask is None:
            if calibration_vectors is None and self.profile.selection_policy == "rolling_variance" and self._rolling_count > 0:
                outlier_mask = self._select_from_variance(self._variance_from_rolling_state())
            elif calibration_vectors is None:
                raise ValueError("freeze requires either outlier_mask or calibration_vectors")
            else:
                outlier_mask = self.select_outliers(calibration_vectors)
        mask = np.asarray(outlier_mask, dtype=bool)
        if mask.shape != (self.head_dim,):
            raise ValueError(f"outlier_mask must have shape ({self.head_dim},)")
        if int(mask.sum()) != self.profile.outlier_channels:
            raise ValueError("outlier_mask does not match configured outlier channel budget")
        self._frozen_outlier_mask = mask.copy()
        return self._frozen_outlier_mask.copy()

    def quantize(self, x: np.ndarray, outlier_mask: np.ndarray | None = None) -> MixedPrecisionPayload:
        x = np.asarray(x, dtype=np.float32)
        if outlier_mask is not None:
            mask = self.freeze(outlier_mask=outlier_mask)
        elif self._frozen_outlier_mask is not None:
            mask = self._frozen_outlier_mask.copy()
        else:
            mask = self.freeze(calibration_vectors=x.reshape(-1, x.shape[-1]))
        regular = self.regular.quantize(x[..., ~mask])
        outlier = self.outlier.quantize(x[..., mask])
        return MixedPrecisionPayload(
            outlier_mask=mask,
            regular_payload=regular,
            outlier_payload=outlier,
            head_dim=self.head_dim,
            profile_name=f"{self.profile.mode}:{self.profile.regular_bits}+{self.profile.outlier_high_bits}@{self.profile.outlier_channels}",
        )

    def dequantize(self, payload: MixedPrecisionPayload) -> np.ndarray:
        regular = self.regular.dequantize(payload.regular_payload)
        outlier = self.outlier.dequantize(payload.outlier_payload)
        out = np.zeros(regular.shape[:-1] + (payload.head_dim,), dtype=np.float32)
        out[..., ~payload.outlier_mask] = regular
        out[..., payload.outlier_mask] = outlier
        return out

    def score_query(self, query: np.ndarray, payload: MixedPrecisionPayload) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        regular = self.regular.score_query(query[..., ~payload.outlier_mask], payload.regular_payload)
        outlier = self.outlier.score_query(query[..., payload.outlier_mask], payload.outlier_payload)
        return regular + outlier

    def score_matrix(self, queries: np.ndarray, payload: MixedPrecisionPayload) -> np.ndarray:
        queries = np.asarray(queries, dtype=np.float32)
        regular = self.regular.score_matrix(queries[..., ~payload.outlier_mask], payload.regular_payload)
        outlier = self.outlier.score_matrix(queries[..., payload.outlier_mask], payload.outlier_payload)
        return regular + outlier
