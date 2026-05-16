"""Versioned TurboMLX runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from turbomlx.exceptions import UnsupportedConfigurationError


class TurboQuantMode(str, Enum):
    MSE = "mse"
    PROD = "prod"


class ValuesMode(str, Enum):
    DENSE = "dense"
    AFFINE = "affine"


class ScorerMode(str, Enum):
    ORACLE_PREVIEW = "oracle_preview"
    NATIVE_MLX = "native_mlx"


class OutlierSelectionPolicy(str, Enum):
    CALIBRATION_VARIANCE = "calibration_variance"
    ROLLING_VARIANCE = "rolling_variance"


def default_mode_for_bits(bits_total: int, *, unbiased_score: bool = False) -> TurboQuantMode:
    if bits_total < 1:
        raise UnsupportedConfigurationError("bits_total must be >= 1")
    if unbiased_score or bits_total <= 2:
        return TurboQuantMode.PROD if bits_total >= 2 else TurboQuantMode.MSE
    return TurboQuantMode.MSE


@dataclass(slots=True)
class MixedPrecisionProfileConfig:
    outlier_channels: int
    outlier_high_bits: int
    regular_bits: int
    outlier_selection_policy: OutlierSelectionPolicy = OutlierSelectionPolicy.CALIBRATION_VARIANCE
    calibration_prefix_tokens: int = 256

    @property
    def enabled(self) -> bool:
        return self.outlier_channels > 0

    @property
    def profile_name(self) -> str:
        return (
            f"{self.outlier_selection_policy.value}:"
            f"outliers={self.outlier_channels},"
            f"hi={self.outlier_high_bits},lo={self.regular_bits}"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "outlier_channels": self.outlier_channels,
            "outlier_high_bits": self.outlier_high_bits,
            "regular_bits": self.regular_bits,
            "outlier_selection_policy": self.outlier_selection_policy.value,
            "calibration_prefix_tokens": self.calibration_prefix_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, int | str]) -> MixedPrecisionProfileConfig:
        return cls(
            outlier_channels=int(payload["outlier_channels"]),
            outlier_high_bits=int(payload["outlier_high_bits"]),
            regular_bits=int(payload["regular_bits"]),
            outlier_selection_policy=OutlierSelectionPolicy(payload["outlier_selection_policy"]),
            calibration_prefix_tokens=int(payload.get("calibration_prefix_tokens", 256)),
        )


@dataclass(slots=True)
class TurboQuantConfig:
    bits_total: int
    mode: TurboQuantMode | None = None
    values_mode: ValuesMode = ValuesMode.DENSE
    value_bits: int = 4
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW
    quantizer_version: str = "turbomlx-v0.1"
    codebook_version: str = "v1-lloyd-max-beta"
    rotation_kind: str = "qr"
    rotation_seed: int = 0
    qjl_seed: int = 1
    quantized_kv_start: int = 0
    packing_layout: str = "little_endian_compact"
    mixed_precision: MixedPrecisionProfileConfig | None = None

    def __post_init__(self):
        self.mode = self.mode or default_mode_for_bits(self.bits_total)
        if self.bits_total < 1:
            raise UnsupportedConfigurationError("bits_total must be >= 1")
        if self.mode == TurboQuantMode.PROD and self.bits_total < 2:
            raise UnsupportedConfigurationError("mode=prod requires bits_total >= 2")
        if self.values_mode == ValuesMode.AFFINE and self.value_bits < 1:
            raise UnsupportedConfigurationError("value_bits must be >= 1 for affine values")
        if (
            self.mixed_precision is not None
            and self.mixed_precision.outlier_high_bits <= self.mixed_precision.regular_bits
        ):
            raise UnsupportedConfigurationError(
                "outlier_high_bits must be greater than regular_bits in mixed precision mode"
            )

    @property
    def bits_mse(self) -> int:
        return self.bits_total - 1 if self.mode == TurboQuantMode.PROD else self.bits_total

    @property
    def values_mode_name(self) -> str:
        return self.values_mode.value

    @property
    def scorer_mode_name(self) -> str:
        return self.scorer_mode.value

    @property
    def mixed_precision_profile(self) -> str:
        return self.mixed_precision.profile_name if self.mixed_precision else "disabled"

    def codebook_id(self, head_dim: int) -> str:
        return f"{self.codebook_version}-d{head_dim}-b{self.bits_mse}"
