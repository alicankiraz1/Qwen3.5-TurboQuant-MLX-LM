"""Optional MLX / mlx-lm dependency helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from importlib import import_module
import subprocess
import sys

from turbomlx.exceptions import MissingDependencyError, UnsupportedRuntimeVersionError


_SUPPORTED_MLX_MIN = (0, 31, 2)
_SUPPORTED_MLX_MAX_EXCLUSIVE = (0, 32, 0)
_SUPPORTED_MLX_LM_MIN = (0, 31, 3)
_SUPPORTED_MLX_LM_MAX_EXCLUSIVE = (0, 32, 0)


_PROBE_SNIPPET = (
    "import importlib;"
    "importlib.import_module('mlx.core');"
    "importlib.import_module('mlx_lm.models.cache')"
)


def _parse_version(version_text: str) -> tuple[int, int, int]:
    numeric_parts: list[int] = []
    for part in version_text.replace("-", ".").split("."):
        if part.isdigit():
            numeric_parts.append(int(part))
        else:
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                numeric_parts.append(int(digits))
        if len(numeric_parts) == 3:
            break
    while len(numeric_parts) < 3:
        numeric_parts.append(0)
    return tuple(numeric_parts[:3])


def _version_in_supported_range(
    version_text: str,
    *,
    minimum: tuple[int, int, int],
    maximum_exclusive: tuple[int, int, int],
) -> bool:
    version_tuple = _parse_version(version_text)
    return minimum <= version_tuple < maximum_exclusive


def _format_supported_range(
    package: str,
    minimum: tuple[int, int, int],
    maximum_exclusive: tuple[int, int, int],
) -> str:
    return f"{package}>={'.'.join(str(part) for part in minimum)},<{'.'.join(str(part) for part in maximum_exclusive)}"


def ensure_supported_runtime_versions() -> dict[str, str]:
    try:
        mlx_version = metadata.version("mlx")
        mlx_lm_version = metadata.version("mlx-lm")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - covered by ensure_mlx_runtime
        raise MissingDependencyError(
            "MLX runtime dependencies are missing. Install `turbomlx[mlx]` to use the runtime backend."
        ) from exc

    incompatibilities: list[str] = []
    requirements = (
        ("mlx", mlx_version, _SUPPORTED_MLX_MIN, _SUPPORTED_MLX_MAX_EXCLUSIVE),
        ("mlx-lm", mlx_lm_version, _SUPPORTED_MLX_LM_MIN, _SUPPORTED_MLX_LM_MAX_EXCLUSIVE),
    )
    for package_name, version_text, minimum, maximum_exclusive in requirements:
        if not _version_in_supported_range(
            version_text, minimum=minimum, maximum_exclusive=maximum_exclusive
        ):
            incompatibilities.append(f"{package_name}=={version_text}")

    if incompatibilities:
        supported = " and ".join(
            _format_supported_range(name, minimum, maximum_exclusive)
            for name, _, minimum, maximum_exclusive in requirements
        )
        raise UnsupportedRuntimeVersionError(
            "TurboMLX preview is tested against "
            f"{supported}; got {', '.join(incompatibilities)}."
        )

    return {"mlx": mlx_version, "mlx-lm": mlx_lm_version}


@lru_cache(maxsize=1)
def mlx_runtime_available() -> bool:
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE_SNIPPET],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def ensure_mlx_runtime():
    if not mlx_runtime_available():
        raise MissingDependencyError(
            "MLX runtime dependencies are missing. Install `turbomlx[mlx]` to use the runtime backend."
        )
    ensure_supported_runtime_versions()
    mx = import_module("mlx.core")
    base = import_module("mlx_lm.models.base")
    cache = import_module("mlx_lm.models.cache")
    return mx, base, cache
