"""Centralized logging configuration for TurboMLX.

TurboMLX follows the Python logging best practice of attaching a
``NullHandler`` to the package root logger so library consumers control
output. Internal modules should call :func:`get_logger` instead of
:func:`logging.getLogger` so the package logger hierarchy stays predictable
and instrumentation hooks (filters, structured logging) can be added in one
place.

Activate diagnostic output from a host application or test with::

    import logging
    logging.getLogger("turbomlx").setLevel(logging.DEBUG)

The ``TURBOMLX_LOG_LEVEL`` environment variable may also be used to enable
package-level diagnostic output without code changes.
"""

from __future__ import annotations

import logging
import os
from typing import Final

_ROOT_LOGGER_NAME: Final[str] = "turbomlx"
_ENV_LEVEL: Final[str] = "TURBOMLX_LOG_LEVEL"


def _resolve_env_level() -> int | None:
    value = os.environ.get(_ENV_LEVEL)
    if not value:
        return None
    if value.isdigit():
        return int(value)
    return logging.getLevelName(value.upper()) if value.strip() else None


def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not any(isinstance(handler, logging.NullHandler) for handler in root.handlers):
        root.addHandler(logging.NullHandler())
    env_level = _resolve_env_level()
    if isinstance(env_level, int):
        root.setLevel(env_level)
    return root


_PACKAGE_LOGGER: Final[logging.Logger] = _configure_root_logger()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a :mod:`logging` logger scoped under the ``turbomlx`` namespace.

    Passing ``None`` or the package name yields the package root logger.
    Submodule names are accepted either as a fully qualified module path
    (``"turbomlx.mlx_runtime.cache"``) or as a short suffix
    (``"mlx_runtime.cache"``). In both cases the returned logger lives under
    the ``turbomlx`` hierarchy.
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return _PACKAGE_LOGGER
    if name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


__all__ = ["get_logger"]
