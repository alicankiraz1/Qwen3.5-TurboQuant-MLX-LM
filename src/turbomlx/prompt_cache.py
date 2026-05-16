"""TurboMLX-owned prompt-cache save/load helpers.

The prompt-cache format ships in three generations:

* **v1** — Legacy pickle payload with a free-form ``class_path``.
  Supported in read-only mode behind an allowlist and a deprecation
  warning. New caches are never written in this layout.
* **v2** — Pickle payload with a stable ``cache_type_id`` metadata key
  and a fallback ``class_path``. Continues to write and read for backward
  compatibility with v0.1 deployments.
* **v3** (default when the optional ``safetensors`` dependency is
  installed) — A JSON sidecar describing structural metadata plus a
  ``safetensors`` container holding the numerical arrays. v3 removes the
  arbitrary-class-loading attack surface, supports zero-copy loads on
  modern Apple Silicon, and stores cache type identifiers from a fixed
  allowlist.

The reader auto-detects the format from the magic header of the on-disk
file, so existing v1 / v2 caches continue to load without changes. Writes
default to v3 when ``safetensors`` is importable and otherwise transparently
fall back to v2.
"""

from __future__ import annotations

import io
import json
import pickle
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from turbomlx._logging import get_logger
from turbomlx.exceptions import PromptCacheSerializationError

_LOGGER = get_logger(__name__)

_PROMPT_CACHE_SCHEMA_VERSION = 3
_SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS = {1, 2, 3}
_DEFAULT_LEGACY_SCHEMA = 2
_SAFETENSORS_MAGIC = b"TQS3"  # 4-byte magic prefix introduced for the JSON header chunk
_TRUSTED_LOCAL_ONLY_WARNING = (
    "TurboMLX prompt-cache loading is trusted-local-only. "
    "Do not load prompt-cache files from untrusted sources."
)
_CLASS_PATH_FALLBACK_WARNING = (
    "TurboMLX prompt-cache restore fell back to deprecated class_path metadata. "
    "Re-save this cache to migrate it to the current schema."
)
_LEGACY_PICKLE_WARNING = (
    "TurboMLX is reading a legacy pickle-backed prompt-cache. "
    "Re-save it with TurboMLX>=0.2 to migrate to the safer safetensors layout."
)

_CACHE_TYPE_TO_CLASS_PATH: dict[str, str] = {
    "turbomlx.cache.turboquant_kv.v1": "turbomlx.mlx_runtime.cache.TurboQuantKVCache",
}
_TRUSTED_CLASS_PATHS: set[str] = set(_CACHE_TYPE_TO_CLASS_PATH.values())


def register_cache_type(cache_type_id: str, class_path: str) -> None:
    """Register a third-party cache class so the loader will accept it.

    This is the only supported way to widen the load-time class allowlist;
    arbitrary ``class_path`` values inside cache payloads are no longer
    honored by default.
    """
    _CACHE_TYPE_TO_CLASS_PATH[cache_type_id] = class_path
    _TRUSTED_CLASS_PATHS.add(class_path)
    _LOGGER.debug("Registered prompt-cache type %s -> %s", cache_type_id, class_path)


def _class_path(value: type) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _cache_type_id(value: type) -> str:
    return getattr(value, "PROMPT_CACHE_TYPE_ID", _class_path(value))


def _load_class(class_path: str) -> type:
    if class_path not in _TRUSTED_CLASS_PATHS:
        raise PromptCacheSerializationError(
            f"Prompt-cache refuses to import non-allowlisted class path: {class_path!r}. "
            "Use turbomlx.prompt_cache.register_cache_type to opt in to third-party classes."
        )
    module_name, _, attr_path = class_path.rpartition(".")
    if not module_name or not attr_path:
        raise PromptCacheSerializationError(f"Invalid prompt-cache class path: {class_path}")
    module = import_module(module_name)
    value: Any = module
    for part in attr_path.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise PromptCacheSerializationError(f"Prompt-cache class path did not resolve to a type: {class_path}")
    return value


def _load_cache_type(cache_type_id: str) -> type:
    class_path = _CACHE_TYPE_TO_CLASS_PATH.get(cache_type_id)
    if class_path is None:
        raise PromptCacheSerializationError(
            f"Unknown cache_type_id={cache_type_id!r}; register it via register_cache_type "
            "before loading caches produced by third-party packages."
        )
    return _load_class(class_path)


def _normalize_state(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_normalize_state(item) for item in value)
    if isinstance(value, list):
        return [_normalize_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_state(item) for key, item in value.items()}
    if isinstance(value, (str, bytes, bytearray, int, float, bool)):
        return value
    if hasattr(value, "shape") or hasattr(value, "__array__"):
        return np.asarray(value)
    return value


def _cache_extra_state(cache: Any) -> dict[str, Any]:
    if hasattr(cache, "prompt_cache_extra_state"):
        return _normalize_state(cache.prompt_cache_extra_state())
    return {}


def _serialize_prompt_cache_entry(cache: Any) -> dict[str, Any]:
    if not hasattr(cache, "state") or not hasattr(cache, "meta_state"):
        raise PromptCacheSerializationError(
            f"Prompt-cache entry {type(cache).__name__} must expose `state` and `meta_state`."
        )
    cache_cls = type(cache)
    return {
        "cache_type_id": _cache_type_id(cache_cls),
        "class_path": _class_path(cache_cls),
        "state": _normalize_state(cache.state),
        "meta_state": _normalize_state(cache.meta_state),
        "extra_state": _cache_extra_state(cache),
    }


def _safetensors_available() -> bool:
    try:
        import_module("safetensors.numpy")
    except ModuleNotFoundError:
        return False
    return True


def _flatten_arrays(prefix: str, value: Any) -> tuple[Any, dict[str, np.ndarray]]:
    """Replace numpy arrays inside ``value`` with structural placeholders.

    Returns the placeholder-rebuilt ``value`` plus a flat ``{tensor_id: array}``
    mapping suitable for the safetensors archive. The placeholders carry
    enough information (dtype, shape, tensor id) to rehydrate the original
    state tree on load.
    """
    arrays: dict[str, np.ndarray] = {}
    counter = {"value": 0}

    def visit(path: str, node: Any) -> Any:
        if node is None or isinstance(node, (str, bytes, bytearray, int, float, bool)):
            return node
        if isinstance(node, np.ndarray):
            tensor_id = f"{prefix}.{counter['value']:06d}"
            counter["value"] += 1
            arrays[tensor_id] = np.ascontiguousarray(node)
            return {
                "__tq_array__": True,
                "tensor_id": tensor_id,
                "dtype": str(node.dtype),
                "shape": list(node.shape),
            }
        if isinstance(node, tuple):
            return {"__tq_tuple__": True, "items": [visit(f"{path}[{i}]", item) for i, item in enumerate(node)]}
        if isinstance(node, list):
            return [visit(f"{path}[{i}]", item) for i, item in enumerate(node)]
        if isinstance(node, dict):
            return {key: visit(f"{path}.{key}", item) for key, item in node.items()}
        if hasattr(node, "shape") or hasattr(node, "__array__"):
            return visit(path, np.asarray(node))
        raise PromptCacheSerializationError(
            f"Unsupported prompt-cache state node {type(node).__name__} at {path}"
        )

    return visit(prefix, value), arrays


def _rehydrate(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if value.get("__tq_array__"):
            tensor = arrays.get(value["tensor_id"])
            if tensor is None:
                raise PromptCacheSerializationError(
                    f"Missing safetensors entry {value['tensor_id']!r} during prompt-cache restore."
                )
            expected_shape = tuple(value.get("shape", tensor.shape))
            if tuple(tensor.shape) != expected_shape:
                tensor = tensor.reshape(expected_shape)
            return tensor
        if value.get("__tq_tuple__"):
            return tuple(_rehydrate(item, arrays) for item in value["items"])
        return {key: _rehydrate(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_rehydrate(item, arrays) for item in value]
    return value


def _write_safetensors_payload(destination: Path, payload: dict[str, Any]) -> Path:
    from safetensors.numpy import save as safetensors_save

    arrays: dict[str, np.ndarray] = {}
    structural_entries: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(payload["entries"]):
        structured, entry_arrays = _flatten_arrays(f"entry_{entry_index:04d}", entry)
        arrays.update(entry_arrays)
        structural_entries.append(structured)

    header = {
        "schema_version": _PROMPT_CACHE_SCHEMA_VERSION,
        "entries": structural_entries,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    archive_bytes = safetensors_save(arrays) if arrays else b""

    with destination.open("wb") as handle:
        handle.write(_SAFETENSORS_MAGIC)
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(archive_bytes)
    return destination


def _read_safetensors_payload(source: Path) -> dict[str, Any]:
    from safetensors.numpy import load as safetensors_load

    with source.open("rb") as handle:
        magic = handle.read(4)
        if magic != _SAFETENSORS_MAGIC:
            raise PromptCacheSerializationError(
                "File is not a TurboMLX v3 safetensors prompt-cache."
            )
        header_size = int.from_bytes(handle.read(8), "little")
        header_bytes = handle.read(header_size)
        archive_bytes = handle.read()

    header = json.loads(header_bytes.decode("utf-8"))
    arrays = safetensors_load(archive_bytes) if archive_bytes else {}
    restored_entries = [_rehydrate(entry, arrays) for entry in header.get("entries", [])]
    return {"schema_version": int(header.get("schema_version", _PROMPT_CACHE_SCHEMA_VERSION)),
            "entries": restored_entries}


def _looks_like_safetensors_cache(handle: io.BufferedReader) -> bool:
    position = handle.tell()
    try:
        return handle.read(4) == _SAFETENSORS_MAGIC
    finally:
        handle.seek(position)


def save_prompt_cache(
    path: str | Path,
    prompt_cache: list[Any],
    *,
    format: str | None = None,
) -> Path:
    """Persist a prompt cache to disk.

    ``format`` may be ``"v3"`` (safetensors, default when available),
    ``"v2"`` (pickle with stable cache_type_id), or ``"auto"`` to let the
    helper pick the safest format the environment supports.
    """
    payload = {
        "schema_version": _PROMPT_CACHE_SCHEMA_VERSION,
        "entries": [_serialize_prompt_cache_entry(cache) for cache in prompt_cache],
    }
    destination = Path(path)
    selected = (format or "auto").lower()

    if selected == "auto":
        selected = "v3" if _safetensors_available() else "v2"

    if selected == "v3":
        if not _safetensors_available():
            raise PromptCacheSerializationError(
                "v3 prompt-cache format requires the optional `safetensors` dependency. "
                "Install it via `pip install turbomlx[serialize]`."
            )
        _LOGGER.info("Writing prompt-cache to %s using safetensors v3 format", destination)
        return _write_safetensors_payload(destination, payload)
    if selected == "v2":
        legacy_payload = dict(payload)
        legacy_payload["schema_version"] = _DEFAULT_LEGACY_SCHEMA
        _LOGGER.info("Writing prompt-cache to %s using legacy pickle v2 format", destination)
        with destination.open("wb") as handle:
            pickle.dump(legacy_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination
    raise PromptCacheSerializationError(f"Unsupported prompt-cache format requested: {format!r}")


def _resolve_prompt_cache_class(entry: dict[str, Any]) -> type:
    cache_type_id = entry.get("cache_type_id")
    if cache_type_id is not None:
        try:
            return _load_cache_type(cache_type_id)
        except PromptCacheSerializationError:
            pass

    class_path = entry.get("class_path")
    if not class_path:
        raise PromptCacheSerializationError(
            "Prompt-cache entry must include `cache_type_id` or `class_path` metadata."
        )
    warnings.warn(_CLASS_PATH_FALLBACK_WARNING, DeprecationWarning, stacklevel=3)
    return _load_class(class_path)


def _restore_prompt_cache_entry(entry: dict[str, Any]) -> Any:
    cls = _resolve_prompt_cache_class(entry)
    if hasattr(cls, "from_prompt_cache_entry"):
        return cls.from_prompt_cache_entry(entry)
    if hasattr(cls, "from_state"):
        cache = cls.from_state(entry["state"], entry["meta_state"])
    else:
        raise PromptCacheSerializationError(
            f"Prompt-cache class {entry.get('class_path', cls)} does not implement `from_state` "
            "or `from_prompt_cache_entry`."
        )

    extra_state = entry.get("extra_state") or {}
    if extra_state and hasattr(cache, "restore_prompt_cache_extra_state"):
        cache.restore_prompt_cache_extra_state(extra_state)
    elif extra_state:
        raise PromptCacheSerializationError(
            f"Prompt-cache class {entry.get('class_path', cls)} does not accept TurboMLX extra state."
        )
    return cache


def _validate_schema_version(version: int) -> None:
    if version not in _SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS:
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_PROMPT_CACHE_SCHEMA_VERSIONS))
        raise PromptCacheSerializationError(
            f"Unsupported TurboMLX prompt-cache schema_version={version}; "
            f"expected one of {{{supported}}}."
        )


def load_prompt_cache(path: str | Path) -> list[Any]:
    warnings.warn(_TRUSTED_LOCAL_ONLY_WARNING, UserWarning, stacklevel=2)

    source = Path(path)
    with source.open("rb") as handle:
        if _looks_like_safetensors_cache(handle):
            payload = _read_safetensors_payload(source)
        else:
            warnings.warn(_LEGACY_PICKLE_WARNING, UserWarning, stacklevel=2)
            payload = pickle.load(handle)

    version = int(payload.get("schema_version", 0))
    _validate_schema_version(version)

    return [_restore_prompt_cache_entry(entry) for entry in payload.get("entries", [])]
