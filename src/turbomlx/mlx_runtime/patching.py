"""Runtime patching helpers for TurboMLX."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module

from turbomlx.mlx_runtime.attention import dispatch_attention
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime
from turbomlx.mlx_runtime.config import TurboQuantConfig


@dataclass(frozen=True, slots=True)
class PatchReport:
    patched_modules: tuple[str, ...]
    already_patched_modules: tuple[str, ...]
    missing_modules: tuple[str, ...]


_PATCHED_FUNCTION = None
_PATCHED_MODULES: set[str] = set()
_ORIGINAL_ATTENTION_FUNCTIONS: dict[str, object] = {}
_SUPPORTED_PATCH_MODULES = (
    "mlx_lm.models.base",
    "mlx_lm.models.qwen3_next",
)


def supported_patch_modules() -> tuple[str, ...]:
    return _SUPPORTED_PATCH_MODULES


def _import_patch_modules():
    _mx, base, _cache = ensure_mlx_runtime()
    modules = {"mlx_lm.models.base": base}
    missing_modules: list[str] = []
    for module_name in _SUPPORTED_PATCH_MODULES[1:]:
        try:
            modules[module_name] = import_module(module_name)
        except ModuleNotFoundError:
            missing_modules.append(module_name)
    return modules, tuple(missing_modules)


def patch_attention_dispatch() -> PatchReport:
    global _PATCHED_FUNCTION
    modules, missing_modules = _import_patch_modules()
    base = modules["mlx_lm.models.base"]
    if _PATCHED_FUNCTION is None:
        original_base = _ORIGINAL_ATTENTION_FUNCTIONS.get(
            "mlx_lm.models.base",
            base.scaled_dot_product_attention,
        )
        _PATCHED_FUNCTION = dispatch_attention(original_base)

    patched_now: list[str] = []
    already_patched: list[str] = []
    for module_name, module in modules.items():
        current = getattr(module, "scaled_dot_product_attention", None)
        if current is _PATCHED_FUNCTION:
            _PATCHED_MODULES.add(module_name)
            already_patched.append(module_name)
            continue
        if current is None:
            continue
        _ORIGINAL_ATTENTION_FUNCTIONS.setdefault(module_name, current)
        module.scaled_dot_product_attention = _PATCHED_FUNCTION
        _PATCHED_MODULES.add(module_name)
        patched_now.append(module_name)

    return PatchReport(
        patched_modules=tuple(sorted(patched_now)),
        already_patched_modules=tuple(sorted(already_patched)),
        missing_modules=tuple(sorted(missing_modules)),
    )


def unpatch_attention_dispatch() -> tuple[str, ...]:
    global _PATCHED_FUNCTION
    if not _ORIGINAL_ATTENTION_FUNCTIONS:
        _PATCHED_FUNCTION = None
        _PATCHED_MODULES.clear()
        return ()

    modules, _missing_modules = _import_patch_modules()
    restored: list[str] = []
    for module_name, original in list(_ORIGINAL_ATTENTION_FUNCTIONS.items()):
        module = modules.get(module_name)
        if module is None:
            try:
                module = import_module(module_name)
            except ModuleNotFoundError:
                _ORIGINAL_ATTENTION_FUNCTIONS.pop(module_name, None)
                _PATCHED_MODULES.discard(module_name)
                continue
        if hasattr(module, "scaled_dot_product_attention"):
            module.scaled_dot_product_attention = original
            restored.append(module_name)
        _ORIGINAL_ATTENTION_FUNCTIONS.pop(module_name, None)
        _PATCHED_MODULES.discard(module_name)

    _PATCHED_FUNCTION = None
    return tuple(sorted(restored))


@contextmanager
def patched_attention_dispatch():
    report = patch_attention_dispatch()
    try:
        yield report
    finally:
        unpatch_attention_dispatch()


def patch_model_for_turbomlx_experimental(model, config: TurboQuantConfig):
    patch_attention_dispatch()
    model._turbomlx_config = config
    return model


def attach_turbomlx(model, config: TurboQuantConfig):
    warnings.warn(
        "`attach_turbomlx()` is deprecated for preview use. Prefer "
        "`generate_with_backend()`, `convert_prompt_cache()`, `save_prompt_cache()`, "
        "and `load_prompt_cache()`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return patch_model_for_turbomlx_experimental(model, config)
