from __future__ import annotations

from types import SimpleNamespace

import pytest

from turbomlx.mlx_runtime import patching


def _configure_fake_patch_runtime(monkeypatch):
    original = object()
    patched = object()
    base_module = SimpleNamespace(scaled_dot_product_attention=original)
    qwen_module = SimpleNamespace(scaled_dot_product_attention=original)
    modules = {
        "mlx_lm.models.qwen3_next": qwen_module,
    }

    monkeypatch.setattr(patching, "_PATCHED_FUNCTION", None)
    monkeypatch.setattr(patching, "_PATCHED_MODULES", set())
    monkeypatch.setattr(patching, "_ORIGINAL_ATTENTION_FUNCTIONS", {})
    monkeypatch.setattr(patching, "ensure_mlx_runtime", lambda: (object(), base_module, object()))
    monkeypatch.setattr(patching, "dispatch_attention", lambda previous: patched)
    monkeypatch.setattr(patching, "import_module", lambda name: modules[name])
    return original, patched, base_module, qwen_module


def test_patch_attention_dispatch_returns_structured_report(monkeypatch):
    _original, patched, base_module, qwen_module = _configure_fake_patch_runtime(monkeypatch)

    report = patching.patch_attention_dispatch()

    assert isinstance(report, patching.PatchReport)
    assert report.patched_modules == ("mlx_lm.models.base", "mlx_lm.models.qwen3_next")
    assert report.already_patched_modules == ()
    assert report.missing_modules == ()
    assert base_module.scaled_dot_product_attention is patched
    assert qwen_module.scaled_dot_product_attention is patched


def test_patch_attention_dispatch_marks_modules_as_already_patched_on_reentry(monkeypatch):
    _configure_fake_patch_runtime(monkeypatch)

    first = patching.patch_attention_dispatch()
    second = patching.patch_attention_dispatch()

    assert first.patched_modules == ("mlx_lm.models.base", "mlx_lm.models.qwen3_next")
    assert second.patched_modules == ()
    assert second.already_patched_modules == ("mlx_lm.models.base", "mlx_lm.models.qwen3_next")


def test_unpatch_attention_dispatch_restores_original_symbols(monkeypatch):
    original, _patched, base_module, qwen_module = _configure_fake_patch_runtime(monkeypatch)

    patching.patch_attention_dispatch()
    restored = patching.unpatch_attention_dispatch()

    assert restored == ("mlx_lm.models.base", "mlx_lm.models.qwen3_next")
    assert base_module.scaled_dot_product_attention is original
    assert qwen_module.scaled_dot_product_attention is original


def test_patched_attention_dispatch_context_manager_restores_originals(monkeypatch):
    original, patched, base_module, qwen_module = _configure_fake_patch_runtime(monkeypatch)

    with patching.patched_attention_dispatch() as report:
        assert report.patched_modules == ("mlx_lm.models.base", "mlx_lm.models.qwen3_next")
        assert base_module.scaled_dot_product_attention is patched
        assert qwen_module.scaled_dot_product_attention is patched

    assert base_module.scaled_dot_product_attention is original
    assert qwen_module.scaled_dot_product_attention is original


def test_dispatch_attention_forwards_sinks_to_baseline_for_non_turbomlx_caches(monkeypatch):
    """Patched dispatcher must forward attention_sinks to the upstream baseline."""
    from turbomlx.mlx_runtime import attention

    monkeypatch.setattr(attention, "_MX_RUNTIME_READY", True)
    monkeypatch.setattr(attention, "mx", object())
    monkeypatch.setattr(attention, "TurboQuantKVCache", type("_NotATurboQuantCache", (), {}))

    forwarded: dict[str, object] = {}

    def upstream(queries, keys, values, cache, *, scale, mask, sinks=None):
        forwarded["sinks"] = sinks
        forwarded["scale"] = scale
        return "upstream-result"

    dispatcher = attention.dispatch_attention(upstream)
    result = dispatcher("q", "k", "v", cache=object(), scale=0.5, mask=None, sinks="sentinel")

    assert result == "upstream-result"
    assert forwarded["sinks"] == "sentinel"
    assert forwarded["scale"] == 0.5


def test_dispatch_attention_rejects_attention_sinks_for_turboquant_caches(monkeypatch):
    """TurboQuantKVCache cannot model attention sinks yet, so we must fail loudly."""
    from turbomlx.exceptions import UnsupportedConfigurationError
    from turbomlx.mlx_runtime import attention

    class _FakeTurboQuantKVCache:
        pass

    monkeypatch.setattr(attention, "_MX_RUNTIME_READY", True)
    monkeypatch.setattr(attention, "mx", object())
    monkeypatch.setattr(attention, "TurboQuantKVCache", _FakeTurboQuantKVCache)

    dispatcher = attention.dispatch_attention(lambda *args, **kwargs: None)
    with pytest.raises(UnsupportedConfigurationError, match="attention sinks"):
        dispatcher("q", "k", "v", cache=_FakeTurboQuantKVCache(), scale=0.5, mask=None, sinks=object())
