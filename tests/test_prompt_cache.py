from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pytest

from turbomlx import prompt_cache as prompt_cache_module
from turbomlx.exceptions import PromptCacheSerializationError
from turbomlx.prompt_cache import (
    _SAFETENSORS_MAGIC,
    _safetensors_available,
    load_prompt_cache,
    register_cache_type,
    save_prompt_cache,
)


class FakeSerializableCache:
    PROMPT_CACHE_TYPE_ID = "tests.fake.serializable.v1"

    def __init__(self, state, meta_state, *, extra_state=None):
        self._state = state
        self._meta_state = meta_state
        self._extra_state = extra_state or {}

    @property
    def state(self):
        return self._state

    @property
    def meta_state(self):
        return self._meta_state

    def prompt_cache_extra_state(self):
        return self._extra_state

    def restore_prompt_cache_extra_state(self, payload):
        self._extra_state = payload

    @classmethod
    def from_state(cls, state, meta_state):
        return cls(state, meta_state)


@pytest.fixture(autouse=True)
def _register_fake_cache_class():
    """Allow tests to load FakeSerializableCache without polluting the global allowlist forever."""
    register_cache_type(FakeSerializableCache.PROMPT_CACHE_TYPE_ID, f"{__name__}.FakeSerializableCache")
    yield
    prompt_cache_module._CACHE_TYPE_TO_CLASS_PATH.pop(FakeSerializableCache.PROMPT_CACHE_TYPE_ID, None)
    prompt_cache_module._TRUSTED_CLASS_PATHS.discard(f"{__name__}.FakeSerializableCache")


def _make_roundtrip_caches():
    dense_cache = FakeSerializableCache(
        state=(np.arange(6, dtype=np.float32).reshape(1, 1, 1, 6),),
        meta_state=(json.dumps({"kind": "dense"}),),
    )
    affine_cache = FakeSerializableCache(
        state=((np.arange(4, dtype=np.uint8).reshape(1, 1, 1, 4), np.ones((1, 1, 1, 4), dtype=np.float32)),),
        meta_state=(json.dumps({"kind": "affine"}),),
    )
    frozen_mixed_cache = FakeSerializableCache(
        state=(np.arange(4, dtype=np.uint8).reshape(1, 1, 1, 4),),
        meta_state=(json.dumps({"kind": "mixed-frozen"}),),
        extra_state={
            "calibration_tokens_seen": 256,
            "outlier_mask_frozen": True,
            "calibration_keys": None,
            "calibration_values": None,
        },
    )
    mid_calibration_cache = FakeSerializableCache(
        state=(np.zeros((0,), dtype=np.uint8),),
        meta_state=(json.dumps({"kind": "mixed-mid-calibration"}),),
        extra_state={
            "calibration_tokens_seen": 32,
            "outlier_mask_frozen": False,
            "calibration_keys": np.arange(24, dtype=np.float32).reshape(1, 1, 2, 12),
            "calibration_values": np.arange(24, dtype=np.float32).reshape(1, 1, 2, 12),
        },
    )
    return [dense_cache, affine_cache, frozen_mixed_cache, mid_calibration_cache]


def test_prompt_cache_v2_pickle_roundtrip_preserves_state_meta_and_extra(tmp_path):
    """The legacy v2 pickle format is still supported for backward compatibility."""
    path = tmp_path / "prompt-cache.tqcache"
    caches = _make_roundtrip_caches()

    save_prompt_cache(path, caches, format="v2")
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    assert payload["schema_version"] == 2
    assert all("cache_type_id" in entry for entry in payload["entries"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = load_prompt_cache(path)
    messages = [str(item.message) for item in caught]
    assert any("trusted-local-only" in message for message in messages)
    assert any("legacy pickle-backed" in message for message in messages)

    assert len(restored) == 4
    assert np.array_equal(restored[0].state[0], caches[0].state[0])
    assert restored[1].meta_state == caches[1].meta_state
    assert restored[2].prompt_cache_extra_state()["outlier_mask_frozen"] is True
    assert restored[3].prompt_cache_extra_state()["calibration_tokens_seen"] == 32


@pytest.mark.skipif(not _safetensors_available(), reason="safetensors optional dependency missing")
def test_prompt_cache_v3_safetensors_roundtrip_uses_typed_header_and_array_chunk(tmp_path):
    path = tmp_path / "prompt-cache-v3.tqcache"
    caches = _make_roundtrip_caches()

    save_prompt_cache(path, caches, format="v3")
    raw_bytes = path.read_bytes()
    assert raw_bytes.startswith(_SAFETENSORS_MAGIC)

    with pytest.warns(UserWarning, match="trusted-local-only"):
        restored = load_prompt_cache(path)

    assert len(restored) == 4
    assert np.array_equal(restored[0].state[0], caches[0].state[0])
    assert restored[1].meta_state == caches[1].meta_state
    affine_payload = restored[1].state[0]
    assert isinstance(affine_payload, tuple)
    assert np.array_equal(affine_payload[0], caches[1].state[0][0])
    assert np.array_equal(affine_payload[1], caches[1].state[0][1])
    assert restored[2].prompt_cache_extra_state()["outlier_mask_frozen"] is True
    assert np.array_equal(
        restored[3].prompt_cache_extra_state()["calibration_keys"],
        caches[3].prompt_cache_extra_state()["calibration_keys"],
    )


@pytest.mark.skipif(not _safetensors_available(), reason="safetensors optional dependency missing")
def test_prompt_cache_auto_format_prefers_v3_when_safetensors_is_installed(tmp_path):
    path = tmp_path / "prompt-cache-auto.tqcache"
    caches = _make_roundtrip_caches()
    save_prompt_cache(path, caches)
    assert path.read_bytes().startswith(_SAFETENSORS_MAGIC)


def test_load_prompt_cache_supports_v1_class_path_fallback_with_warning(tmp_path):
    path = tmp_path / "prompt-cache-v1.tqcache"
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "class_path": f"{__name__}.FakeSerializableCache",
                "state": (np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),),
                "meta_state": (json.dumps({"kind": "legacy"}),),
                "extra_state": {"flag": True},
            }
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = load_prompt_cache(path)

    messages = [str(item.message) for item in caught]
    assert any("trusted-local-only" in message for message in messages)
    assert any("deprecated class_path metadata" in message for message in messages)
    assert restored[0].meta_state == payload["entries"][0]["meta_state"]


def test_load_prompt_cache_prefers_cache_type_id_over_class_path(tmp_path):
    """When both cache_type_id and class_path are present, the typed identifier wins."""
    path = tmp_path / "prompt-cache-v2.tqcache"
    payload = {
        "schema_version": 2,
        "entries": [
            {
                "cache_type_id": FakeSerializableCache.PROMPT_CACHE_TYPE_ID,
                "class_path": "broken.module.DoesNotExist",
                "state": (np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),),
                "meta_state": (json.dumps({"kind": "typed"}),),
                "extra_state": {},
            }
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.warns(UserWarning, match="trusted-local-only"):
        restored = load_prompt_cache(path)

    assert isinstance(restored[0], FakeSerializableCache)
    assert restored[0].meta_state == payload["entries"][0]["meta_state"]


def test_load_prompt_cache_rejects_unknown_class_paths_even_with_warning(tmp_path):
    """Loading a v1 cache that references an unregistered class must be refused."""
    path = tmp_path / "prompt-cache-rogue.tqcache"
    payload = {
        "schema_version": 1,
        "entries": [
            {
                "class_path": "rogue.attacker.controlled.Class",
                "state": (np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),),
                "meta_state": (json.dumps({"kind": "rogue"}),),
                "extra_state": {},
            }
        ],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(PromptCacheSerializationError, match="non-allowlisted"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_prompt_cache(path)


def test_save_prompt_cache_rejects_unknown_format():
    with pytest.raises(PromptCacheSerializationError, match="format requested"):
        save_prompt_cache("/tmp/turbomlx-prompt-cache-test.tqcache", [], format="v9")
