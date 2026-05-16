"""Benchmark and cache memory accounting helpers."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any


def recursive_nbytes(value: Any, _seen_ids: set[int] | None = None) -> int:
    if value is None:
        return 0

    if _seen_ids is None:
        _seen_ids = set()

    value_id = id(value)
    if value_id in _seen_ids:
        return 0
    _seen_ids.add(value_id)

    try:
        nbytes = value.nbytes
    except Exception:
        nbytes = None
    if nbytes is not None:
        return int(nbytes)

    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(recursive_nbytes(v, _seen_ids) for v in value)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            recursive_nbytes(key, _seen_ids) + recursive_nbytes(item, _seen_ids)
            for key, item in value.items()
        )
    if hasattr(value, "__dict__"):
        return sys.getsizeof(value) + recursive_nbytes(vars(value), _seen_ids)

    slots = getattr(type(value), "__slots__", ())
    if slots:
        total = sys.getsizeof(value)
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if hasattr(value, slot):
                total += recursive_nbytes(getattr(value, slot), _seen_ids)
        return total

    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


@dataclass(slots=True)
class MemoryMetrics:
    allocated_cache_bytes: int
    used_state_bytes: int
    key_path_bytes: int
    value_path_bytes: int
    total_kv_bytes: int
    native_working_set_bytes: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
