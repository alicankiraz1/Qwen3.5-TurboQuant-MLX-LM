"""Reference packing utilities for packed payload serialization.

The packed layout is little-endian and compact: index ``j`` occupies bits
``[j * bits, (j + 1) * bits)`` of the flattened bit stream. Bytes hold up
to eight bits each and entries may straddle byte boundaries.

The implementation uses two strategies:

* A vectorized fast path for ``bits in {1, 2, 4, 8}`` when ``last_dim`` is a
  multiple of ``8 // bits``. This is the path exercised by every TurboQuant
  preview configuration in the current release and runs entirely on
  contiguous NumPy buffers without Python-level inner loops.
* A general fallback that loops only over ``last_dim`` (typically the head
  dimension, 64-128) while keeping every operation across the leading
  batch / head / token axes vectorized.

The output is bit-for-bit identical to the previous Python-level
implementation; only the runtime cost moves from ``O(rows * last_dim)``
Python iterations to ``O(last_dim)`` NumPy operations.
"""

from __future__ import annotations

from math import ceil

import numpy as np


def packed_nbytes(last_dim: int, bits: int) -> int:
    return ceil(last_dim * bits / 8)


def _validate_bits(bits: int) -> None:
    if bits < 1 or bits > 8:
        raise ValueError("bits must be in [1, 8]")


def _packs_evenly(last_dim: int, bits: int) -> bool:
    return 8 % bits == 0 and last_dim % (8 // bits) == 0


def _pack_bits_fast(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack ``values`` when ``bits in {1, 2, 4, 8}`` divides 8 cleanly."""
    per_byte = 8 // bits
    last_dim = values.shape[-1]
    mask = np.uint16((1 << bits) - 1)
    chunks = (values & mask).reshape(*values.shape[:-1], last_dim // per_byte, per_byte)
    shifts = (np.arange(per_byte, dtype=np.uint16) * bits).reshape(*((1,) * (chunks.ndim - 1)), per_byte)
    packed = (chunks.astype(np.uint16) << shifts).sum(axis=-1, dtype=np.uint16)
    return packed.astype(np.uint8)


def _pack_bits_general(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack arbitrary ``bits`` widths with a vectorized scatter loop."""
    last_dim = values.shape[-1]
    packed_dim = packed_nbytes(last_dim, bits)
    mask = np.uint32((1 << bits) - 1)
    masked = (values.astype(np.uint32) & mask)
    out = np.zeros(values.shape[:-1] + (packed_dim,), dtype=np.uint32)
    bit_positions = np.arange(last_dim, dtype=np.int64) * bits
    byte_indices = (bit_positions // 8).astype(np.int64)
    bit_offsets = (bit_positions % 8).astype(np.uint32)

    for j in range(last_dim):
        shifted = masked[..., j] << bit_offsets[j]
        out[..., byte_indices[j]] |= shifted & 0xFF
        spill = int(bit_offsets[j]) + bits - 8
        if spill > 0 and byte_indices[j] + 1 < packed_dim:
            out[..., byte_indices[j] + 1] |= (shifted >> 8) & 0xFF
    return out.astype(np.uint8)


def pack_bits(indices: np.ndarray, bits: int) -> np.ndarray:
    _validate_bits(bits)
    values = np.asarray(indices, dtype=np.uint16)
    last_dim = values.shape[-1]
    if last_dim == 0:
        return np.zeros(values.shape[:-1] + (0,), dtype=np.uint8)
    if _packs_evenly(last_dim, bits):
        return _pack_bits_fast(values, bits)
    return _pack_bits_general(values, bits)


def _unpack_bits_fast(packed: np.ndarray, bits: int, original_dim: int) -> np.ndarray:
    """Unpack ``packed`` when ``bits in {1, 2, 4, 8}`` divides 8 cleanly."""
    per_byte = 8 // bits
    mask = np.uint8((1 << bits) - 1)
    shifts = (np.arange(per_byte, dtype=np.uint8) * bits)
    expanded = (packed[..., :, None] >> shifts) & mask
    return expanded.reshape(*packed.shape[:-1], original_dim).astype(np.uint8)


def _unpack_bits_general(packed: np.ndarray, bits: int, original_dim: int) -> np.ndarray:
    """Unpack arbitrary ``bits`` widths with a vectorized gather loop."""
    packed_dim = packed.shape[-1]
    out = np.zeros(packed.shape[:-1] + (original_dim,), dtype=np.uint8)
    mask = np.uint16((1 << bits) - 1)
    bit_positions = np.arange(original_dim, dtype=np.int64) * bits
    byte_indices = (bit_positions // 8).astype(np.int64)
    bit_offsets = (bit_positions % 8).astype(np.uint16)

    for j in range(original_dim):
        if byte_indices[j] >= packed_dim:
            break
        primary = packed[..., byte_indices[j]].astype(np.uint16) >> bit_offsets[j]
        spill = int(bit_offsets[j]) + bits - 8
        if spill > 0 and byte_indices[j] + 1 < packed_dim:
            primary |= packed[..., byte_indices[j] + 1].astype(np.uint16) << (8 - bit_offsets[j])
        out[..., j] = (primary & mask).astype(np.uint8)
    return out


def unpack_bits(packed: np.ndarray, bits: int, original_dim: int) -> np.ndarray:
    _validate_bits(bits)
    values = np.asarray(packed, dtype=np.uint8)
    if original_dim <= 0:
        return np.zeros(values.shape[:-1] + (0,), dtype=np.uint8)
    if _packs_evenly(original_dim, bits):
        return _unpack_bits_fast(values, bits, original_dim)
    return _unpack_bits_general(values, bits, original_dim)


def pack_sign_bits(signs: np.ndarray) -> np.ndarray:
    sign_bits = (np.asarray(signs) > 0).astype(np.uint8)
    return np.packbits(sign_bits, axis=-1, bitorder="little")


def unpack_sign_bits(packed: np.ndarray, original_dim: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, bitorder="little")
    bits = bits[..., :original_dim]
    return (bits.astype(np.int8) * 2) - 1
