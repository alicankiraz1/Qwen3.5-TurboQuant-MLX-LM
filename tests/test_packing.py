import numpy as np
import pytest

from turbomlx.core_ref.packing import (
    _packs_evenly,
    pack_bits,
    pack_sign_bits,
    unpack_bits,
    unpack_sign_bits,
)


def test_pack_unpack_roundtrip_for_multiple_bitwidths():
    rng = np.random.default_rng(0)
    for bits in (1, 2, 3, 4):
        levels = 1 << bits
        indices = rng.integers(0, levels, size=(3, 5, 17), dtype=np.uint8)
        packed = pack_bits(indices, bits)
        unpacked = unpack_bits(packed, bits, 17)
        assert np.array_equal(indices, unpacked)


def test_pack_unpack_sign_bits_roundtrip():
    signs = np.array([[1, -1, 1, 1, -1, -1, 1, -1, 1]], dtype=np.int8)
    packed = pack_sign_bits(signs)
    unpacked = unpack_sign_bits(packed, signs.shape[-1])
    assert np.array_equal(signs, unpacked)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_pack_bits_fast_path_matches_general_path(bits):
    rng = np.random.default_rng(bits)
    levels = 1 << bits
    aligned_dim = 16 if bits != 8 else 8
    indices = rng.integers(0, levels, size=(2, 3, aligned_dim), dtype=np.uint8)

    assert _packs_evenly(aligned_dim, bits) is True
    fast_packed = pack_bits(indices, bits)
    fast_unpacked = unpack_bits(fast_packed, bits, aligned_dim)
    assert np.array_equal(indices, fast_unpacked)


@pytest.mark.parametrize("bits", [3, 5, 6, 7])
def test_pack_bits_general_path_handles_odd_widths(bits):
    rng = np.random.default_rng(100 + bits)
    levels = 1 << bits
    indices = rng.integers(0, levels, size=(2, 3, 19), dtype=np.uint8)
    packed = pack_bits(indices, bits)
    unpacked = unpack_bits(packed, bits, 19)
    assert np.array_equal(indices, unpacked)


def test_vectorized_pack_bits_matches_realistic_qwen_head_dim_shape():
    rng = np.random.default_rng(7)
    indices = rng.integers(0, 16, size=(1, 4, 256, 128), dtype=np.uint8)
    packed = pack_bits(indices, bits=4)
    assert packed.shape == (1, 4, 256, 64)
    unpacked = unpack_bits(packed, bits=4, original_dim=128)
    assert np.array_equal(indices, unpacked)


def test_pack_bits_handles_empty_last_dimension():
    indices = np.zeros((2, 0), dtype=np.uint8)
    packed = pack_bits(indices, bits=4)
    assert packed.shape == (2, 0)
    assert unpack_bits(packed, bits=4, original_dim=0).shape == (2, 0)


def test_pack_bits_rejects_out_of_range_widths():
    indices = np.zeros((4,), dtype=np.uint8)
    with pytest.raises(ValueError):
        pack_bits(indices, bits=0)
    with pytest.raises(ValueError):
        pack_bits(indices, bits=9)
    with pytest.raises(ValueError):
        unpack_bits(np.zeros((4,), dtype=np.uint8), bits=9, original_dim=4)
