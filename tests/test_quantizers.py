import numpy as np
import pytest

from turbomlx.core_ref.quantizers import MSEPayload, ProdPayload, TurboQuantMSERef, TurboQuantProdRef
from turbomlx.core_ref.rotation import apply_rotation


def _random_unit_vectors(num_vectors: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((num_vectors, dim), dtype=np.float32)
    return data / np.linalg.norm(data, axis=-1, keepdims=True)


def test_mse_quantizer_reconstruction_improves_with_more_bits():
    vectors = _random_unit_vectors(64, 16)
    mse_2 = TurboQuantMSERef(16, 2)
    mse_3 = TurboQuantMSERef(16, 3)
    recon_2 = mse_2.dequantize(mse_2.quantize(vectors))
    recon_3 = mse_3.dequantize(mse_3.quantize(vectors))
    err_2 = np.mean(np.sum((vectors - recon_2) ** 2, axis=-1))
    err_3 = np.mean(np.sum((vectors - recon_3) ** 2, axis=-1))
    assert err_3 <= err_2


def test_prod_quantizer_bias_is_small_on_average():
    x = _random_unit_vectors(256, 16, seed=1)
    y = _random_unit_vectors(256, 16, seed=2)
    quantizer = TurboQuantProdRef(16, 3)
    payload = quantizer.quantize(x)
    reconstructed = quantizer.dequantize(payload)
    exact = np.sum(x * y, axis=-1)
    approx = np.sum(reconstructed * y, axis=-1)
    assert abs(float(np.mean(approx - exact))) < 0.05


def test_score_matrix_shape():
    keys = _random_unit_vectors(32, 8, seed=3).reshape(1, 2, 16, 8)
    queries = _random_unit_vectors(8, 8, seed=4).reshape(1, 2, 4, 8)
    quantizer = TurboQuantMSERef(8, 2)
    payload = quantizer.quantize(keys)
    scores = quantizer.score_matrix(queries, payload)
    assert scores.shape == (1, 2, 4, 16)


def test_mse_payload_metadata_mismatch_raises():
    quantizer = TurboQuantMSERef(8, 2)
    payload = quantizer.quantize(_random_unit_vectors(4, 8, seed=5))
    bad_payload = MSEPayload(
        packed_indices=payload.packed_indices,
        norms=payload.norms,
        head_dim=16,
        bits_total=payload.bits_total,
        codebook_id=payload.codebook_id,
    )
    with pytest.raises(ValueError):
        quantizer.dequantize(bad_payload)


def test_prod_payload_codebook_mismatch_raises():
    quantizer = TurboQuantProdRef(8, 3)
    payload = quantizer.quantize(_random_unit_vectors(4, 8, seed=6))
    bad_payload = ProdPayload(
        packed_indices=payload.packed_indices,
        packed_signs=payload.packed_signs,
        norms=payload.norms,
        residual_norms=payload.residual_norms,
        head_dim=payload.head_dim,
        bits_total=payload.bits_total,
        bits_mse=payload.bits_mse,
        codebook_id="wrong-codebook",
    )
    with pytest.raises(ValueError):
        quantizer.score_matrix(_random_unit_vectors(8, 8, seed=7).reshape(1, 2, 4, 8), bad_payload)


def test_searchsorted_quantization_matches_argmin_reference():
    """Searchsorted-based assignment must reproduce the legacy argmin output."""
    vectors = _random_unit_vectors(128, 16, seed=42)
    quantizer = TurboQuantMSERef(16, 3)
    fast_indices = quantizer._quantize_unit(
        vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)
    )
    rotated = apply_rotation(
        vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8),
        quantizer.rotation,
    )
    legacy_indices = np.argmin(
        np.abs(rotated[..., None] - quantizer.centroids[None, :]), axis=-1
    ).astype(np.uint8)
    assert np.array_equal(fast_indices, legacy_indices)


def test_quantize_memory_path_avoids_centroid_broadcast_blowup():
    """Quantizing a wide tensor must not allocate the K-fold broadcast tensor."""
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((1, 4, 64, 32), dtype=np.float32)
    quantizer = TurboQuantMSERef(32, 4)
    payload = quantizer.quantize(keys)
    reconstructed = quantizer.dequantize(payload)
    assert reconstructed.shape == keys.shape
