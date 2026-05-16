from __future__ import annotations

import os

import pytest

from turbomlx.eval.runtime import encode_text, load_model_and_tokenizer
from turbomlx.mlx_runtime.availability import mlx_runtime_available
from turbomlx.mlx_runtime.config import TurboQuantConfig
from turbomlx.mlx_runtime.generation import generate_with_backend

pytestmark = pytest.mark.skipif(
    not mlx_runtime_available(),
    reason="MLX runtime is unavailable in this environment.",
)


@pytest.mark.skipif(
    not os.environ.get("TURBOMLX_SMOKE_QWEN_MODEL"),
    reason="Set TURBOMLX_SMOKE_QWEN_MODEL to run the Qwen preview smoke test.",
)
def test_qwen_preview_generate_with_backend_smoke():
    model_id = os.environ["TURBOMLX_SMOKE_QWEN_MODEL"]
    model, tokenizer = load_model_and_tokenizer(model_id)
    prompt = encode_text(tokenizer, "TurboMLX Qwen preview smoke test prompt.")
    generated, _logprobs, stats = generate_with_backend(
        model,
        prompt,
        max_tokens=2,
        backend="turbomlx",
        config=TurboQuantConfig(bits_total=4),
    )
    assert generated.size > 0
    assert stats.backend == "turbomlx"
