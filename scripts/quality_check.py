#!/usr/bin/env python3
"""Compare last-token logits across backends to quantify TurboQuant fidelity.

Uses the same prompt across baseline / mlx_quant / TurboMLX backends and
reports:

* cosine similarity vs. baseline
* top-1 / top-5 / top-10 token agreement
* mean absolute difference between log-probabilities

This is the smallest faithful check that the TurboQuant key-path
quantization does not corrupt downstream sampling distributions on the
configured Qwen3.5 model.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from turbomlx.eval.logit import logit_cosine_similarity
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig, TurboQuantMode, ValuesMode
from turbomlx.mlx_runtime.generation import (
    _cache_metrics,
    convert_prompt_cache,
)
from turbomlx.mlx_runtime.patching import patch_attention_dispatch


def _last_token_logits(model, tokens, *, backend: str, config: TurboQuantConfig) -> np.ndarray:
    """Replay the prompt under ``backend`` and return the trailing logits as numpy."""
    _mx, _base, cache_mod = ensure_mlx_runtime()
    patch_attention_dispatch()
    prompt_cache = cache_mod.make_prompt_cache(model)
    prompt_size = int(tokens.size)
    processed = 0
    prefill_step_size = 2048
    while prompt_size - processed > 1:
        remaining = (prompt_size - processed) - 1
        n_to_process = min(prefill_step_size, remaining)
        model(tokens[processed: processed + n_to_process][None], cache=prompt_cache)
        convert_prompt_cache(prompt_cache, config, backend=backend)
        mx.eval([cache.state for cache in prompt_cache])
        processed += n_to_process
    logits = model(tokens[processed:][None], cache=prompt_cache)[:, -1, :]
    convert_prompt_cache(prompt_cache, config, backend=backend)
    metrics = _cache_metrics(prompt_cache)
    # MLX tensors with bfloat16 dtype cannot cross the PEP 3118 buffer
    # boundary directly; promote to float32 before handing them to numpy.
    return np.asarray(logits.squeeze(0).astype(mx.float32), dtype=np.float32), metrics


def _topk_agreement(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    ref_top = set(np.argsort(reference)[-k:].tolist())
    cand_top = set(np.argsort(candidate)[-k:].tolist())
    if not ref_top:
        return 0.0
    return len(ref_top & cand_top) / float(k)


def _format_row(label: str, metric: dict[str, float]) -> str:
    return (
        f"  {label:<30s} cos={metric['cosine']:.6f} | "
        f"top1={metric['top1']:.3f} | top5={metric['top5']:.3f} | top10={metric['top10']:.3f} | "
        f"mean|Δlogprob|={metric['log_l1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TURBOMLX_SMOKE_QWEN_MODEL",
            os.path.expanduser("~/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit"),
        ),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "TurboMLX v0.2 fidelity probe. Answer the following question as concisely as possible. "
            "Question: List three Apple Silicon GPU benefits for on-device language model inference."
        ),
    )
    args = parser.parse_args()

    print(f"[setup] loading {args.model}")
    model, tokenizer = load(args.model)
    tokens = tokenizer.encode(args.prompt, return_tensors="mlx")[0]
    prompt_size = int(tokens.size)
    print(f"[setup] prompt = {prompt_size} tokens")

    print("\n[run]  baseline (mlx-lm)")
    base_config = TurboQuantConfig(bits_total=4)
    baseline_logits, _ = _last_token_logits(model, tokens, backend="baseline", config=base_config)

    targets = {
        "mlx_quant (4-bit affine)": ("mlx_quant", base_config),
        "turbomlx + oracle_preview": (
            "turbomlx",
            TurboQuantConfig(
                bits_total=4,
                mode=TurboQuantMode.MSE,
                values_mode=ValuesMode.DENSE,
                scorer_mode=ScorerMode.ORACLE_PREVIEW,
            ),
        ),
        "turbomlx + native_mlx": (
            "turbomlx",
            TurboQuantConfig(
                bits_total=4,
                mode=TurboQuantMode.MSE,
                values_mode=ValuesMode.DENSE,
                scorer_mode=ScorerMode.NATIVE_MLX,
            ),
        ),
    }

    baseline_logprobs = baseline_logits - np.log(np.sum(np.exp(baseline_logits)))
    rows: dict[str, dict[str, float]] = {}
    for label, (backend, config) in targets.items():
        print(f"[run]  {label}")
        logits, _metrics = _last_token_logits(model, tokens, backend=backend, config=config)
        logprobs = logits - np.log(np.sum(np.exp(logits)))
        rows[label] = {
            "cosine": logit_cosine_similarity(baseline_logits, logits),
            "top1": _topk_agreement(baseline_logits, logits, 1),
            "top5": _topk_agreement(baseline_logits, logits, 5),
            "top10": _topk_agreement(baseline_logits, logits, 10),
            "log_l1": float(np.mean(np.abs(baseline_logprobs - logprobs))),
        }

    print("\n" + "=" * 100)
    print(f"Fidelity vs baseline @ prompt={prompt_size} tokens")
    print("=" * 100)
    for label, metric in rows.items():
        print(_format_row(label, metric))

    print("\nLegend:")
    print("  cos           cosine similarity of last-token logits (1.0 = identical direction)")
    print("  topK          fraction of the top-K predicted tokens that agree with baseline")
    print("  mean|Δlogprob| average absolute difference in normalized log-probabilities per token")

    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quality_check.json"
    payload = {
        "prompt": args.prompt,
        "prompt_tokens": prompt_size,
        "results": rows,
    }
    import json
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[setup] wrote fidelity payload to {output_path}")


if __name__ == "__main__":
    main()
