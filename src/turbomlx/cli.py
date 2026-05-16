"""TurboMLX CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from turbomlx.eval.jsonl_eval import evaluate_jsonl_backend
from turbomlx.eval.logit import logit_cosine_similarity
from turbomlx.eval.needle import run_needle_backend_eval
from turbomlx.eval.perplexity import perplexity_from_text_file
from turbomlx.exceptions import MissingDependencyError, UnsupportedRuntimeVersionError
from turbomlx.mlx_runtime.availability import ensure_mlx_runtime
from turbomlx.mlx_runtime.benchmarking import run_benchmark_series
from turbomlx.mlx_runtime.config import (
    ScorerMode,
    TurboQuantConfig,
    TurboQuantMode,
    ValuesMode,
)
from turbomlx.mlx_runtime.generation import generate_with_backend
from turbomlx.mlx_runtime.sampling import make_sampler

app = typer.Typer(help="TurboMLX companion CLI for mlx-lm.")


def _require_runtime():
    try:
        return ensure_mlx_runtime()
    except (MissingDependencyError, UnsupportedRuntimeVersionError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def _load_mlx_lm_loader():
    return __import__("mlx_lm", fromlist=["load"]).load


def _make_config(
    bits_total: int,
    mode: TurboQuantMode | None,
    values_mode: ValuesMode,
    value_bits: int,
    quantized_kv_start: int,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
) -> TurboQuantConfig:
    return TurboQuantConfig(
        bits_total=bits_total,
        mode=mode,
        values_mode=values_mode,
        value_bits=value_bits,
        quantized_kv_start=quantized_kv_start,
        scorer_mode=scorer_mode,
    )


@app.command()
def generate(
    model_id: str,
    prompt: str,
    max_tokens: int = 16,
    backend: str = "turbomlx",
    bits_total: int = 4,
    mode: TurboQuantMode | None = None,
    values_mode: ValuesMode = ValuesMode.DENSE,
    value_bits: int = 4,
    quantized_kv_start: int = 0,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    seed: int | None = None,
):
    """Generate text with baseline, mlx_quant, or TurboMLX backend.

    Defaults to greedy argmax decoding (``temperature=0``). Pass any of
    ``--temperature``, ``--top-k``, ``--top-p``, or ``--min-p`` to enable
    categorical sampling. ``--seed`` controls the sampler RNG.
    """
    _mx, _base, _cache = _require_runtime()
    loader = _load_mlx_lm_loader()
    model, tokenizer = loader(model_id)
    tokens = tokenizer.encode(prompt, return_tensors="mlx")[0]
    config = _make_config(
        bits_total,
        mode,
        values_mode,
        value_bits,
        quantized_kv_start,
        scorer_mode=scorer_mode,
    )
    sampler = make_sampler(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        seed=seed,
    )
    generated, _logprobs, stats = generate_with_backend(
        model,
        tokens,
        max_tokens=max_tokens,
        backend=backend,
        config=config,
        sampler=sampler,
    )
    typer.echo(tokenizer.decode(generated.tolist()))
    typer.echo(json.dumps(stats.as_dict(), indent=2))


@app.command()
def benchmark(
    model_id: str,
    prompt_tokens: int = 128,
    generation_tokens: int = 8,
    backend: str = "turbomlx",
    bits_total: int = 4,
    mode: TurboQuantMode | None = None,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
    warmup_runs: int = 1,
    repeats: int = 3,
):
    """Run a warmup/repeat synthetic benchmark and report scorer route metadata."""
    mx, _base, _cache = _require_runtime()
    loader = _load_mlx_lm_loader()
    model, _tokenizer, config = loader(model_id, return_config=True)
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = mx.random.randint(0, vocab_size, (prompt_tokens,), dtype=mx.int32)
    tq_config = _make_config(
        bits_total,
        mode,
        ValuesMode.DENSE,
        4,
        0,
        scorer_mode=scorer_mode,
    )
    results = run_benchmark_series(
        model,
        prompt,
        backend=backend,
        config=tq_config,
        generation_tokens=generation_tokens,
        warmup_runs=warmup_runs,
        repeats=repeats,
    )
    typer.echo(json.dumps(results, indent=2))


@app.command("eval-logit")
def eval_logit(reference_file: Path, candidate_file: Path):
    """Compute cosine similarity between two saved logit arrays."""
    reference = np.load(reference_file)
    candidate = np.load(candidate_file)
    typer.echo(
        json.dumps(
            {"logit_cosine_similarity": logit_cosine_similarity(reference, candidate)},
            indent=2,
        )
    )


@app.command("eval-ppl")
def eval_ppl(
    model_id: str,
    text_file: Path,
    backend: str = "turbomlx",
    bits_total: int = 4,
    mode: TurboQuantMode | None = None,
    values_mode: ValuesMode = ValuesMode.DENSE,
    value_bits: int = 4,
    quantized_kv_start: int = 0,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
):
    """Compute perplexity from backend logprobs on a local text file."""
    config = _make_config(
        bits_total,
        mode,
        values_mode,
        value_bits,
        quantized_kv_start,
        scorer_mode=scorer_mode,
    )
    typer.echo(json.dumps(perplexity_from_text_file(model_id, text_file, backend=backend, config=config), indent=2))


@app.command("eval-needle")
def eval_needle(
    model_id: str,
    context_file: Path,
    needle: str,
    query: str,
    backend: str = "turbomlx",
    bits_total: int = 4,
    mode: TurboQuantMode | None = None,
    values_mode: ValuesMode = ValuesMode.DENSE,
    value_bits: int = 4,
    quantized_kv_start: int = 0,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
    max_tokens: int = 64,
    insertion_depth_pct: int = 50,
    seed: int = 0,
):
    """Run a backend-backed needle-in-a-haystack preview eval."""
    context = context_file.read_text()
    config = _make_config(
        bits_total,
        mode,
        values_mode,
        value_bits,
        quantized_kv_start,
        scorer_mode=scorer_mode,
    )
    typer.echo(
        json.dumps(
            run_needle_backend_eval(
                model_id,
                context,
                needle,
                query,
                backend=backend,
                config=config,
                max_tokens=max_tokens,
                insertion_depth_pct=insertion_depth_pct,
                seed=seed,
            ),
            indent=2,
        )
    )


@app.command("eval-jsonl")
def eval_jsonl(
    model_id: str,
    dataset_file: Path,
    backend: str = "turbomlx",
    bits_total: int = 4,
    mode: TurboQuantMode | None = None,
    values_mode: ValuesMode = ValuesMode.DENSE,
    value_bits: int = 4,
    quantized_kv_start: int = 0,
    scorer_mode: ScorerMode = ScorerMode.ORACLE_PREVIEW,
    max_tokens: int = 64,
    limit: int | None = None,
):
    """Run a backend-backed local JSONL eval with exact/substring scoring."""
    config = _make_config(
        bits_total,
        mode,
        values_mode,
        value_bits,
        quantized_kv_start,
        scorer_mode=scorer_mode,
    )
    typer.echo(
        json.dumps(
            evaluate_jsonl_backend(
                model_id,
                dataset_file,
                backend=backend,
                config=config,
                max_tokens=max_tokens,
                limit=limit,
            ),
            indent=2,
        )
    )
