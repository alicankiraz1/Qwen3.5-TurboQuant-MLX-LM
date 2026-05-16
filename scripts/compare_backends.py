#!/usr/bin/env python3
"""Compare baseline / mlx_quant / TurboMLX backends side by side.

Runs the same synthetic prompt against the requested backend / scorer
combinations using ``run_benchmark_series`` so warmup / repeat / median
semantics are identical to ``turbomlx benchmark``. Reports prompt TPS,
decode TPS, key-path bytes, total KV bytes, and the active scorer route
in a single comparison table.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import mlx.core as mx
from mlx_lm import load

from turbomlx.mlx_runtime.benchmarking import run_benchmark_series
from turbomlx.mlx_runtime.config import ScorerMode, TurboQuantConfig, TurboQuantMode, ValuesMode


@dataclass(slots=True)
class BackendSpec:
    label: str
    backend: str
    scorer_mode: ScorerMode | None = None
    bits_total: int = 4

    def make_config(self) -> TurboQuantConfig:
        scorer = self.scorer_mode or ScorerMode.ORACLE_PREVIEW
        return TurboQuantConfig(
            bits_total=self.bits_total,
            mode=TurboQuantMode.MSE,
            values_mode=ValuesMode.DENSE,
            scorer_mode=scorer,
        )


def _format_mib(value: float | int) -> str:
    return f"{value / (1024 * 1024):8.2f}"


def _format_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("backend", "Backend", 28),
        ("prompt_tps", "Prompt TPS", 11),
        ("generation_tps", "Decode TPS", 11),
        ("key_path_mib", "Key Path", 10),
        ("total_kv_mib", "Total KV", 10),
        ("native_ws_mib", "Native WS", 10),
        ("peak_mem_gb", "Peak Mem", 10),
        ("scorer_route", "Scorer Route", 18),
    ]
    header = " | ".join(f"{label:>{width}}" if key not in ("backend", "scorer_route") else f"{label:<{width}}"
                        for key, label, width in columns)
    separator = "-+-".join("-" * width for _, _, width in columns)
    lines = [header, separator]
    for row in rows:
        cells = []
        for key, _, width in columns:
            raw = row[key]
            if key in ("backend", "scorer_route"):
                cells.append(f"{str(raw):<{width}}")
            elif key in ("prompt_tps", "generation_tps", "peak_mem_gb"):
                cells.append(f"{raw:>{width}.2f}")
            else:
                cells.append(f"{raw:>{width}}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _serialize_median(payload: dict[str, Any], spec: BackendSpec) -> dict[str, Any]:
    summary = payload["median"]
    return {
        "backend": spec.label,
        "prompt_tps": float(summary["prompt_tps"]),
        "generation_tps": float(summary["generation_tps"]),
        "key_path_mib": _format_mib(summary["key_path_bytes"]),
        "total_kv_mib": _format_mib(summary["total_kv_bytes"]),
        "native_ws_mib": _format_mib(summary["native_working_set_bytes"]),
        "peak_mem_gb": float(summary["peak_memory_gb"]),
        "scorer_route": str(summary["scorer_route"]),
    }


def _run_one_backend(
    model,
    prompt_tokens,
    spec: BackendSpec,
    *,
    generation_tokens: int,
    warmup_runs: int,
    repeats: int,
) -> dict[str, Any]:
    print(f"\n>>> {spec.label}: warming up + measuring ({warmup_runs} warmup + {repeats} repeats)...")
    mx.reset_peak_memory()
    gc.collect()
    start = time.perf_counter()
    payload = run_benchmark_series(
        model,
        prompt_tokens,
        backend=spec.backend,
        config=spec.make_config(),
        generation_tokens=generation_tokens,
        warmup_runs=warmup_runs,
        repeats=repeats,
    )
    elapsed = time.perf_counter() - start
    payload["wall_time_s"] = elapsed
    print(
        f"<<< {spec.label}: done in {elapsed:5.1f}s "
        f"(median prompt_tps={payload['median']['prompt_tps']:.2f}, "
        f"decode_tps={payload['median']['generation_tps']:.2f}, "
        f"route={payload['median']['scorer_route']})"
    )
    return payload


def _build_specs(backends: list[str]) -> list[BackendSpec]:
    catalog: dict[str, BackendSpec] = {
        "baseline": BackendSpec(label="baseline (mlx-lm)", backend="baseline"),
        "mlx_quant": BackendSpec(label="mlx_quant (4-bit affine)", backend="mlx_quant"),
        "turbomlx-oracle": BackendSpec(
            label="turbomlx + oracle_preview", backend="turbomlx", scorer_mode=ScorerMode.ORACLE_PREVIEW
        ),
        "turbomlx-native": BackendSpec(
            label="turbomlx + native_mlx", backend="turbomlx", scorer_mode=ScorerMode.NATIVE_MLX
        ),
    }
    selected = []
    for name in backends:
        if name not in catalog:
            raise SystemExit(f"unknown backend {name!r}; pick from {sorted(catalog)}")
        selected.append(catalog[name])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TURBOMLX_SMOKE_QWEN_MODEL",
            os.path.expanduser("~/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit"),
        ),
        help="Path or HuggingFace id of the MLX model to load.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["baseline", "mlx_quant", "turbomlx-oracle", "turbomlx-native"],
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump the full payload for each backend.",
    )
    args = parser.parse_args()

    specs = _build_specs(args.backends)

    print(f"[setup] loading model: {args.model}")
    t0 = time.perf_counter()
    model, _tokenizer, config = load(args.model, return_config=True)
    vocab_size = config.get("vocab_size") or config.get("text_config", {}).get("vocab_size")
    if vocab_size is None:
        raise SystemExit("could not determine vocab_size from the loaded model config")
    print(f"[setup] model loaded in {time.perf_counter() - t0:.1f}s (vocab={vocab_size})")

    prompt = mx.random.randint(0, int(vocab_size), (int(args.prompt_tokens),), dtype=mx.int32)
    print(
        f"[setup] synthetic prompt: {args.prompt_tokens} tokens, decoding {args.generation_tokens}, "
        f"{args.warmup_runs} warmup + {args.repeats} repeats per backend"
    )

    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {
        "config": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
        },
        "backends": {},
    }

    for spec in specs:
        payload = _run_one_backend(
            model,
            prompt,
            spec,
            generation_tokens=args.generation_tokens,
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
        )
        rows.append(_serialize_median(payload, spec))
        raw["backends"][spec.label] = payload

    print("\n" + "=" * 100)
    print(f"Comparison @ prompt={args.prompt_tokens} / decode={args.generation_tokens} "
          f"(median over {args.repeats} repeats, {args.warmup_runs} warmup)")
    print("=" * 100)
    print(_format_table(rows))

    if "baseline (mlx-lm)" in {row["backend"] for row in rows}:
        baseline = next(row for row in rows if row["backend"] == "baseline (mlx-lm)")
        print("\nDeltas vs baseline (positive => TurboMLX wins):")
        for row in rows:
            if row["backend"] == baseline["backend"]:
                continue
            pt = (row["prompt_tps"] / baseline["prompt_tps"] - 1.0) * 100.0
            dt = (row["generation_tps"] / baseline["generation_tps"] - 1.0) * 100.0
            kv = (1.0 - float(row["total_kv_mib"]) / float(baseline["total_kv_mib"])) * 100.0
            print(
                f"  {row['backend']:<30s} prompt_tps={pt:+6.2f}% | decode_tps={dt:+6.2f}% | total_kv={kv:+6.2f}%"
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(raw, indent=2, default=str))
        print(f"\n[setup] wrote full payload to {args.json_out}")


if __name__ == "__main__":
    main()
