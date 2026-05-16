"""Repeated benchmark helpers for preview runtime comparisons."""

from __future__ import annotations

from statistics import median

from turbomlx.mlx_runtime.generation import GenerationStats, generate_with_backend

_NUMERIC_STAT_FIELDS = (
    "prompt_tokens",
    "generation_tokens",
    "timed_generation_tokens",
    "prompt_tps",
    "generation_tps",
    "generation_wall_time_s",
    "peak_memory_gb",
    "allocated_cache_bytes",
    "used_state_bytes",
    "key_path_bytes",
    "value_path_bytes",
    "total_kv_bytes",
    "native_working_set_bytes",
)


def median_generation_stats(stats_list: list[GenerationStats]) -> dict[str, int | float | str]:
    if not stats_list:
        raise ValueError("median_generation_stats requires at least one GenerationStats item.")
    payload: dict[str, int | float | str] = {
        field: median([getattr(item, field) for item in stats_list]) for field in _NUMERIC_STAT_FIELDS
    }
    payload["backend"] = stats_list[0].backend
    scorer_routes = {item.scorer_route for item in stats_list}
    payload["scorer_route"] = stats_list[0].scorer_route if len(scorer_routes) == 1 else "mixed"
    return payload


def run_benchmark_series(
    model,
    prompt_tokens,
    *,
    backend: str,
    config,
    generation_tokens: int,
    warmup_runs: int = 1,
    repeats: int = 3,
):
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    for _ in range(warmup_runs):
        generate_with_backend(
            model,
            prompt_tokens,
            max_tokens=generation_tokens,
            backend=backend,
            config=config,
        )

    runs: list[GenerationStats] = []
    for _ in range(repeats):
        _tokens, _logprobs, stats = generate_with_backend(
            model,
            prompt_tokens,
            max_tokens=generation_tokens,
            backend=backend,
            config=config,
        )
        runs.append(stats)

    return {
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "median": median_generation_stats(runs),
        "runs": [stats.as_dict() for stats in runs],
    }
