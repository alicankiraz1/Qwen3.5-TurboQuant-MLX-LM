import json

import numpy as np
from typer.testing import CliRunner

from turbomlx.cli import app

runner = CliRunner()


def test_eval_ppl_invokes_backend_path(monkeypatch, tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("alpha beta gamma")
    captured = {}

    def fake_eval(model_id, path, *, backend, config):
        captured["model_id"] = model_id
        captured["path"] = path
        captured["backend"] = backend
        captured["config"] = config
        return {"ok": True}

    monkeypatch.setattr("turbomlx.cli.perplexity_from_text_file", fake_eval)
    result = runner.invoke(
        app,
        [
            "eval-ppl",
            "demo-model",
            str(text_file),
            "--backend",
            "baseline",
            "--bits-total",
            "3",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["ok"] is True
    assert captured["model_id"] == "demo-model"
    assert captured["backend"] == "baseline"
    assert captured["config"].bits_total == 3


def test_eval_needle_invokes_backend_generation(monkeypatch, tmp_path):
    context_file = tmp_path / "context.txt"
    context_file.write_text("context body")
    captured = {}

    def fake_eval(
        model_id,
        context,
        needle,
        query,
        *,
        backend,
        config,
        max_tokens,
        insertion_depth_pct,
        seed,
    ):
        captured["model_id"] = model_id
        captured["context"] = context
        captured["needle"] = needle
        captured["query"] = query
        captured["backend"] = backend
        captured["config"] = config
        captured["max_tokens"] = max_tokens
        captured["insertion_depth_pct"] = insertion_depth_pct
        captured["seed"] = seed
        return {"needle_score": 1.0}

    monkeypatch.setattr("turbomlx.cli.run_needle_backend_eval", fake_eval)
    result = runner.invoke(
        app,
        [
            "eval-needle",
            "demo-model",
            str(context_file),
            "secret fact",
            "What is the fact?",
            "--backend",
            "turbomlx",
            "--max-tokens",
            "21",
            "--insertion-depth-pct",
            "65",
            "--seed",
            "9",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["needle_score"] == 1.0
    assert captured["model_id"] == "demo-model"
    assert captured["context"] == "context body"
    assert captured["backend"] == "turbomlx"
    assert captured["max_tokens"] == 21
    assert captured["insertion_depth_pct"] == 65
    assert captured["seed"] == 9


def test_eval_jsonl_invokes_backend_generation(monkeypatch, tmp_path):
    dataset_file = tmp_path / "dataset.jsonl"
    dataset_file.write_text('{"prompt":"hello","answers":["world"]}\n')
    captured = {}

    def fake_eval(model_id, path, *, backend, config, max_tokens, limit):
        captured["model_id"] = model_id
        captured["path"] = path
        captured["backend"] = backend
        captured["config"] = config
        captured["max_tokens"] = max_tokens
        captured["limit"] = limit
        return {"rows": 1}

    monkeypatch.setattr("turbomlx.cli.evaluate_jsonl_backend", fake_eval)
    result = runner.invoke(
        app,
        [
            "eval-jsonl",
            "demo-model",
            str(dataset_file),
            "--backend",
            "mlx_quant",
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["rows"] == 1
    assert captured["model_id"] == "demo-model"
    assert captured["backend"] == "mlx_quant"
    assert captured["limit"] == 2


def test_benchmark_invokes_series_runner_with_repeats_and_warmup(monkeypatch):
    captured = {}

    class _FakeMx:
        int32 = np.int32

        class random:
            @staticmethod
            def randint(_low, _high, shape, dtype=None):
                return np.zeros(shape, dtype=np.int32)

    def fake_require_runtime():
        return _FakeMx(), object(), object()

    def fake_loader(_model_id, return_config=False):
        assert return_config is True
        return object(), object(), {"vocab_size": 128}

    def fake_benchmark_series(model, prompt, *, backend, config, generation_tokens, warmup_runs, repeats):
        captured["model"] = model
        captured["prompt_shape"] = tuple(int(dim) for dim in prompt.shape)
        captured["backend"] = backend
        captured["config"] = config
        captured["generation_tokens"] = generation_tokens
        captured["warmup_runs"] = warmup_runs
        captured["repeats"] = repeats
        return {
            "warmup_runs": warmup_runs,
            "repeats": repeats,
            "median": {
                "generation_tps": 10.0,
                "timed_generation_tokens": 6,
                "native_working_set_bytes": 128,
                "scorer_route": "native_mlx",
            },
            "runs": [],
        }

    monkeypatch.setattr("turbomlx.cli._require_runtime", fake_require_runtime)
    monkeypatch.setattr("turbomlx.cli._load_mlx_lm_loader", lambda: fake_loader)
    monkeypatch.setattr("turbomlx.cli.run_benchmark_series", fake_benchmark_series)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "demo-model",
            "--backend",
            "turbomlx",
            "--prompt-tokens",
            "32",
            "--generation-tokens",
            "7",
            "--warmup-runs",
            "2",
            "--repeats",
            "4",
            "--bits-total",
            "4",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["warmup_runs"] == 2
    assert payload["repeats"] == 4
    assert payload["median"]["scorer_route"] == "native_mlx"
    assert payload["median"]["timed_generation_tokens"] == 6
    assert payload["median"]["native_working_set_bytes"] == 128
    assert captured["backend"] == "turbomlx"
    assert captured["generation_tokens"] == 7
    assert captured["warmup_runs"] == 2
    assert captured["repeats"] == 4
    assert captured["prompt_shape"] == (32,)
