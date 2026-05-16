# Qwen3.5-TurboQuant-MLX-LM

`TurboMLX v0.2 Modernization Preview`

This repository packages the TurboMLX preview work for GitHub under the name `Qwen3.5-TurboQuant-MLX-LM`. The Python package and CLI remain `turbomlx`.

`v0.2` is a backward-compatible refresh on top of the `v0.1 Research Preview`:

- **Dependencies**: refreshed envelopes (`mlx>=0.31.2,<0.32`, `mlx-lm>=0.31.3,<0.32`, NumPy 2.x support, `typer>=0.16`, `pytest>=8.3`)
- **Performance**: ~52× faster `pack_bits`, `O(N log K)` `searchsorted` codebook lookup, batched perplexity scoring
- **Security**: optional `safetensors`-backed prompt-cache (`v3`) and an opt-in class allowlist for legacy `pickle` loads
- **API**: sampling strategies (`temperature`, `top-k`, `top-p`, `min-p`), `stream_with_backend` streaming events, `attention_sinks` dispatch wiring

TurboMLX `v0.2` still targets Qwen3 / Qwen3.5 full-attention `KVCache` layers only.

Its public contract for the current preview is:

- paper-faithful key-path implementations of `TurboQuantmse` and `TurboQuantprod`
- TurboMLX-owned prompt-cache save/load wrappers for a TurboQuant KV backend
- reference math utilities, mixed-precision paper profiles, and eval helpers

Important limitations:

- values are dense by default
- end-to-end KV behavior is therefore not fully paper-equivalent unless value quantization is enabled
- runtime preview is Qwen-first and currently patches `qwen3_next` plus the shared `mlx_lm.models.base` dispatch symbol
- mixed-architecture Qwen stacks remain experimental as a whole; TurboQuant conversion applies only to full-attention `KVCache` layers and leaves linear-attention `ArraysCache` layers untouched
- rotating/sliding-window families remain unsupported in preview
- `v0.2 Modernization Preview` focuses on correctness, fidelity, and surface modernization — not throughput parity with `mlx_quant` on the prefill phase
- preview runtime scoring defaults to `oracle_preview`; a narrow `native_mlx` scorer preview now exists only for Qwen3 / Qwen3.5 full-attention `KVCache` with `mode=mse`, `bits_total=4`, and `values_mode=dense`
- `native_mlx` is a Stage A remediation path, not the final packed-index direct score-space scorer
- the supported public runtime entrypoints are `generate_with_backend`, `convert_prompt_cache`, `save_prompt_cache`, and `load_prompt_cache`

## Release Status

- release label: `v0.2 Modernization Preview`
- package identity: `turbomlx`
- CLI: `turbomlx`
- supported public preview target: Qwen3 / Qwen3.5 full-attention `KVCache` only
- non-goal for this release: throughput parity with `mlx_quant`

## Install

```bash
pip install -e .                  # core install (NumPy + Typer + reference math)
pip install -e ".[mlx]"           # add the MLX runtime backend
pip install -e ".[serialize]"     # add safetensors v3 prompt-cache support
pip install -e ".[dev]"           # add pytest, ruff, mypy for development
```

## v0.2 Verification Snapshot

Tested on `2026-05-16` with:

- **Hardware**: Apple `M5 Max`, `64 GB` unified memory, macOS `26.4.1`
- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- `numpy==2.4.5`
- `safetensors==0.7.0`
- smoke model: `mlx-community/Qwen3.5-9B-MLX-4bit` (hybrid: 8 full-attention + 24 linear-attention layers, `head_dim=256`)

Verification results:

- `python3 -m compileall src` passed
- `pytest -q` -> **`97 passed, 7 skipped`** (skipped suites require the optional MLX backend on Apple Silicon)
- `ruff check src/ tests/` -> clean
- Qwen native smoke generate passed with `scorer_route = native_mlx`

### Backend comparison

All numbers are medians with `warmup_runs=1` and `repeats=3`, produced by
[`scripts/compare_backends.py`](scripts/compare_backends.py). Lower
`Key Path` / `Total KV` is better; higher TPS is better. The `Native WS`
column is the on-device cost of the `native_mlx` scorer rotation /
centroid lookup tables.

#### 128 Prompt / 16 Generation

| Route | Prompt TPS | Decode TPS | Key Path | Total KV | Native WS | Scorer Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `2421.90` | `93.52` | `53.60 MiB` | `53.60 MiB` | `0.00 MiB` | `baseline` |
| `mlx_quant` | `2366.65` | `85.33` | `50.39 MiB` | `50.39 MiB` | `0.00 MiB` | `mlx_quant` |
| `turbomlx` + `oracle_preview` | `1019.96` | `70.50` | `54.17 MiB` | `56.41 MiB` | `0.00 MiB` | `oracle_preview` |
| `turbomlx` + `native_mlx` | `1054.23` | `70.94` | `49.70 MiB` | `51.94 MiB` | `4.47 MiB` | `native_mlx` |

#### 512 Prompt / 64 Generation

| Route | Prompt TPS | Decode TPS | Key Path | Total KV | Native WS | Scorer Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `3252.99` | `93.55` | `67.10 MiB` | `67.10 MiB` | `0.00 MiB` | `baseline` |
| `mlx_quant` | `3225.92` | `87.67` | `54.18 MiB` | `54.18 MiB` | `0.00 MiB` | `mlx_quant` |
| `turbomlx` + `oracle_preview` | `1274.40` | `67.32` | `69.41 MiB` | `78.40 MiB` | `0.00 MiB` | `oracle_preview` |
| `turbomlx` + `native_mlx` | `1327.78` | `68.07` | `51.44 MiB` | `60.43 MiB` | `17.97 MiB` | `native_mlx` |

#### 2048 Prompt / 64 Generation

| Route | Prompt TPS | Decode TPS | Key Path | Total KV | Native WS | Scorer Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `3436.54` | `90.34` | `115.10 MiB` | `115.10 MiB` | `0.00 MiB` | `baseline` |
| `mlx_quant` | `3085.89` | `84.31` | `67.68 MiB` | `67.68 MiB` | `0.00 MiB` | `mlx_quant` |
| `turbomlx` + `oracle_preview` | `1207.17` | `66.68` | `123.60 MiB` | `156.58 MiB` | `0.00 MiB` | `oracle_preview` |
| `turbomlx` + `native_mlx` | `1182.38` | `65.73` | **`57.63 MiB`** | **`90.62 MiB`** | `65.97 MiB` | `native_mlx` |

Interpretation:

- this snapshot is environment-specific and not a throughput guarantee
- at `2048/64`, `turbomlx` + `native_mlx` cuts the key-path footprint **`50.0%` versus `baseline`** (`115.10` -> `57.63 MiB`) and **`14.9%` versus `mlx_quant`** (`67.68` -> `57.63 MiB`) — confirming that TurboQuant's key-path compression beats stock 4-bit affine on the long-context profile that motivated the paper
- inside the `turbomlx` backend, switching from `oracle_preview` to `native_mlx` reduces key-path memory by **`53.4%`** at `2048/64` (`123.60` -> `57.63 MiB`) and total KV bytes by **`42.1%`** (`156.58` -> `90.62 MiB`)
- prompt-prefill throughput is still preview-grade (~`1/3` of `baseline`); this is the headline cost of the current NumPy ↔ MLX boundary in the quantization path and is the next target for `v0.3`

### Fidelity vs baseline (last-token logits)

Same prompt replayed across backends; `cos` is the cosine similarity of
the trailing logits, `top-K` is the share of the K highest-logit tokens
that match `baseline`, and `mean|Δlogprob|` is the average absolute
difference between normalized log-probabilities. Generated by
[`scripts/quality_check.py`](scripts/quality_check.py).

#### Short prompt (37 tokens)

| Route | Cosine sim | Top-1 | Top-5 | Top-10 | mean \|Δlogprob\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlx_quant` | `0.996038` | `1.00` | `1.00` | `0.90` | `0.2019` |
| `turbomlx` + `oracle_preview` | `0.998911` | `1.00` | `1.00` | `0.90` | `0.0855` |
| `turbomlx` + `native_mlx` | `0.998911` | `1.00` | `1.00` | `0.90` | `0.0859` |

#### Longer prompt (400 tokens)

| Route | Cosine sim | Top-1 | Top-5 | Top-10 | mean \|Δlogprob\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlx_quant` | `0.999646` | `1.00` | `1.00` | `0.90` | `0.1872` |
| `turbomlx` + `oracle_preview` | **`0.999948`** | `1.00` | `1.00` | **`1.00`** | `0.1017` |
| `turbomlx` + `native_mlx` | **`0.999941`** | `1.00` | `1.00` | **`1.00`** | **`0.0878`** |

Interpretation:

- TurboQuant's key-path compression is **measurably more faithful to the baseline distribution than `mlx_quant`** on this Qwen3.5-9B-MLX-4bit smoke target — higher cosine similarity, full top-10 agreement on the long prompt, and ~half the mean `|Δlogprob|`
- `oracle_preview` and `native_mlx` produce statistically equivalent logits at `bits_total=4`, as expected: they share the same quantization and only differ in the scoring path
- this is a smoke-grade signal, not a benchmark; for end-to-end task evaluation use `eval-needle` / `eval-jsonl` against a real dataset

## Bit Semantics

- `bits_total` is the user-facing total per-channel bit budget
- `mode=mse`: all `bits_total` go to the Lloyd-Max main quantizer
- `mode=prod`: `bits_mse = bits_total - 1`, plus a 1-bit QJL residual
- `mode=prod` is supported only for `bits_total >= 2`
- default policy:
  - `1-bit`: `mse`
  - `2-bit`: `prod`
  - `3/4-bit`: `mse`

## Mixed Precision Paper Profile

Paper-style non-integer effective settings such as `2.5` and `3.5` bits are
represented as explicit mixed-precision outlier profiles.

Supported profile knobs:

- `outlier_channels`
- `outlier_high_bits`
- `regular_bits`
- `outlier_selection_policy`

If mixed precision is disabled, all quality and memory claims are restricted to
integer-bit configurations.

## Release Policy

- `v0.1 Research Preview`
  - correctness
  - serialization stability
  - prompt-cache continuity
  - long-context quality helpers
  - honest benchmark reporting
- `v0.2 Modernization Preview` (current)
  - all of `v0.1`
  - refreshed MLX / mlx-lm / NumPy 2.x / Typer envelopes
  - performance: vectorized `pack_bits` (~`52x`), `searchsorted` codebook lookup, chunked perplexity
  - security: optional `safetensors` v3 prompt-cache + class allowlist
  - API: `make_sampler`, `stream_with_backend`, attention-sinks dispatch wiring
- `v1.0 Stable`
  - all of the above
  - explicit `mlx_quant` decode parity target on the reference benchmark matrix

## Scope

Official preview target:

- Qwen3 / Qwen3.5 full-attention layers that use the default non-rotating `KVCache` path

Experimental:

- mixed full-attention / linear-attention Qwen stacks as a whole
- value quantization on the dense-key preview fallback path
- rotating or sliding-window cache families

## Preview Eval Surface

- `eval-needle` is a preview retrieval harness that inserts the needle only inside the haystack context
- `eval-jsonl` is a local JSONL exact/substring harness and is not a real LongBench-E implementation
- prompt-cache roundtrip support is provided by `turbomlx.save_prompt_cache()` and `turbomlx.load_prompt_cache()`, not by upstream `mlx-lm` loaders

## Prompt-Cache Policy

- prompt-cache files are trusted-local-only
- schema `v3` is the current default write format when the optional `safetensors` dependency is installed (`pip install turbomlx[serialize]`)
  - structural metadata lives in a typed JSON header
  - numerical arrays live in a `safetensors` container — zero-copy reads, no pickle bytecode in the payload
  - a 4-byte `TQS3` magic prefix lets the loader auto-detect the format
- schema `v2` (pickle with stable `cache_type_id` metadata) remains writable for environments without `safetensors`
- schema `v1` files still load in read-only compatibility mode via deprecated `class_path` fallback, but only for class paths registered through `register_cache_type()`
- the v1 / v2 loader no longer imports arbitrary `class_path` values — third-party cache classes must opt in via `turbomlx.register_cache_type(cache_type_id, class_path)`
- if you load an older cache, re-save it with TurboMLX `>=0.2` to migrate to schema `v3`

## Qwen Preview Runtime

- Qwen3 / Qwen3.5 preview correctness uses a dense reconstructed key fallback on full-attention layers
- grouped-query attention math is delegated back to MLX native SDPA after reconstructing dense keys from the TurboQuant cache
- this path is correctness-first and intentionally preview-grade; it is not a throughput claim
- `--scorer-mode native_mlx` is a narrower preview path layered on top of the same Qwen-first contract
- supported `native_mlx` config:
  - Qwen3 / Qwen3.5
  - full-attention `KVCache`
  - `mode=mse`
  - `bits_total=4`
  - `values_mode=dense`
- unsupported `native_mlx` combinations emit a warning once per reason and fall back to the preview scorer path
- the current native scorer is still an intermediate on-device remediation step, not the final packed-index direct-score architecture

## Verification

- unit and regression suite: `pytest -q` (uses `pythonpath = ["src"]` from `pyproject.toml`)
- lint: `ruff check src/ tests/`
- MLX smoke and benchmark authority: any Python 3.11+ venv with `pip install -e ".[mlx,dev]"` on Apple Silicon
- recommended smoke target: set `TURBOMLX_SMOKE_QWEN_MODEL=/path/to/mlx-community/Qwen3.5-9B-MLX-4bit` and run `pytest tests/test_mlx_smoke.py`
- side-by-side backend benchmark: `python scripts/compare_backends.py --prompt-tokens 2048 --generation-tokens 64`
- last-token logit fidelity check: `python scripts/quality_check.py`
- benchmark methodology for current preview work:
  - use at least 1 warmup run
  - use at least 3 measured repeats
  - report median results
  - inspect `scorer_route` in the output to verify whether `native_mlx` actually ran or fell back
  - inspect `timed_generation_tokens` and `native_working_set_bytes` in the output when comparing preview scorer routes

## Preview Bundle

- this source tree is a preview-candidate working tree, not a release-ready artifact
- use `scripts/export_preview_bundle.py` to create a clean shareable preview bundle under `dist/`
- use `python3 scripts/export_preview_bundle.py --output-dir /ABS/PATH/Qwen3.5-TurboQuant-MLX-LM` to create a clean public source tree
- distribute the exported preview bundle, not a raw workspace zip
- the exported bundle excludes local virtualenvs, cache directories, transient artifacts, and reference PDFs

## Tested Runtime Stack

- `mlx>=0.31.2,<0.32`
- `mlx-lm>=0.31.3,<0.32`
- `numpy>=1.26,<3` (NumPy 2.x supported)
- `typer>=0.16`
- optional `safetensors>=0.4.5` for the `v3` prompt-cache format

This repository started from a blank directory plus the TurboQuant paper, so
the current implementation emphasizes clean interfaces and verifiable reference
math first. MLX runtime hardening is intentionally staged behind the preview
release boundary.

## v0.2 API Cheat Sheet

```python
from turbomlx import (
    TurboQuantConfig, ScorerMode, ValuesMode,
    generate_with_backend, stream_with_backend,
    make_sampler, save_prompt_cache, load_prompt_cache,
)

config = TurboQuantConfig(bits_total=4, scorer_mode=ScorerMode.NATIVE_MLX)
sampler = make_sampler(temperature=0.7, top_p=0.9, seed=42)

# One-shot generation
tokens, _logprobs, stats = generate_with_backend(
    model, prompt, max_tokens=64, backend="turbomlx",
    config=config, sampler=sampler,
)

# Streaming generation
for event in stream_with_backend(
    model, prompt, max_tokens=128, backend="turbomlx",
    config=config, sampler=sampler,
):
    print(event.token, event.logprob, event.position)

# safetensors-backed prompt cache (auto-selects v3 when installed)
save_prompt_cache("cache.tqcache", prompt_cache_list)
restored = load_prompt_cache("cache.tqcache")
```

## v0.2 CLI Additions

The `turbomlx generate` command grew sampler flags:

```bash
turbomlx generate MODEL_ID "PROMPT" \
    --backend turbomlx \
    --bits-total 4 \
    --scorer-mode native_mlx \
    --temperature 0.7 \
    --top-p 0.9 \
    --top-k 40 \
    --min-p 0.05 \
    --seed 42
```

`--temperature 0` (the default) selects greedy argmax decoding so existing
invocations behave exactly like `v0.1`.

## Logging

TurboMLX uses the standard `logging` module with a `NullHandler` attached by
default. Opt in to diagnostic output:

```bash
TURBOMLX_LOG_LEVEL=DEBUG turbomlx benchmark MODEL_ID
```

or programmatically:

```python
import logging
logging.getLogger("turbomlx").setLevel(logging.INFO)
```
