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
- `v0.1 Research Preview` focuses on correctness and quality gates, not throughput leadership
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

## Latest Verification Snapshot

Tested on `2026-03-29` with:

- `mlx==0.31.1`
- `mlx-lm==0.31.1`
- smoke model: `mlx-community/Qwen3.5-9B-MLX-4bit`

Verification results:

- `python3 -m compileall src` passed
- `PYTHONPATH=src .venv/bin/python -m pytest -q` -> `72 passed, 1 skipped`
- Qwen native smoke generate passed with `scorer_route = native_mlx`
- short native smoke snapshot:
  - `prompt_tps`: `54.70`
  - `generation_tps`: `42.59`
  - `key_path_bytes`: `26504384`
  - `total_kv_bytes`: `27110976`
  - `native_working_set_bytes`: `1212416`

Benchmark snapshot on the tested Apple Silicon stack. All numbers below are medians with warmup `1` and repeats `3`.

### 512 Prompt / 64 Generation

| Route | Prompt TPS | Decode TPS | Key Path | Total KV | Native Working Set | Scorer Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `1394.12` | `55.00` | `43.10 MiB` | `43.10 MiB` | `0.00 MiB` | `baseline` |
| `mlx_quant` | `1366.74` | `47.83` | `30.18 MiB` | `30.18 MiB` | `0.00 MiB` | `mlx_quant` |
| `turbomlx` + `oracle_preview` | `285.39` | `42.02` | `45.41 MiB` | `54.40 MiB` | `0.00 MiB` | `oracle_preview` |
| `turbomlx` + `native_mlx` | `380.04` | `42.71` | `27.44 MiB` | `36.43 MiB` | `17.97 MiB` | `native_mlx` |

### 2048 Prompt / 64 Generation

| Route | Prompt TPS | Decode TPS | Key Path | Total KV | Native Working Set | Scorer Route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | `1419.62` | `54.32` | `91.10 MiB` | `91.10 MiB` | `0.00 MiB` | `baseline` |
| `mlx_quant` | `1365.77` | `52.10` | `43.68 MiB` | `43.68 MiB` | `0.00 MiB` | `mlx_quant` |
| `turbomlx` + `oracle_preview` | `285.52` | `39.50` | `99.60 MiB` | `132.58 MiB` | `0.00 MiB` | `oracle_preview` |
| `turbomlx` + `native_mlx` | `401.84` | `40.44` | `33.63 MiB` | `66.62 MiB` | `65.97 MiB` | `native_mlx` |

Interpretation:

- this snapshot is environment-specific and not a throughput guarantee
- the strongest current TurboQuant signal is inside the same `turbomlx` backend: `scorer_route = native_mlx` produces nonzero `native_working_set_bytes`, lower key-path memory, and lower total KV bytes than `oracle_preview`
- at `512/64`, `native_mlx` reduced key-path memory by `39.57%`, reduced total KV bytes by `33.03%`, improved prompt TPS by `33.16%`, and improved decode TPS by `1.65%` versus `oracle_preview`
- at `2048/64`, `native_mlx` reduced key-path memory by `66.23%`, reduced total KV bytes by `49.76%`, improved prompt TPS by `40.74%`, and improved decode TPS by `2.38%` versus `oracle_preview`
- this repo does not claim throughput parity with `baseline` or `mlx_quant`; the current research-preview claim is that TurboQuant is active and measurably changes the `turbomlx` runtime profile

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

- prompt-cache files are trusted-local-only and currently remain `pickle`-backed
- schema `v2` is the current write format and includes `cache_type_id` metadata
- schema `v1` files still load in read-only compatibility mode via deprecated `class_path` fallback
- if you load an older cache, re-save it to migrate to `v2`

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

- unit and regression suite: `PYTHONPATH=src .venv/bin/python -m pytest -q`
- MLX smoke and benchmark authority: `.venv312`
- recommended smoke target: `TURBOMLX_SMOKE_QWEN_MODEL=/Users/alican/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-4bit`
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
