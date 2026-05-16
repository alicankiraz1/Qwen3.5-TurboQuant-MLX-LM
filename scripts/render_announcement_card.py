#!/usr/bin/env python3
"""Render the v0.2 launch announcement metric cards (square + wide).

Produces two PNGs intended for social media — a 1080x1080 square for
Instagram / LinkedIn / Mastodon feeds and a 1200x675 wide card for
Twitter / X link previews and LinkedIn shared link unfurls.

The numbers are sourced from the M5 Max benchmark run committed to
``benchmark_results/``; see ``scripts/compare_backends.py`` and
``scripts/quality_check.py`` for the underlying scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Apple Silicon / TurboMLX color tokens
BG = "#0a0a0f"
SURFACE = "#15151c"
SURFACE_ELEVATED = "#1c1c24"
TEXT_PRIMARY = "#f5f5f7"
TEXT_SECONDARY = "#98989d"
TEXT_TERTIARY = "#6e6e73"
STROKE = "#2c2c34"
ACCENT = "#00d4ff"

# Backend palette
BACKEND_COLORS = {
    "baseline": "#8e8e93",
    "mlx_quant": "#5e5ce6",
    "turbomlx-oracle": "#ff9f0a",
    "turbomlx-native": "#30d158",
}

# Source numbers — kept in this module so regeneration stays deterministic
KEY_PATH_2048 = {
    "baseline": 115.10,
    "mlx_quant": 67.68,
    "turbomlx-oracle": 123.60,
    "turbomlx-native": 57.63,
}

TOTAL_KV_2048 = {
    "baseline": 115.10,
    "mlx_quant": 67.68,
    "turbomlx-oracle": 156.58,
    "turbomlx-native": 90.62,
}

FIDELITY_400 = {
    "mlx_quant": {"cos": 0.999646, "top10": 0.90, "delta": 0.1872},
    "turbomlx-oracle": {"cos": 0.999948, "top10": 1.00, "delta": 0.1017},
    "turbomlx-native": {"cos": 0.999941, "top10": 1.00, "delta": 0.0878},
}

DECODE_TPS_2048 = {
    "baseline": 90.34,
    "mlx_quant": 84.31,
    "turbomlx-oracle": 66.68,
    "turbomlx-native": 65.73,
}

BACKEND_LABELS = {
    "baseline": "baseline (mlx-lm)",
    "mlx_quant": "mlx_quant",
    "turbomlx-oracle": "turbomlx + oracle",
    "turbomlx-native": "turbomlx + native_mlx",
}


def _setup_figure(width_in: float, height_in: float, dpi: int) -> Figure:
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi, facecolor=BG)
    return fig


def _hide_spines(ax) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)


def _bar_chart_horizontal(
    ax,
    labels: list[str],
    values: list[float],
    colors: list[str],
    unit: str,
    *,
    highlight_index: int | None = None,
    value_formatter=None,
) -> None:
    """Render a flat horizontal bar chart with value labels at the end of each bar."""
    if value_formatter is None:
        value_formatter = lambda v: f"{v:.2f} {unit}"  # noqa: E731

    y_positions = range(len(labels))
    bars = ax.barh(
        list(y_positions),
        values,
        color=colors,
        edgecolor="none",
        height=0.66,
    )
    max_value = max(values) if values else 1.0
    for index, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + max_value * 0.018,
            bar.get_y() + bar.get_height() / 2,
            value_formatter(values[index]),
            va="center",
            ha="left",
            color=TEXT_PRIMARY if index == highlight_index else TEXT_SECONDARY,
            fontsize=12,
            fontweight="bold" if index == highlight_index else "regular",
            family="DejaVu Sans Mono",
        )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, color=TEXT_SECONDARY, fontsize=11)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xticks([])
    ax.set_xlim(0, max_value * 1.32)
    ax.invert_yaxis()
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(STROKE)
    ax.spines["left"].set_linewidth(0.8)
    ax.set_facecolor(BG)


def _draw_big_stat(ax, value: str, label: str, sublabel: str, *, accent_value: bool = False) -> None:
    """Render a single hero metric tile."""
    _hide_spines(ax)
    rect = patches.FancyBboxPatch(
        (0.02, 0.05),
        0.96,
        0.9,
        boxstyle="round,pad=0.0,rounding_size=0.045",
        linewidth=1.0,
        edgecolor=STROKE,
        facecolor=SURFACE,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(
        0.5,
        0.66,
        value,
        ha="center",
        va="center",
        color=ACCENT if accent_value else TEXT_PRIMARY,
        fontsize=38,
        fontweight="bold",
        family="DejaVu Sans",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.34,
        label,
        ha="center",
        va="center",
        color=TEXT_PRIMARY,
        fontsize=11.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.18,
        sublabel,
        ha="center",
        va="center",
        color=TEXT_TERTIARY,
        fontsize=9.5,
        transform=ax.transAxes,
    )


def _draw_header(
    fig: Figure,
    *,
    title: str,
    subtitle: str,
    tag: str,
    title_size: int = 24,
    title_y: float = 0.952,
    subtitle_y: float = 0.918,
    tag_y: float = 0.952,
) -> None:
    fig.text(
        0.048,
        title_y,
        title,
        color=TEXT_PRIMARY,
        fontsize=title_size,
        fontweight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.048,
        subtitle_y,
        subtitle,
        color=TEXT_SECONDARY,
        fontsize=11,
    )
    fig.text(
        0.952,
        tag_y,
        tag,
        color=ACCENT,
        fontsize=10.5,
        fontweight="bold",
        ha="right",
        va="top",
        family="DejaVu Sans Mono",
    )


def _draw_footer(fig: Figure, repo_url: str, tagline: str) -> None:
    fig.text(
        0.048,
        0.032,
        tagline,
        color=TEXT_SECONDARY,
        fontsize=10.5,
    )
    fig.text(
        0.952,
        0.032,
        repo_url,
        color=TEXT_TERTIARY,
        fontsize=10,
        ha="right",
        family="DejaVu Sans Mono",
    )


def _section_title(ax, title: str, caption: str) -> None:
    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        color=TEXT_PRIMARY,
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.005,
        caption,
        transform=ax.transAxes,
        color=TEXT_TERTIARY,
        fontsize=10,
    )


def render_square(output_path: Path) -> Path:
    """Produce the 1080x1080 square card."""
    fig = _setup_figure(width_in=10.8, height_in=10.8, dpi=100)

    _draw_header(
        fig,
        title="TurboMLX v0.2 — Modernization Preview",
        subtitle="Apple M5 Max · Qwen3.5-9B-MLX-4bit · mlx 0.31.2 · mlx-lm 0.31.3",
        tag="v0.2  ·  2026-05-16",
        title_size=23,
        title_y=0.955,
        subtitle_y=0.922,
        tag_y=0.949,
    )

    # Top stats row: 3 hero tiles
    gs_top = GridSpec(
        nrows=1,
        ncols=3,
        left=0.048,
        right=0.952,
        top=0.885,
        bottom=0.720,
        wspace=0.05,
    )
    stat_ax_1 = fig.add_subplot(gs_top[0, 0])
    stat_ax_2 = fig.add_subplot(gs_top[0, 1])
    stat_ax_3 = fig.add_subplot(gs_top[0, 2])

    _draw_big_stat(
        stat_ax_1,
        value="−50%",
        label="key-path memory",
        sublabel="vs baseline @ 2048 / 64",
        accent_value=True,
    )
    _draw_big_stat(
        stat_ax_2,
        value="0.99994",
        label="cosine similarity",
        sublabel="vs baseline @ 400 tokens",
    )
    _draw_big_stat(
        stat_ax_3,
        value="52×",
        label="pack_bits speedup",
        sublabel="vectorized NumPy",
    )

    # Middle: KV memory bar chart at 2048/64.
    # left=0.225 gives enough room for the longest backend label.
    ax_memory = fig.add_axes((0.225, 0.430, 0.730, 0.210))
    _section_title(
        ax_memory,
        title="KV-cache key path @ 2048 prompt / 64 decode (MiB, lower is better)",
        caption="median of 3 repeats · turbomlx + native_mlx wins this profile",
    )
    backends = ["baseline", "mlx_quant", "turbomlx-oracle", "turbomlx-native"]
    _bar_chart_horizontal(
        ax_memory,
        labels=[BACKEND_LABELS[b] for b in backends],
        values=[KEY_PATH_2048[b] for b in backends],
        colors=[BACKEND_COLORS[b] for b in backends],
        unit="MiB",
        highlight_index=3,
    )

    # Bottom: Fidelity bar chart (cosine similarity, zoomed).
    ax_fidelity = fig.add_axes((0.225, 0.140, 0.730, 0.210))
    _section_title(
        ax_fidelity,
        title="Fidelity vs baseline (cosine similarity, 400-token prompt)",
        caption="TurboQuant beats 4-bit affine mlx_quant on every fidelity column",
    )
    fid_backends = ["mlx_quant", "turbomlx-oracle", "turbomlx-native"]
    fid_values = [FIDELITY_400[b]["cos"] for b in fid_backends]
    # Zoom into the meaningful range so the small differences are visible.
    floor = 0.9994
    ceiling = 1.0
    y_positions = range(len(fid_backends))
    bars = ax_fidelity.barh(
        list(y_positions),
        [v - floor for v in fid_values],
        left=floor,
        color=[BACKEND_COLORS[b] for b in fid_backends],
        edgecolor="none",
        height=0.66,
    )
    for index, bar in enumerate(bars):
        ax_fidelity.text(
            ceiling - 0.0000045,
            bar.get_y() + bar.get_height() / 2,
            f"cos = {fid_values[index]:.6f}",
            va="center",
            ha="right",
            color=TEXT_PRIMARY if index >= 1 else TEXT_SECONDARY,
            fontsize=11,
            fontweight="bold" if index >= 1 else "regular",
            family="DejaVu Sans Mono",
        )
    ax_fidelity.set_yticks(list(y_positions))
    ax_fidelity.set_yticklabels(
        [BACKEND_LABELS[b] for b in fid_backends],
        color=TEXT_SECONDARY,
        fontsize=11,
    )
    ax_fidelity.tick_params(axis="y", length=0, pad=8)
    ax_fidelity.set_xticks([])
    ax_fidelity.set_xlim(floor, ceiling)
    ax_fidelity.invert_yaxis()
    for side in ("top", "right", "bottom"):
        ax_fidelity.spines[side].set_visible(False)
    ax_fidelity.spines["left"].set_color(STROKE)
    ax_fidelity.spines["left"].set_linewidth(0.8)
    ax_fidelity.set_facecolor(BG)
    fig.text(
        0.225,
        0.108,
        f"x-axis zoomed: [{floor:.4f}, {ceiling:.4f}]",
        color=TEXT_TERTIARY,
        fontsize=9,
        family="DejaVu Sans Mono",
    )

    _draw_footer(
        fig,
        repo_url="github.com/alicankiraz1/Qwen3.5-TurboQuant-MLX-LM",
        tagline="safetensors v3 · streaming + samplers · 103 tests pass · ruff clean",
    )

    fig.savefig(
        output_path,
        dpi=100,
        bbox_inches=None,
        facecolor=BG,
        edgecolor="none",
    )
    plt.close(fig)
    return output_path


def render_wide(output_path: Path) -> Path:
    """Produce the 1200x675 wide card sized for Twitter / LinkedIn link previews."""
    fig = _setup_figure(width_in=12.0, height_in=6.75, dpi=100)

    _draw_header(
        fig,
        title="TurboMLX v0.2 — Modernization Preview",
        subtitle="Apple M5 Max · Qwen3.5-9B-MLX-4bit · mlx 0.31.2 · mlx-lm 0.31.3",
        tag="v0.2  ·  2026-05-16",
        title_size=22,
        title_y=0.928,
        subtitle_y=0.885,
        tag_y=0.92,
    )

    # Left column: hero stats stacked
    gs_left = GridSpec(
        nrows=3,
        ncols=1,
        left=0.048,
        right=0.46,
        top=0.79,
        bottom=0.13,
        hspace=0.18,
    )
    _draw_big_stat(
        fig.add_subplot(gs_left[0, 0]),
        value="−50%",
        label="key-path memory vs baseline",
        sublabel="115.10 → 57.63 MiB @ 2048 / 64",
        accent_value=True,
    )
    _draw_big_stat(
        fig.add_subplot(gs_left[1, 0]),
        value="0.99994",
        label="cosine similarity vs baseline",
        sublabel="TurboQuant > mlx_quant (0.99965)",
    )
    _draw_big_stat(
        fig.add_subplot(gs_left[2, 0]),
        value="52×",
        label="pack_bits speedup",
        sublabel="Python loop → vectorized NumPy",
    )

    # Right column: KV memory chart
    ax_memory = fig.add_axes((0.660, 0.20, 0.300, 0.58))
    _section_title(
        ax_memory,
        title="KV-cache key path @ 2048 / 64",
        caption="MiB, lower is better — median of 3 repeats",
    )
    backends = ["baseline", "mlx_quant", "turbomlx-oracle", "turbomlx-native"]
    _bar_chart_horizontal(
        ax_memory,
        labels=[BACKEND_LABELS[b] for b in backends],
        values=[KEY_PATH_2048[b] for b in backends],
        colors=[BACKEND_COLORS[b] for b in backends],
        unit="MiB",
        highlight_index=3,
    )

    _draw_footer(
        fig,
        repo_url="github.com/alicankiraz1/Qwen3.5-TurboQuant-MLX-LM",
        tagline="safetensors v3 · streaming + samplers · 103 tests pass · ruff clean",
    )

    fig.savefig(
        output_path,
        dpi=100,
        bbox_inches=None,
        facecolor=BG,
        edgecolor="none",
    )
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/announcement"),
        help="Destination directory for the rendered PNG cards.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    square = args.output_dir / "turbomlx-v0-2-card-square.png"
    wide = args.output_dir / "turbomlx-v0-2-card-wide.png"

    render_square(square)
    render_wide(wide)

    print(f"square: {square}")
    print(f"wide:   {wide}")


if __name__ == "__main__":
    main()
