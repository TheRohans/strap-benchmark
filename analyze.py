#!/usr/bin/env python3
"""
Load benchmark result JSON files and produce comparison charts.

Usage:
    python analyze.py                            # show charts interactively
    python analyze.py --out-dir reports/         # save all charts as PNGs
    python analyze.py results/20260603_*.json    # specific files
    python analyze.py --source gsm8k             # filter by dataset
    python analyze.py --out report.png --chart source  # single chart to file
    python analyze.py --chart ruler --out-dir reports/ # RULER heatmap only
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


RUNNER_COLORS = {
    "strap":     "#4e79a7",
    "base_qwen": "#f28e2b",
    "opencode":  "#59a14f",
}
DEFAULT_COLOR = "#bab0ac"


# ── data loading ──────────────────────────────────────────────────────────────

def load_file(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if "results" in raw and "meta" in raw:
        return raw
    return {"meta": {"timestamp": path.stem, "datasets": ["unknown"], "sources": []}, "results": raw}


def load_all(paths: list[Path], source_filter: str | None) -> list[dict]:
    runs = []
    for p in sorted(paths):
        run = load_file(p)
        if source_filter:
            run["results"] = {
                runner: [r for r in records if r.get("source") == source_filter]
                for runner, records in run["results"].items()
            }
        runs.append(run)
    return runs


def records_for(run: dict, runner: str, source: str | None = None) -> list[dict]:
    rs = run["results"].get(runner, [])
    if source:
        rs = [r for r in rs if r.get("source") == source]
    return rs


def accuracy(records: list[dict]) -> float | None:
    if not records:
        return None
    return sum(r["pass"] for r in records) / len(records) * 100


def avg_elapsed(records: list[dict]) -> float | None:
    if not records:
        return None
    return sum(r["elapsed"] for r in records) / len(records)


def all_runners(runs: list[dict]) -> list[str]:
    seen, out = set(), []
    for run in runs:
        for r in run["results"]:
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def all_sources(runs: list[dict]) -> list[str]:
    seen, out = set(), []
    for run in runs:
        for records in run["results"].values():
            for r in records:
                s = r.get("source", "local")
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


# ── plots ─────────────────────────────────────────────────────────────────────

def bar_group(ax, groups: list[str], data: dict[str, list[float | None]],
              ylabel: str, title: str, ylim: tuple | None = None) -> None:
    runners = list(data.keys())
    n_groups = len(groups)
    n_runners = len(runners)
    width = 0.8 / n_runners
    x = np.arange(n_groups)

    for i, runner in enumerate(runners):
        vals = data[runner]
        heights = [v if v is not None else 0 for v in vals]
        color = RUNNER_COLORS.get(runner, DEFAULT_COLOR)
        bars = ax.bar(x + i * width - (n_runners - 1) * width / 2,
                      heights, width * 0.9, label=runner, color=color)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="lower right")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)


def plot_by_source(runs: list[dict], out_path: str | None) -> None:
    runners = all_runners(runs)
    sources = all_sources(runs)

    if not sources:
        print("No data to plot.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(sources) * 2.5), 5))
    fig.suptitle("Benchmark comparison by dataset", fontsize=13, fontweight="bold")

    acc_data: dict[str, list[float | None]] = {r: [] for r in runners}
    spd_data: dict[str, list[float | None]] = {r: [] for r in runners}

    for runner in runners:
        for source in sources:
            all_records: list[dict] = []
            for run in runs:
                all_records += records_for(run, runner, source)
            acc_data[runner].append(accuracy(all_records))
            spd_data[runner].append(avg_elapsed(all_records))

    bar_group(axes[0], sources, acc_data, "Accuracy (%)", "Accuracy by dataset", ylim=(0, 108))
    bar_group(axes[1], sources, spd_data, "Avg time (s)", "Avg elapsed by dataset")

    plt.tight_layout()
    _finish(fig, out_path)


def plot_over_time(runs: list[dict], out_path: str | None) -> None:
    if len(runs) < 2:
        print("Need ≥2 result files to plot over time.")
        return

    runners = all_runners(runs)
    timestamps = [r["meta"]["timestamp"] for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(runs) * 2), 5))
    fig.suptitle("Benchmark trends over time", fontsize=13, fontweight="bold")

    for runner in runners:
        color = RUNNER_COLORS.get(runner, DEFAULT_COLOR)
        accs = [accuracy(records_for(run, runner)) for run in runs]
        spds = [avg_elapsed(records_for(run, runner)) for run in runs]
        mask_a = [(i, v) for i, v in enumerate(accs) if v is not None]
        mask_s = [(i, v) for i, v in enumerate(spds) if v is not None]
        if mask_a:
            xs, ys = zip(*mask_a)
            axes[0].plot(xs, ys, "o-", label=runner, color=color)
        if mask_s:
            xs, ys = zip(*mask_s)
            axes[1].plot(xs, ys, "o-", label=runner, color=color)

    for ax in axes:
        ax.set_xticks(range(len(timestamps)))
        ax.set_xticklabels(timestamps, rotation=25, ha="right", fontsize=8)
        ax.legend()
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy over time")
    axes[1].set_ylabel("Avg time (s)")
    axes[1].set_title("Avg elapsed over time")

    plt.tight_layout()
    _finish(fig, out_path)


def plot_per_task(runs: list[dict], out_path: str | None) -> None:
    """Scatter: elapsed vs pass/fail per task, coloured by runner."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Per-task elapsed time (● pass, ✕ fail)")

    runners = all_runners(runs)
    for run in runs:
        for runner in runners:
            color = RUNNER_COLORS.get(runner, DEFAULT_COLOR)
            for r in records_for(run, runner):
                marker = "o" if r["pass"] else "x"
                ax.scatter(r["id"], r["elapsed"], marker=marker, color=color,
                           label=runner, alpha=0.7, s=60)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Elapsed (s)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish(fig, out_path)


def plot_elapsed_distribution(runs: list[dict], out_path: str | None) -> None:
    """Box plot of elapsed time distribution per runner, split by source."""
    runners = all_runners(runs)
    sources = all_sources(runs)

    if not sources:
        return

    n_sources = len(sources)
    fig, axes = plt.subplots(1, n_sources, figsize=(max(8, n_sources * 4), 5), sharey=True)
    fig.suptitle("Elapsed time distribution by dataset", fontsize=13, fontweight="bold")
    if n_sources == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        data = []
        labels = []
        for runner in runners:
            all_records: list[dict] = []
            for run in runs:
                all_records += records_for(run, runner, source)
            if all_records:
                data.append([r["elapsed"] for r in all_records])
                labels.append(runner)

        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            for patch, label in zip(bp["boxes"], labels):
                patch.set_facecolor(RUNNER_COLORS.get(label, DEFAULT_COLOR))
                patch.set_alpha(0.7)

        ax.set_title(source)
        ax.set_ylabel("Elapsed (s)" if source == sources[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _finish(fig, out_path)


def plot_ruler_heatmap(runs: list[dict], out_path: str | None) -> None:
    """Heatmap of RULER NIAH pass rate by context length × needle depth per runner."""
    ruler_records: dict[tuple, list[bool]] = {}

    for run in runs:
        for runner, results in run["results"].items():
            for r in results:
                if r.get("source") != "ruler_niah":
                    continue
                m = re.match(r"ruler_niah_(\d+)tok_(\d+)pct_", r["id"])
                if not m:
                    continue
                length = int(m.group(1))
                depth_pct = int(m.group(2))
                key = (runner, length, depth_pct)
                ruler_records.setdefault(key, []).append(r["pass"])

    if not ruler_records:
        print("No RULER results found, skipping heatmap.")
        return

    ruler_runners = sorted({k[0] for k in ruler_records})
    lengths = sorted({k[1] for k in ruler_records})
    depths_pct = sorted({k[2] for k in ruler_records})

    n_runners = len(ruler_runners)
    fig, axes = plt.subplots(1, n_runners, figsize=(6 * n_runners, max(3, len(depths_pct) * 1.2)))
    fig.suptitle("RULER NIAH: pass rate by context length × needle depth",
                 fontsize=13, fontweight="bold")
    if n_runners == 1:
        axes = [axes]

    for ax, runner in zip(axes, ruler_runners):
        matrix = np.full((len(depths_pct), len(lengths)), np.nan)
        for i, depth in enumerate(depths_pct):
            for j, length in enumerate(lengths):
                records = ruler_records.get((runner, length, depth), [])
                if records:
                    matrix[i, j] = sum(records) / len(records) * 100

        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f"{l:,}" for l in lengths], rotation=15, ha="right")
        ax.set_yticks(range(len(depths_pct)))
        ax.set_yticklabels([f"{d}%" for d in depths_pct])
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel("Needle depth")
        ax.set_title(runner)

        for i in range(len(depths_pct)):
            for j in range(len(lengths)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < 30 or val > 85 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=text_color)

        plt.colorbar(im, ax=ax, label="Pass %")

    plt.tight_layout()
    _finish(fig, out_path)


def _finish(fig, out_path: str | None) -> None:
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="result JSON files (default: results/*.json)")
    parser.add_argument("--source", help="filter to one dataset source (e.g. gsm8k, math500)")
    parser.add_argument("--out", help="save to this path (only for a single --chart)")
    parser.add_argument("--out-dir", help="directory to save all charts as PNGs (e.g. reports/)")
    parser.add_argument("--chart",
                        choices=["source", "time", "tasks", "elapsed", "ruler", "all"],
                        default="all",
                        help="which chart(s) to produce (default: all)")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else sorted(Path("results").glob("*.json"))
    paths = [p for p in paths if p.name != ".gitkeep"]
    if not paths:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    runs = load_all(paths, args.source)
    print(f"Loaded {len(runs)} run file(s): {[r['meta']['timestamp'] for r in runs]}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def chart_path(name: str) -> str | None:
        if args.chart != "all" and args.out:
            return args.out
        if out_dir:
            return str(out_dir / f"{name}.png")
        return None

    if args.chart in ("source", "all"):
        plot_by_source(runs, chart_path("accuracy_by_dataset"))
    if args.chart in ("time", "all"):
        plot_over_time(runs, chart_path("trends_over_time"))
    if args.chart in ("tasks", "all"):
        plot_per_task(runs, chart_path("per_task_elapsed"))
    if args.chart in ("elapsed", "all"):
        plot_elapsed_distribution(runs, chart_path("elapsed_distribution"))
    if args.chart in ("ruler", "all"):
        plot_ruler_heatmap(runs, chart_path("ruler_heatmap"))


if __name__ == "__main__":
    main()
