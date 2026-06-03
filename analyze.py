#!/usr/bin/env python3
"""
Load benchmark result JSON files and produce comparison charts.

Usage:
    python analyze.py                          # all files in results/
    python analyze.py results/20260603_*.json  # specific files
    python analyze.py --source gsm8k           # filter by dataset
    python analyze.py --out report.png         # save instead of show
"""
import argparse
import json
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
    # support both old format (bare dict) and new envelope format
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

    # merge all runs — average across multiple run files
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

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel("Task ID")
    ax.set_ylabel("Elapsed (s)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish(fig, out_path)


def _finish(fig, out_path: str | None) -> None:
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="result JSON files (default: results/*.json)")
    parser.add_argument("--source", help="filter to one dataset source (e.g. gsm8k, math)")
    parser.add_argument("--out", help="save figure to this path instead of showing it")
    parser.add_argument("--chart", choices=["source", "time", "tasks", "all"], default="all",
                        help="which chart(s) to produce (default: all)")
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files else sorted(Path("results").glob("*.json"))
    if not paths:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    runs = load_all(paths, args.source)
    print(f"Loaded {len(runs)} run file(s): {[r['meta']['timestamp'] for r in runs]}")

    if args.chart in ("source", "all"):
        plot_by_source(runs, args.out if args.chart != "all" else None)
    if args.chart in ("time", "all"):
        plot_over_time(runs, args.out if args.chart != "all" else None)
    if args.chart in ("tasks", "all"):
        plot_per_task(runs, args.out if args.chart != "all" else None)


if __name__ == "__main__":
    main()
