#!/usr/bin/env python3
"""
Run the benchmark across all configured runners and print a comparison table.

Usage:
    python run.py                              # local JSON tasks, all runners
    python run.py --runner strap               # single runner
    python run.py --type reasoning             # single task type
    python run.py --tasks tasks/memory/001.json

    # Standard datasets
    python run.py --dataset gsm8k --max-samples 50
    python run.py --dataset math --max-samples 20 --max-level 3
    python run.py --dataset ruler --context-lengths 2000,4000,8000 --needle-depths 0.1,0.5,0.9
    python run.py --dataset gsm8k --dataset math --max-samples 30
"""
import argparse
import glob
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from datasets import disable_progress_bar
from evaluate import score
from runners.opencode import OpenCodeRunner
from runners.qwen import QwenRunner
from runners.strap import StrapRunner

disable_progress_bar()

TASK_TYPES = ["memory", "reasoning", "context"]

RUNNERS = {
    "base_qwen": lambda: QwenRunner(),
    "opencode":  lambda: OpenCodeRunner(),
    "strap":     lambda: StrapRunner(),
}


# ── task loading ──────────────────────────────────────────────────────────────

def load_local_tasks(pattern: str | None, task_type: str | None) -> list[dict]:
    files = glob.glob(pattern or "tasks/**/*.json", recursive=True)
    tasks = []
    for f in sorted(files):
        task = json.loads(Path(f).read_text())
        if task_type and task.get("type") != task_type:
            continue
        tasks.append(task)
    return tasks


def load_dataset_tasks(names: list[str], args: argparse.Namespace) -> list[dict]:
    from adapters import load_gsm8k, load_math, generate_ruler

    tasks: list[dict] = []
    for name in names:
        if name == "gsm8k":
            print(f"Loading GSM8K (max {args.max_samples or 'all'})...")
            tasks += load_gsm8k(max_samples=args.max_samples)

        elif name == "math":
            print(f"Loading MATH-500 (max {args.max_samples or 'all'}, level ≤{args.max_level})...")
            tasks += load_math(max_samples=args.max_samples, max_level=args.max_level)

        elif name == "ruler":
            lengths = [int(x) for x in args.context_lengths.split(",")]
            depths  = [float(x) for x in args.needle_depths.split(",")]
            print(f"Generating RULER NIAH tasks (lengths={lengths}, depths={depths})...")
            tasks += generate_ruler(
                context_lengths=lengths,
                needle_depths=depths,
                n_per_config=args.ruler_n,
            )
        else:
            print(f"Unknown dataset: {name!r}", file=sys.stderr)
            sys.exit(1)

    return tasks


# ── runner execution ──────────────────────────────────────────────────────────

def run_runner(name: str, runner_factory, tasks: list[dict]) -> list[dict]:
    runner = runner_factory()

    if hasattr(runner, "connect"):
        print(f"  connecting {name}...")
        runner.connect()

    results = []
    for task in tasks:
        print(f"  [{name}] {task['id']} ... ", end="", flush=True)
        runner.reset()
        t0 = time.time()
        try:
            response = runner.run(task)
            result = score(task, response)
        except Exception as e:
            result = {"pass": False, "method": "error", "error": str(e), "got": ""}
        elapsed = time.time() - t0
        result.update({"id": task["id"], "type": task["type"], "elapsed": round(elapsed, 2)})
        # tag source dataset if present
        if "source" in task:
            result["source"] = task["source"]
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(status, f"({elapsed:.1f}s)")
        if not result["pass"]:
            expected = task.get("expected", "?")
            got = result.get("got", "").strip()
            print(f"    expected : {expected}")
            if result.get("method") == "error":
                print(f"      error  : {result.get('error', '?')}")
            if result.get("method") == "llm_judge":
                print(f"     verdict : {result.get('verdict', '?')}")
            for line in got.splitlines():
                print(f"             | {line}")

    runner.close()
    return results


# ── output ────────────────────────────────────────────────────────────────────

def print_table(all_results: dict[str, list[dict]]) -> None:
    runner_names = list(all_results.keys())
    col = 18

    sources = sorted({r.get("source", "local") for rs in all_results.values() for r in rs})
    rows = TASK_TYPES + [f"  {s}" for s in sources] + ["overall"]

    width = col + col * len(runner_names)
    print("\n" + "=" * width)
    print(f"{'':>{col}}" + "".join(f"{n:>{col}}" for n in runner_names))
    print("-" * width)

    for label in rows:
        is_source = label.startswith("  ")
        key = label.strip()

        row = f"{label:<{col}}"
        for name in runner_names:
            rs = all_results[name]
            if is_source:
                subset = [r for r in rs if r.get("source") == key]
            elif key == "overall":
                subset = rs
            else:
                subset = [r for r in rs if r["type"] == key]

            if not subset:
                row += f"{'—':>{col}}"
            else:
                pct = sum(r["pass"] for r in subset) / len(subset) * 100
                avg_s = sum(r["elapsed"] for r in subset) / len(subset)
                row += f"{pct:>{col - 5}.0f}%  {avg_s:5.1f}s"
        print(row)

    print("=" * width)


def save_results(all_results: dict[str, list[dict]], args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"{timestamp}.json"

    sources = sorted({r.get("source", "local") for rs in all_results.values() for r in rs})
    envelope = {
        "meta": {
            "timestamp": timestamp,
            "datasets": args.datasets or ["local"],
            "sources": sources,
            "max_samples": args.max_samples,
            "max_level": getattr(args, "max_level", None),
        },
        "results": all_results,
    }
    out.write_text(json.dumps(envelope, indent=2))
    print(f"\nResults saved to {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=list(RUNNERS), help="run only this runner")
    parser.add_argument("--type", choices=TASK_TYPES, dest="task_type", help="filter local tasks by type")
    parser.add_argument("--tasks", help="glob pattern for local task files")

    # dataset flags
    parser.add_argument("--dataset", action="append", dest="datasets",
                        choices=["gsm8k", "math", "ruler"],
                        help="load a standard dataset (repeatable)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="cap samples per dataset (default: all)")
    parser.add_argument("--max-level", type=int, default=5,
                        help="MATH max difficulty level 1-5 (default: 5)")
    parser.add_argument("--context-lengths", default="2000,4000,8000",
                        help="RULER context lengths in tokens, comma-separated")
    parser.add_argument("--needle-depths", default="0.1,0.5,0.9",
                        help="RULER needle positions 0.0-1.0, comma-separated")
    parser.add_argument("--ruler-n", type=int, default=3,
                        help="RULER tasks per (length, depth) config (default: 3)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging verbosity (default: WARNING)")

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(name)s %(levelname)s %(message)s")

    # load tasks
    if args.datasets:
        tasks = load_dataset_tasks(args.datasets, args)
    else:
        tasks = load_local_tasks(args.tasks, args.task_type)

    if not tasks:
        print("No tasks found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(tasks)} task(s).")

    active_runners = {args.runner: RUNNERS[args.runner]} if args.runner else RUNNERS

    all_results: dict[str, list[dict]] = {}
    for name, factory in active_runners.items():
        print(f"\nRunning {name}...")
        all_results[name] = run_runner(name, factory, tasks)

    print_table(all_results)
    save_results(all_results, args)


if __name__ == "__main__":
    main()
