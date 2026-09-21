#!/usr/bin/env python3
"""Source-clustered paired bootstrap for two robustness prediction files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def load(path: Path) -> dict[tuple[str, str, str], bool]:
    rows: dict[tuple[str, str, str], bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["family"] == "clean":
            continue
        key = (str(row["source_relative_path"]), str(row["family"]), str(row["level"]))
        if key in rows:
            raise ValueError(f"Duplicate prediction key in {path}: {key}")
        rows[key] = bool(row["correct"])
    return rows


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline = load(args.baseline_predictions)
    candidate = load(args.candidate_predictions)
    if baseline.keys() != candidate.keys():
        raise ValueError("Prediction files do not contain identical source/condition keys")
    by_source: dict[str, list[float]] = defaultdict(list)
    for key in sorted(baseline):
        source = key[0]
        by_source[source].append(float(candidate[key]) - float(baseline[key]))
    if any(len(values) != 15 for values in by_source.values()):
        raise ValueError("Every source must contain exactly 15 degraded conditions")
    source_differences = np.asarray([np.mean(by_source[source]) for source in sorted(by_source)])
    rng = np.random.default_rng(args.seed)
    draws = np.empty(args.iterations, dtype=np.float64)
    for index in range(args.iterations):
        sampled = rng.integers(0, len(source_differences), size=len(source_differences))
        draws[index] = source_differences[sampled].mean()
    report = {
        "schema_version": 1,
        "pairing": "source_relative_path_and_condition",
        "cluster": "source_relative_path",
        "source_count": len(source_differences),
        "conditions_per_source": 15,
        "iterations": args.iterations,
        "seed": args.seed,
        "candidate_minus_baseline_mean": float(source_differences.mean()),
        "percentile_95_interval": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
