#!/usr/bin/env python3
"""Summarize BirdPilot's frozen three-seed paired robustness decision."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", action="append", required=True, help="SEED=selection.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clean-loss", type=float, default=0.005)
    parser.add_argument("--min-robust-gain", type=float, default=0.02)
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parsed: list[tuple[int, Path]] = []
    for item in args.selection:
        seed_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError(f"Selection must use SEED=PATH: {item}")
        parsed.append((int(seed_text), Path(path_text)))
    if len(parsed) != 3 or len({seed for seed, _ in parsed}) != 3:
        raise ValueError("Exactly three unique seed selections are required")

    seeds: list[dict[str, object]] = []
    reasons: list[str] = []
    baseline_clean: float | None = None
    baseline_robust: float | None = None
    for seed, path in sorted(parsed):
        report = json.loads(path.read_text(encoding="utf-8"))
        base = report["baseline"]
        current_baseline_clean = float(base["clean_accuracy"])
        current_baseline_robust = float(base["robust_condition_mean_accuracy"])
        if baseline_clean is None:
            baseline_clean = current_baseline_clean
            baseline_robust = current_baseline_robust
        elif baseline_clean != current_baseline_clean or baseline_robust != current_baseline_robust:
            raise ValueError("Seed selections do not use the same frozen baseline")
        clean = report["clean_arm"]["selected"]
        augmented = report["augmented_arm"]["selected"]
        if clean is None or augmented is None:
            reasons.append(f"seed_{seed}_missing_eligible_checkpoint")
            continue
        clean_accuracy = float(clean["clean_accuracy"])
        clean_robust = float(clean["robust_condition_mean_accuracy"])
        augmented_accuracy = float(augmented["clean_accuracy"])
        augmented_robust = float(augmented["robust_condition_mean_accuracy"])
        seeds.append(
            {
                "seed": seed,
                "clean_arm": clean,
                "augmented_arm": augmented,
                "augmented_minus_baseline_robust": augmented_robust - baseline_robust,
                "augmented_minus_clean_arm_robust": augmented_robust - clean_robust,
                "augmented_clean_loss_vs_baseline": baseline_clean - augmented_accuracy,
                "augmented_clean_loss_vs_clean_arm": clean_accuracy - augmented_accuracy,
            }
        )

    summary: dict[str, object] = {}
    if len(seeds) == 3:
        metrics = {
            key: [float(seed[key]) for seed in seeds]
            for key in (
                "augmented_minus_baseline_robust",
                "augmented_minus_clean_arm_robust",
                "augmented_clean_loss_vs_baseline",
                "augmented_clean_loss_vs_clean_arm",
            )
        }
        summary = {key: mean_std(values) for key, values in metrics.items()}
        if any(value <= 0 for value in metrics["augmented_minus_clean_arm_robust"]):
            reasons.append("augmented_minus_clean_arm_not_positive_in_every_seed")
        if statistics.mean(metrics["augmented_minus_baseline_robust"]) < args.min_robust_gain:
            reasons.append("mean_robust_gain_vs_baseline_below_gate")
        if statistics.mean(metrics["augmented_minus_clean_arm_robust"]) < args.min_robust_gain:
            reasons.append("mean_robust_gain_vs_clean_arm_below_gate")
        if statistics.mean(metrics["augmented_clean_loss_vs_baseline"]) > args.max_clean_loss:
            reasons.append("mean_clean_loss_vs_baseline_exceeds_limit")
        if statistics.mean(metrics["augmented_clean_loss_vs_clean_arm"]) > args.max_clean_loss:
            reasons.append("mean_clean_loss_vs_clean_arm_exceeds_limit")

    output = {
        "schema_version": 1,
        "source_partition": "valid",
        "thresholds": {
            "seed_count": 3,
            "max_mean_clean_loss": args.max_clean_loss,
            "min_mean_robust_gain": args.min_robust_gain,
            "require_positive_augmented_minus_clean_in_every_seed": True,
        },
        "baseline": {
            "clean_accuracy": baseline_clean,
            "robust_condition_mean_accuracy": baseline_robust,
        },
        "seeds": seeds,
        "summary": summary,
        "repeated_gate_passed": not reasons,
        "reasons": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
