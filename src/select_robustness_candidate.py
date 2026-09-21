#!/usr/bin/env python3
"""Apply BirdPilot's frozen validation-only robustness promotion rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--clean-metrics", type=Path, required=True)
    parser.add_argument("--augmented-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-epochs", type=int, default=10)
    parser.add_argument("--max-clean-loss", type=float, default=0.005)
    parser.add_argument("--min-robust-gain", type=float, default=0.02)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No metric records: {path}")
    epochs = [int(record["epoch"]) for record in records]
    if epochs != list(range(1, max(epochs) + 1)):
        raise ValueError(f"Epochs must be consecutive from 1: {path}")
    return records


def baseline_scores(path: Path) -> tuple[float, float]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("source_partition") != "valid":
        raise ValueError("Baseline must be validation-only")
    conditions = summary["conditions"]
    clean = float(conditions["clean"]["top1_accuracy"])
    robust = [float(value["top1_accuracy"]) for key, value in conditions.items() if key != "clean"]
    if len(robust) != 15:
        raise ValueError(f"Expected 15 robustness conditions, found {len(robust)}")
    return clean, sum(robust) / len(robust)


def select(records: list[dict[str, object]], minimum_clean: float) -> dict[str, object] | None:
    eligible = [
        record
        for record in records
        if float(record["clean_validation_accuracy"]) >= minimum_clean
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            float(record["robust_condition_mean_accuracy"]),
            float(record["clean_validation_accuracy"]),
            -int(record["epoch"]),
        ),
    )


def compact(record: dict[str, object] | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "epoch": int(record["epoch"]),
        "clean_accuracy": float(record["clean_validation_accuracy"]),
        "robust_condition_mean_accuracy": float(record["robust_condition_mean_accuracy"]),
    }


def main() -> None:
    args = parse_args()
    if args.required_epochs <= 0 or args.max_clean_loss < 0 or args.min_robust_gain < 0:
        raise ValueError("Thresholds and required epochs must be nonnegative/positive")
    baseline_clean, baseline_robust = baseline_scores(args.baseline_summary)
    clean_records = read_jsonl(args.clean_metrics)
    augmented_records = read_jsonl(args.augmented_metrics)
    minimum_clean = baseline_clean - args.max_clean_loss
    clean_best = select(clean_records, minimum_clean)
    augmented_best = select(augmented_records, minimum_clean)
    complete = len(clean_records) >= args.required_epochs and len(augmented_records) >= args.required_epochs
    reasons: list[str] = []
    if not complete:
        reasons.append("planned_epochs_incomplete")
    if clean_best is None:
        reasons.append("clean_arm_has_no_eligible_checkpoint")
    if augmented_best is None:
        reasons.append("augmented_arm_has_no_eligible_checkpoint")
    if clean_best is not None and augmented_best is not None:
        augmented_clean = float(augmented_best["clean_validation_accuracy"])
        augmented_robust = float(augmented_best["robust_condition_mean_accuracy"])
        clean_clean = float(clean_best["clean_validation_accuracy"])
        clean_robust = float(clean_best["robust_condition_mean_accuracy"])
        if baseline_clean - augmented_clean > args.max_clean_loss:
            reasons.append("augmented_clean_loss_vs_baseline_exceeds_limit")
        if clean_clean - augmented_clean > args.max_clean_loss:
            reasons.append("augmented_clean_loss_vs_clean_arm_exceeds_limit")
        if augmented_robust - baseline_robust < args.min_robust_gain:
            reasons.append("augmented_robust_gain_vs_baseline_below_gate")
        if augmented_robust - clean_robust < args.min_robust_gain:
            reasons.append("augmented_robust_gain_vs_clean_arm_below_gate")
    report = {
        "schema_version": 1,
        "source_partition": "valid",
        "thresholds": {
            "required_epochs": args.required_epochs,
            "max_clean_loss": args.max_clean_loss,
            "min_robust_gain": args.min_robust_gain,
        },
        "baseline": {
            "clean_accuracy": baseline_clean,
            "robust_condition_mean_accuracy": baseline_robust,
        },
        "clean_arm": {"epochs_present": len(clean_records), "selected": compact(clean_best)},
        "augmented_arm": {"epochs_present": len(augmented_records), "selected": compact(augmented_best)},
        "promote_augmented": not reasons,
        "reasons": reasons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
