"""Build a small, deterministic BirdPilot onboarding dataset package.

The package is for repository onboarding and degradation-pipeline checks. It is
not a substitute for the full Birds-525 dataset and must not be used to report
the formal 524-class accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "birds"
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "team-share" / "BirdPilot_starter_data_v1"

DEFAULT_CLASSES = (
    "BALD EAGLE",
    "BARN OWL",
    "AMERICAN FLAMINGO",
    "EMPEROR PENGUIN",
    "PEACOCK",
    "PUFFIN",
    "RED TAILED HAWK",
    "SNOWY OWL",
)

SAMPLES_PER_SPLIT = {"train": 20, "valid": 5, "test": 5}
RANDOM_SEED = 20260901


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_deterministic_zip(source: Path) -> Path:
    archive = source.with_suffix(".zip")
    fixed_timestamp = (2026, 9, 1, 0, 0, 0)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=fixed_timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes(), compresslevel=9)
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    csv_path = dataset_root / "birds.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    frame = pd.read_csv(csv_path)
    required = {"labels", "filepaths", "data set"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")

    missing_classes = set(DEFAULT_CLASSES) - set(frame["labels"].unique())
    if missing_classes:
        raise ValueError(f"Dataset is missing classes: {sorted(missing_classes)}")

    selected_parts = []
    for split, sample_count in SAMPLES_PER_SPLIT.items():
        for class_index, label in enumerate(DEFAULT_CLASSES):
            candidates = frame[
                (frame["data set"] == split) & (frame["labels"] == label)
            ].copy()
            if len(candidates) < sample_count:
                raise ValueError(
                    f"Not enough {split} samples for {label}: "
                    f"need {sample_count}, found {len(candidates)}"
                )
            selected_parts.append(
                candidates.sample(
                    n=sample_count,
                    random_state=RANDOM_SEED + class_index,
                )
            )

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["data set", "labels", "filepaths"])
    records = []

    for _, row in selected.iterrows():
        source_rel = Path(row["filepaths"])
        source = dataset_root / source_rel
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")

        split = row["data set"]
        label = row["labels"]
        destination_rel = Path(split) / label / source.name
        destination = output / destination_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "labels": label,
                "filepaths": destination_rel.as_posix(),
                "data set": split,
                "source_filepaths": source_rel.as_posix(),
                "sha256": sha256(destination),
            }
        )

    manifest = pd.DataFrame(records).sort_values(
        ["data set", "labels", "filepaths"]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "manifest.csv", index=False)
    (output / "labels.txt").write_text(
        "\n".join(DEFAULT_CLASSES) + "\n", encoding="utf-8"
    )
    (output / "README.txt").write_text(
        "BirdPilot starter data v1\n\n"
        "Purpose: onboarding, image inspection, and degradation-pipeline checks.\n"
        "Not for reporting the formal 524-class model accuracy.\n\n"
        "Contents: 8 classes; 20 train, 5 valid, and 5 test images per class; "
        "240 images total.\n"
        "Keep train/valid/test separate. Never train on test images.\n"
        "Each manifest row records its original dataset path and SHA-256.\n",
        encoding="utf-8",
    )

    archive = make_deterministic_zip(output)
    print(f"Created: {output}")
    print(f"Images: {len(manifest)}")
    print(manifest.groupby(["data set", "labels"]).size())
    print(f"Archive: {archive}")
    print(f"Archive SHA-256: {sha256(archive)}")


if __name__ == "__main__":
    main()
