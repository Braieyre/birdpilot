#!/usr/bin/env python3
"""Create a deterministic, training-only RKNN calibration image list."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help="Optional new directory of no-space symlinks for tools that cannot parse spaces in paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if not args.train_dir.is_dir():
        raise NotADirectoryError(f"Training directory does not exist: {args.train_dir}")

    images = sorted(
        path.resolve()
        for path in args.train_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < args.count:
        raise ValueError(f"Need {args.count} training images, found {len(images)}")

    selected = random.Random(args.seed).sample(images, args.count)
    calibration_paths = selected
    manifest_entries = None
    if args.staging_dir:
        if args.staging_dir.exists():
            raise FileExistsError(
                f"Staging directory already exists; choose a new versioned directory: {args.staging_dir}"
            )
        args.staging_dir.mkdir(parents=True)
        calibration_paths = []
        manifest_entries = []
        for index, source_path in enumerate(selected):
            staged_path = args.staging_dir / f"cal_{index:04d}{source_path.suffix.lower()}"
            staged_path.symlink_to(source_path)
            calibration_paths.append(staged_path.absolute())
            manifest_entries.append(
                {"staged_path": str(staged_path.absolute()), "source_path": str(source_path)}
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(map(str, calibration_paths)) + "\n", encoding="utf-8")
    if manifest_entries is not None:
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(
                {"seed": args.seed, "count": args.count, "entries": manifest_entries},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote source mapping manifest to: {manifest_path}")
    print(f"Wrote {len(calibration_paths)} training-only calibration paths to: {args.output}")


if __name__ == "__main__":
    main()
