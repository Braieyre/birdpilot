#!/usr/bin/env python3
"""Build hash-verified Sunlogin transfer archives for BirdPilot's L20 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--refresh-source", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, path: Path, arcname: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    archive.add(path, arcname=arcname, recursive=False)


def main() -> None:
    args = parse_args()
    if args.refresh_source:
        required = {
            "birdpilot_train_valid_data.tar",
            "birdpilot_training_assets.tar",
            "PAYLOAD_MANIFEST.json",
        }
        missing = [name for name in required if not (args.output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Cannot refresh source; missing existing payload files: {missing}")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True)
    if dirty:
        raise RuntimeError("Tracked working tree must be clean before packaging")

    source_tar = args.output_dir / "birdpilot_source.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(source_tar), "HEAD"], cwd=PROJECT_ROOT, check=True)
    with tarfile.open(source_tar, "a") as archive:
        payload = (commit + "\n").encode()
        info = tarfile.TarInfo("SOURCE_VERSION.txt")
        info.size = len(payload)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(payload))

    if args.refresh_source:
        payload_manifest_path = args.output_dir / "PAYLOAD_MANIFEST.json"
        payload_manifest = json.loads(payload_manifest_path.read_text(encoding="utf-8"))
        data_tar = args.output_dir / "birdpilot_train_valid_data.tar"
        assets_tar = args.output_dir / "birdpilot_training_assets.tar"
        archives = [source_tar, data_tar, assets_tar]
        checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in archives)
        (args.output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        payload_manifest["source_commit"] = commit
        payload_manifest["archives"] = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in archives
        ]
        payload_manifest_path.write_text(
            json.dumps(payload_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload_manifest, ensure_ascii=False, indent=2))
        return

    csv_path = PROJECT_ROOT / "data/birds/birds.csv"
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    requested = sorted({row["filepaths"] for row in rows if row.get("data set") in {"train", "valid"}})
    selected = [relative for relative in requested if (PROJECT_ROOT / "data/birds" / relative).is_file()]
    missing = [relative for relative in requested if not (PROJECT_ROOT / "data/birds" / relative).is_file()]
    data_tar = args.output_dir / "birdpilot_train_valid_data.tar"
    with tarfile.open(data_tar, "w") as archive:
        add_file(archive, csv_path, "data/birds/birds.csv")
        for relative in selected:
            add_file(archive, PROJECT_ROOT / "data/birds" / relative, f"data/birds/{relative}")

    assets = [
        "models/exp005_mobilenetv3_full_best.pt",
        "outputs/exp005_mobilenetv3_full_idx_to_label.json",
        "outputs/degradation_valid_full_v2/manifest.json",
        "outputs/exp013_m0_robustness_full_v2/summary.json",
        "outputs/exp013_m0_robustness_full_v2/predictions.jsonl",
    ]
    manifest = json.loads((PROJECT_ROOT / assets[2]).read_text(encoding="utf-8"))
    assets.extend(f"outputs/degradation_valid_full_v2/{row['output_path']}" for row in manifest["records"])
    assets_tar = args.output_dir / "birdpilot_training_assets.tar"
    with tarfile.open(assets_tar, "w") as archive:
        for relative in assets:
            add_file(archive, PROJECT_ROOT / relative, relative)

    archives = [source_tar, data_tar, assets_tar]
    checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in archives)
    (args.output_dir / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    payload_manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "data_partitions": ["train", "valid"],
        "test_images_included": False,
        "csv_selected_path_count": len(selected),
        "missing_train_valid_path_count": len(missing),
        "missing_train_valid_paths": missing,
        "degraded_variant_count": len(manifest["records"]),
        "archives": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in archives],
    }
    (args.output_dir / "PAYLOAD_MANIFEST.json").write_text(
        json.dumps(payload_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
