#!/usr/bin/env python3
"""Build the ignored, deterministic 24-scene YOLOX board acceptance bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose_records(inat_scene: list[dict], inat_class: list[dict], empty_scene: list[dict]) -> list[dict]:
    best_class: dict[str, dict] = {}
    for row in inat_class:
        name = row["source_candidate_file"]
        if name not in best_class or row["detector_confidence"] > best_class[name]["detector_confidence"]:
            best_class[name] = row
    detected = {row["candidate_file"]: row for row in inat_scene if row["bird_detection_count"]}
    correct = [row for name, row in detected.items() if best_class[name]["matches_expected"]]
    correct.sort(key=lambda row: row["detections"][0]["area_fraction"])
    quantile_indices = [0, round((len(correct)-1)*.2), round((len(correct)-1)*.4),
                        round((len(correct)-1)*.6), round((len(correct)-1)*.8), len(correct)-1]
    selected: list[dict] = []
    used: set[str] = set()

    def add(rows: list[dict], category: str, limit: int) -> None:
        for row in rows:
            name = row["candidate_file"]
            if name not in used and len([x for x in selected if x["acceptance_category"] == category]) < limit:
                selected.append({**row, "acceptance_category": category})
                used.add(name)

    add([correct[index] for index in quantile_indices], "correct_area_quantile", 6)
    misses = sorted((row for row in inat_scene if not row["bird_detection_count"]), key=lambda r: r["candidate_file"])
    add(misses, "detector_miss", 4)
    wrong = sorted(
        (detected[name] for name, row in best_class.items() if not row["matches_expected"]),
        key=lambda row: (-row["detections"][0]["confidence"], row["candidate_file"]),
    )
    add(wrong, "detected_classification_error", 4)
    empties = sorted((row for row in empty_scene if not row["bird_detection_count"]), key=lambda r: r["candidate_file"])
    add(empties, "empty_no_trigger", 4)
    near = sorted(
        (row for row in inat_scene + empty_scene if row["bird_detection_count"]),
        key=lambda row: (abs(row["detections"][0]["confidence"] - 0.25), row["candidate_file"]),
    )
    add(near, "threshold_or_label_review", 4)
    remaining = sorted(inat_scene + empty_scene, key=lambda row: row["candidate_file"])
    add(remaining, "lexicographic_fill", 24)
    return selected[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inat-root", type=Path, required=True)
    parser.add_argument("--inat-manifest", type=Path, required=True)
    parser.add_argument("--inat-scenes", type=Path, required=True)
    parser.add_argument("--inat-classifier", type=Path, required=True)
    parser.add_argument("--empty-root", type=Path, required=True)
    parser.add_argument("--empty-manifest", type=Path, required=True)
    parser.add_argument("--empty-scenes", type=Path, required=True)
    parser.add_argument("--detector-rknn", type=Path, required=True)
    parser.add_argument("--classifier-rknn", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--onnx-reference", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    inat_scene = jsonl(args.inat_scenes)
    empty_scene_all = jsonl(args.empty_scenes)
    empty_scene = [row for row in empty_scene_all if row.get("source_category") == "empty"]
    class_rows = jsonl(args.inat_classifier)
    selected = choose_records(inat_scene, class_rows, empty_scene)
    if len(selected) != 24:
        raise RuntimeError(f"Expected 24 acceptance scenes, got {len(selected)}")

    args.output_dir.mkdir(parents=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    inat_license = {row["candidate_file"]: row for row in json.loads(args.inat_manifest.read_text())["records"]}
    empty_license = {row["candidate_file"]: row for row in json.loads(args.empty_manifest.read_text())["records"]}
    licenses = []
    for row in selected:
        name = row["candidate_file"]
        source_root = args.empty_root if name in empty_license else args.inat_root
        source = source_root / name
        if sha256(source) != row["source_sha256"]:
            raise ValueError(f"Input hash mismatch: {source}")
        shutil.copy2(source, image_dir / name)
        license_row = (empty_license if name in empty_license else inat_license)[name]
        licenses.append({"file": name, "acceptance_category": row["acceptance_category"], **license_row})

    copies = {
        args.detector_rknn: args.output_dir / "yolox_nano_fp16.rknn",
        args.classifier_rknn: args.output_dir / "bird_classifier_fp16.rknn",
        args.labels: args.output_dir / "labels.json",
        args.conversion_manifest: args.output_dir / "detector_conversion_manifest.json",
        Path(__file__).with_name("infer_yolox_rknn.py"): args.output_dir / "infer_yolox_rknn.py",
        Path(__file__).resolve().parents[2] / "src" / "yolox_contract.py": args.output_dir / "yolox_contract.py",
    }
    for source, target in copies.items():
        shutil.copy2(source, target)
    reference = json.loads(args.onnx_reference.read_text())
    selected_names = {row["candidate_file"] for row in selected}
    reference["records"] = [
        row for row in reference["records"] if row["file"] in selected_names
    ]
    (args.output_dir / "onnx_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "LICENSE_MANIFEST.json").write_text(
        json.dumps({"schema_version": 1, "records": licenses}, ensure_ascii=False, indent=2) + "\n"
    )
    readme = """# BirdPilot YOLOX-Nano RK3588 acceptance bundle

Verify: `sha256sum -c SHA256SUMS`

Run on the board from this directory:

`python3 infer_yolox_rknn.py --model yolox_nano_fp16.rknn --classifier-model bird_classifier_fp16.rknn --labels labels.json --image-dir images --output board_results.json`

Upload example:

`scp -r birdpilot_exp016_board_bundle USER@BOARD:/home/USER/birdpilot/`

The included images are licensed proxy data. They validate deployment parity only; they are not camera or outdoor evidence.
"""
    (args.output_dir / "README.md").write_text(readme)
    files = sorted(path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(json.dumps({"output": str(args.output_dir), "image_count": len(selected),
                      "categories": {name: sum(r["acceptance_category"] == name for r in selected)
                                     for name in sorted({r["acceptance_category"] for r in selected})}}, indent=2))


if __name__ == "__main__":
    main()
