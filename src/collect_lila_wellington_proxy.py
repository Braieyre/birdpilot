#!/usr/bin/env python3
"""Collect a small fixed-camera bird/empty proxy set from LILA Wellington."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


DATASET_PAGE = "https://lila.science/datasets/wellingtoncameratraps"
IMAGE_BASE = "https://storage.googleapis.com/public-datasets-lila/wellington-unzipped/images/"
LICENSE = "CDLA-Permissive-1.0"
LICENSE_URL = "https://cdla.dev/permissive-1-0/"
USER_AGENT = "BirdPilot-exp016/1.0 (bounded LILA proxy collection)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bird-count", type=int, default=20)
    parser.add_argument("--empty-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_diverse(
    images: list[dict[str, object]], count: int, rng: random.Random
) -> list[dict[str, object]]:
    candidates = list(images)
    rng.shuffle(candidates)
    selected: list[dict[str, object]] = []
    used_locations: set[str] = set()
    used_sequences: set[str] = set()
    for require_new_location in (True, False):
        for image in candidates:
            location = str(image.get("location"))
            sequence = str(image.get("seq_id"))
            if sequence in used_sequences:
                continue
            if require_new_location and location in used_locations:
                continue
            selected.append(image)
            used_locations.add(location)
            used_sequences.add(sequence)
            if len(selected) == count:
                return selected
    raise ValueError(f"Only {len(selected)} diverse images available for requested count {count}")


def download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        output.write_bytes(response.read())
    with Image.open(output) as image:
        image.verify()


def main() -> None:
    args = parse_args()
    if args.bird_count <= 0 or args.empty_count <= 0 or args.workers <= 0:
        raise ValueError("bird-count, empty-count, and workers must be positive")
    for path in (args.metadata, args.metadata_archive):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    raw = json.loads(args.metadata.read_text(encoding="utf-8"))
    category_name = {int(row["id"]): str(row["name"]) for row in raw["categories"]}
    image_by_id = {str(row["id"]): row for row in raw["images"]}
    grouped: dict[str, list[dict[str, object]]] = {"bird": [], "empty": []}
    for annotation in raw["annotations"]:
        name = category_name[int(annotation["category_id"])]
        if name in grouped:
            grouped[name].append(image_by_id[str(annotation["image_id"])])

    rng = random.Random(args.seed)
    selected = {
        "bird": select_diverse(grouped["bird"], args.bird_count, rng),
        "empty": select_diverse(grouped["empty"], args.empty_count, rng),
    }
    args.output_dir.mkdir(parents=True)
    jobs: list[tuple[str, int, dict[str, object]]] = []
    for category in ("bird", "empty"):
        for index, image in enumerate(selected[category]):
            jobs.append((category, index, image))

    def collect(job: tuple[str, int, dict[str, object]]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        category, index, image = job
        source_name = str(image["file_name"])
        suffix = Path(source_name).suffix.lower() or ".jpg"
        filename = f"{category}_{index:02d}_{image['id']}{suffix}"
        url = urllib.parse.urljoin(IMAGE_BASE, urllib.parse.quote(source_name))
        output = args.output_dir / filename
        try:
            download(url, output)
            with Image.open(output) as opened:
                width, height = opened.size
        except Exception as error:
            if output.exists():
                output.unlink()
            return None, {
                "dataset_image_id": image["id"],
                "source_category": category,
                "reason": "download_or_image_error",
                "detail": str(error),
            }
        return {
            "candidate_file": filename,
            "sha256": sha256(output),
            "width": width,
            "height": height,
            "source_category": category,
            "dataset_image_id": image["id"],
            "source_file_name": source_name,
            "media_url": url,
            "license": LICENSE,
            "species_accuracy_eligible": False,
            "annotation_scope": "sequence_level_may_not_describe_every_frame",
            "review_status": "pending_fixed_view_review",
        }, None

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for completed, (category, index, _) in zip(pool.map(collect, jobs), jobs):
            record, exclusion = completed
            if record is not None:
                records.append(record)
            if exclusion is not None:
                exclusions.append(exclusion)
            print(
                json.dumps(
                    {
                        "source_category": category,
                        "processed": index + 1,
                        "requested": len(selected[category]),
                        "accepted": record is not None,
                    }
                ),
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "purpose": "exp016_lila_wellington_fixed_camera_candidates",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Wellington Camera Traps",
        "dataset_page": DATASET_PAGE,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "required_citation": "Anton, Hartley, Geldenhuis, and Wittmer (2018), Journal of Urban Ecology 4(1)",
        "metadata_archive_sha256": sha256(args.metadata_archive),
        "coordinates_retained": False,
        "selection": "seeded_one_frame_per_sequence_preferring_unique_camera_locations",
        "seed": args.seed,
        "record_count": len(records),
        "records": records,
        "exclusions": exclusions,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "record_count": len(records),
                "exclusion_count": len(exclusions),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
