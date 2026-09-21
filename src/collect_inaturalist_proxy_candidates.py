#!/usr/bin/env python3
"""Collect a small, licensed iNaturalist candidate set for exp016 review.

The output is a source-audited candidate pool, not an accepted benchmark.
Reviewers must still reject close portraits and select fixed-view scenes before
any full-frame versus oracle-crop comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


API_URL = "https://api.inaturalist.org/v1/observations"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}
USER_AGENT = "BirdPilot-exp016/1.0 (licensed fixed-view candidate collection)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-count", type=int, default=20)
    parser.add_argument("--candidates-per-class", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=1.1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_labels(path: Path, count: int) -> list[tuple[int, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    labels = sorted((int(index), str(label)) for index, label in raw.items())
    if count <= 0 or count > len(labels):
        raise ValueError(f"class-count must be between 1 and {len(labels)}")
    if count == 1:
        return [labels[0]]
    positions = [round(index * (len(labels) - 1) / (count - 1)) for index in range(count)]
    return [labels[position] for position in positions]


def api_results(label: str, per_page: int) -> list[dict[str, object]]:
    query = urllib.parse.urlencode(
        {
            "taxon_name": label,
            "quality_grade": "research",
            "photos": "true",
            "photo_license": ",".join(sorted(ALLOWED_LICENSES)),
            "per_page": per_page,
            "order_by": "id",
            "order": "desc",
        }
    )
    request = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return list(json.load(response).get("results", []))


def large_photo_url(url: str) -> str:
    for size in ("square", "small", "medium"):
        marker = f"/{size}."
        if marker in url:
            return url.replace(marker, "/large.")
    return url


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())
    with Image.open(path) as image:
        image.verify()


def main() -> None:
    args = parse_args()
    if args.candidates_per_class <= 0 or args.request_delay < 1.0:
        raise ValueError("candidates-per-class must be positive and request-delay must be at least 1 second")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    targets = selected_labels(args.labels, args.class_count)
    for target_number, (class_index, label) in enumerate(targets):
        try:
            observations = api_results(label, max(10, args.candidates_per_class * 3))
        except Exception as error:
            exclusions.append({"class_index": class_index, "model_label": label, "reason": "api_error", "detail": str(error)})
            time.sleep(args.request_delay)
            continue

        accepted = 0
        seen_photos: set[int] = set()
        for observation in observations:
            if accepted >= args.candidates_per_class:
                break
            for photo in observation.get("photos", []):
                photo_id = int(photo["id"])
                license_code = str(photo.get("license_code") or "").lower()
                if photo_id in seen_photos or license_code not in ALLOWED_LICENSES:
                    continue
                seen_photos.add(photo_id)
                image_url = large_photo_url(str(photo["url"]))
                suffix = Path(urllib.parse.urlparse(image_url).path).suffix.lower() or ".jpg"
                filename = f"class_{class_index:03d}_candidate_{accepted:02d}_{photo_id}{suffix}"
                path = args.output_dir / filename
                try:
                    download(image_url, path)
                    with Image.open(path) as image:
                        width, height = image.size
                except Exception as error:
                    if path.exists():
                        path.unlink()
                    exclusions.append(
                        {
                            "class_index": class_index,
                            "model_label": label,
                            "observation_id": observation.get("id"),
                            "photo_id": photo_id,
                            "reason": "download_or_image_error",
                            "detail": str(error),
                        }
                    )
                    continue
                taxon = observation.get("taxon") or {}
                records.append(
                    {
                        "candidate_file": filename,
                        "sha256": sha256(path),
                        "width": width,
                        "height": height,
                        "class_index": class_index,
                        "model_label": label,
                        "observation_id": observation.get("id"),
                        "observation_url": observation.get("uri"),
                        "photo_id": photo_id,
                        "media_url": image_url,
                        "license": license_code,
                        "attribution": photo.get("attribution"),
                        "scientific_name": taxon.get("name"),
                        "source_common_name": taxon.get("preferred_common_name"),
                        "observed_on": observation.get("observed_on"),
                        "review_status": "pending_fixed_view_review",
                    }
                )
                accepted += 1
                break
        if accepted < args.candidates_per_class:
            exclusions.append(
                {
                    "class_index": class_index,
                    "model_label": label,
                    "reason": "insufficient_licensed_candidates",
                    "requested": args.candidates_per_class,
                    "collected": accepted,
                }
            )
        print(
            json.dumps(
                {"class_index": class_index, "model_label": label, "collected": accepted},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if target_number + 1 < len(targets):
            time.sleep(args.request_delay)

    manifest = {
        "schema_version": 1,
        "purpose": "exp016_inaturalist_candidates_pending_fixed_view_review",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": API_URL,
        "allowed_photo_licenses": sorted(ALLOWED_LICENSES),
        "coordinates_retained": False,
        "class_sampling": "evenly_spaced_model_class_indices",
        "requested_class_count": args.class_count,
        "candidates_per_class": args.candidates_per_class,
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
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
