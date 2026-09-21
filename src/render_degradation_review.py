#!/usr/bin/env python3
"""Render all degradation conditions into paged contact sheets for visual QA."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=4)
    return parser.parse_args()


def make_tile(image_path: Path, caption: str, tile_size: int = 128) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((tile_size, tile_size - 22))
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    tile.paste(image, ((tile_size - image.width) // 2, 0))
    ImageDraw.Draw(tile).text((3, tile_size - 20), caption[:20], fill="black")
    return tile


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records_by_source: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in manifest["records"]:
        records_by_source[int(record["source_index"])].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(manifest["families"])}
    levels = {name: index for index, name in enumerate(manifest["levels"])}
    source_indexes = sorted(records_by_source)
    for page_start in range(0, len(source_indexes), args.rows_per_page):
        page_indexes = source_indexes[page_start : page_start + args.rows_per_page]
        sheet = Image.new("RGB", (16 * 128, len(page_indexes) * 128), (225, 225, 225))
        for row_index, source_index in enumerate(page_indexes):
            records = sorted(
                records_by_source[source_index],
                key=lambda item: (order[str(item["family"])], levels[str(item["level"])]),
            )
            original_path = args.data_root / str(records[0]["source_relative_path"])
            tiles = [make_tile(original_path, f"s{source_index:02d} original")]
            for record in records:
                tiles.append(
                    make_tile(
                        args.manifest.parent / str(record["output_path"]),
                        f"{record['family']}:{record['level']}",
                    )
                )
            for column, tile in enumerate(tiles):
                sheet.paste(tile, (column * 128, row_index * 128))
        page_number = page_start // args.rows_per_page + 1
        sheet.save(args.output_dir / f"review_page_{page_number:02d}.jpg", quality=92)


if __name__ == "__main__":
    main()
