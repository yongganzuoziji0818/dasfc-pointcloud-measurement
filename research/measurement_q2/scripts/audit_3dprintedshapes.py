"""Audit the downloaded 3DPrintedShapes layout and a bounded PCD sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from SoundBarrierSystem.core.datasets import ThreeDPrintedShapesDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--max-point-clouds", type=int, default=6)
    parser.add_argument("--max-meshes", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = ThreeDPrintedShapesDataset(args.dataset_root, args.metadata_csv)
    report = dataset.validate_layout()
    report["sampled_point_clouds"] = []
    report["sampled_meshes"] = []

    for record in list(dataset.records())[: max(0, args.max_meshes)]:
        sample = dataset.inspect_mesh(record.mesh_path)
        sample["specimen_id"] = record.specimen_id
        report["sampled_meshes"].append(sample)

    remaining = max(0, args.max_point_clouds)
    for record in dataset.records():
        for scanner, variants in record.scan_paths.items():
            for variant, path in variants.items():
                if remaining == 0:
                    break
                if path.is_file():
                    sample = dataset.inspect_point_cloud(path)
                    sample.update(
                        specimen_id=record.specimen_id,
                        scanner=scanner,
                        variant=variant,
                    )
                    report["sampled_point_clouds"].append(sample)
                    remaining -= 1
            if remaining == 0:
                break
        if remaining == 0:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["layout_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
