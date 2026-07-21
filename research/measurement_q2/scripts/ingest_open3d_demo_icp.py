"""Download and audit Open3D's real Redwood ICP demo fragments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)

    dataset = o3d.data.DemoICPPointClouds(str(args.data_root.resolve()))
    clouds = []
    for raw_path in dataset.paths:
        path = Path(raw_path).resolve()
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise RuntimeError(f"Unreadable demo point cloud: {path}")
        finite = np.isfinite(points).all(axis=1)
        if not finite.all():
            raise RuntimeError(f"Non-finite coordinates in demo point cloud: {path}")
        clouds.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "points": int(points.shape[0]),
                "min_xyz": points.min(axis=0).tolist(),
                "max_xyz": points.max(axis=0).tolist(),
            }
        )

    transform_path = Path(dataset.transformation_log_path).resolve()
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "open3d-demo-icp-point-clouds",
        "source": "Open3D DemoICPPointClouds; Redwood living-room1 fragments",
        "official_docs": "https://www.open3d.org/docs/latest/tutorial/data/index.html",
        "license": "CC-BY-3.0",
        "evidence_role": "engineering smoke only; not a sound-barrier displacement benchmark",
        "clouds": clouds,
        "transformation_log": {
            "path": str(transform_path),
            "size_bytes": transform_path.stat().st_size,
            "sha256": sha256(transform_path),
        },
        "ok": len(clouds) == 3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

