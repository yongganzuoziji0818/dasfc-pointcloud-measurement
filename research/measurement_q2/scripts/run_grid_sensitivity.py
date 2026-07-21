"""Run one predeclared grid-sensitivity variant with fresh seed families."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.measurement_q2.scripts import run_das_fc_formal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--points-x", required=True, type=int)
    parser.add_argument("--points-z", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.points_x < 4 or args.points_z < 4:
        raise ValueError("grid dimensions must both be at least four")
    config = deepcopy(json.loads(args.base_config.read_text(encoding="utf-8")))
    config["experiment_id"] = f"das-fc-grid-{args.points_x}x{args.points_z}"
    config["grid"] = {"points_x": args.points_x, "points_z": args.points_z}
    config["seed_bases"] = {
        "tuning": 810_000,
        "validation": 910_000,
        "calibration": 1_010_000,
        "test_known": 1_110_000,
        "test_unseen": 1_210_000,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective = args.output_dir / "effective_config.json"
    effective.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.argv = [
        "run_das_fc_formal.py",
        "--config",
        str(effective),
        "--output-dir",
        str(args.output_dir),
    ]
    return run_das_fc_formal.main()


if __name__ == "__main__":
    raise SystemExit(main())

