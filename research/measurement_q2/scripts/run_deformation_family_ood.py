"""Run the frozen DAS-FC pipeline on a non-basis deformation family."""

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
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = deepcopy(json.loads(args.base_config.read_text(encoding="utf-8")))
    config["experiment_id"] = "das-fc-deformation-family-ood"
    config["seed_bases"]["test_unseen"] = 1_310_000
    config["seed_bases"]["test_known"] = 1_410_000
    config["counts_per_unseen_domain"]["test"] = 240
    config["unseen_domains"] = {
        "deformation_family_ood": {
            "sensor": {
                "noise_mm": 1.2,
                "heteroscedasticity": 1.8,
                "dropout": 0.18,
                "outlier_fraction": 0.05,
                "occlusion_fraction": 0.18,
                "pose_translation_mm": 14.0,
                "pose_rotation_deg": 2.0,
                "density_jitter": 0.20,
                "coordinate_jitter_fraction": 0.18,
                "support_candidate_contamination": 0.35,
                "support_candidate_miss": 0.25,
            },
            "geometry": {
                "deformation_family": "rbf_kink",
                "panel_count": 4,
                "panel_width_mm": 900.0,
                "height_mm": 2100.0,
                "joint_gap_mm": 30.0,
                "amplitude_mm": 24.0,
                "joint_slip_mm": 5.0,
            },
        }
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

