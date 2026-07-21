"""Create the immutable fresh-seed P3 configuration on cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite P3 config: {args.output}")
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    config["schema_version"] = "1.1-p3-classical-baselines"
    config["experiment_id"] = "das-fc-p3-classical-baselines-fresh-seeds"
    config["classical_baselines"] = True
    config["counts_per_unseen_domain"]["test"] = 0
    config["seed_bases"] = {
        "tuning": 1_100_000,
        "validation": 1_200_000,
        "calibration": 1_300_000,
        "test_known": 1_400_000,
        "test_unseen": 1_500_000,
    }
    config["estimator"] = {"query_workers": int(protocol["workers"])}
    config["p3_metadata"] = {
        "prospective_extension": True,
        "test_cases_disjoint_from_formal_v1": True,
        "holm_contrasts": [
            "homoscedastic_grouped",
            "classical_max_t",
            "raw_local_grouped"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "experiment_id": config["experiment_id"],
        "classical_baselines": config["classical_baselines"],
        "test_known_seed_base": config["seed_bases"]["test_known"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

