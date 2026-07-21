"""Prepare three shared-seed P1 front-end configurations."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite P1 configs: {args.output_dir}")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    frontends = protocol["frontends"]
    args.output_dir.mkdir(parents=True)
    manifest = {"status": "P1_CONFIGS_FROZEN_NOT_EXECUTED", "configs": []}
    for frontend in frontends:
        config = deepcopy(base)
        config["schema_version"] = "1.1-p1-frontends"
        config["experiment_id"] = f"das-fc-p1-{frontend}-fresh-shared-seeds"
        config["seed_bases"] = {
            "tuning": 1_600_000,
            "validation": 1_700_000,
            "calibration": 1_800_000,
            "test_known": 1_900_000,
            "test_unseen": 2_000_000,
        }
        config["estimator"] = {
            "frontend": frontend,
            "query_workers": int(protocol["workers"]),
        }
        config["p1_metadata"] = {
            "frontend": frontend,
            "shared_cases_across_frontends": True,
            "disjoint_from_formal_v1_and_p3": True,
        }
        filename = f"{frontend}.json"
        (args.output_dir / filename).write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["configs"].append(filename)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

