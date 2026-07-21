"""Prepare frozen repeated-split configurations for cloud execution."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite config directory: {args.output_dir}")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    repeated = protocol["repeated_splits"]
    split_counts = dict(base["counts_per_known_domain"])
    maximum_calibration = max(repeated["calibration_sizes_per_group"])
    if split_counts["calibration"] < maximum_calibration:
        raise ValueError("base configuration has too few calibration trajectories")
    pool_size = (
        split_counts["tuning"]
        + split_counts["validation"]
        + maximum_calibration
        + split_counts["test"]
    )
    args.output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0",
        "status": "P2_CONFIGS_FROZEN_NOT_EXECUTED",
        "base_config": str(args.base_config),
        "protocol": str(args.protocol),
        "pool_size_per_known_domain": pool_size,
        "known_domains": list(base["known_domains"]),
        "runs": [],
    }
    for repetition, split_seed in enumerate(repeated["seeds"]):
        assigned: dict[str, dict[str, list[int]]] = {
            stage: {} for stage in ("tuning", "validation", "calibration", "test_known")
        }
        for domain_index, domain_name in enumerate(base["known_domains"]):
            pool = 6_000_000 + domain_index * 100_000 + np.arange(pool_size)
            rng = np.random.default_rng(split_seed + domain_index * 10_000)
            shuffled = rng.permutation(pool).tolist()
            t_end = split_counts["tuning"]
            v_end = t_end + split_counts["validation"]
            c_end = v_end + maximum_calibration
            assigned["tuning"][domain_name] = shuffled[:t_end]
            assigned["validation"][domain_name] = shuffled[t_end:v_end]
            assigned["calibration"][domain_name] = shuffled[v_end:c_end]
            assigned["test_known"][domain_name] = shuffled[c_end:]

        for calibration_size in repeated["calibration_sizes_per_group"]:
            config = deepcopy(base)
            config["schema_version"] = "1.1-p2-explicit-seeds"
            config["experiment_id"] = (
                f"das-fc-p2-r{repetition:02d}-cal{calibration_size:03d}"
            )
            config["counts_per_known_domain"]["calibration"] = calibration_size
            config["counts_per_unseen_domain"]["test"] = 0
            config["estimator"] = {"query_workers": int(protocol["workers"])}
            config["explicit_seeds"] = {
                "tuning": assigned["tuning"],
                "validation": assigned["validation"],
                "calibration": {
                    domain: values[:calibration_size]
                    for domain, values in assigned["calibration"].items()
                },
                "test_known": assigned["test_known"],
                "test_unseen": {domain: [] for domain in base["unseen_domains"]},
            }
            config["p2_metadata"] = {
                "repetition": repetition,
                "split_seed": split_seed,
                "calibration_trajectories_per_group": calibration_size,
                "shared_pool_across_repetitions": True,
            }
            filename = f"r{repetition:02d}_cal{calibration_size:03d}.json"
            (args.output_dir / filename).write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            manifest["runs"].append({
                "config": filename,
                **config["p2_metadata"],
            })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "run_count": len(manifest["runs"]),
        "output_dir": str(args.output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
