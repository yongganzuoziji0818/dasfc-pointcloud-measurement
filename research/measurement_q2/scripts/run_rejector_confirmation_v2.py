"""Run DAS-FC unchanged on fresh test seeds for rejector confirmation v2."""

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
    config["experiment_id"] = "das-fc-rejector-confirmation-v2"
    config["seed_bases"]["test_known"] = 600_000
    config["seed_bases"]["test_unseen"] = 700_000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = args.output_dir / "effective_config.json"
    effective_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sys.argv = [
        "run_das_fc_formal.py",
        "--config",
        str(effective_config),
        "--output-dir",
        str(args.output_dir),
    ]
    return run_das_fc_formal.main()


if __name__ == "__main__":
    raise SystemExit(main())
