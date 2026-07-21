"""Audit the official public SHREC'19 real-deformation mesh archive on cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MODEL_IDS = [f"{index:03d}" for index in range(50)]
EXPECTED_UNIQUE_PAIRS = {0: 14, 1: 26, 2: 19, 3: 17}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_obj(path: Path) -> dict:
    vertices = 0
    texture_vertices = 0
    normals = 0
    faces = 0
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    face_arities: Counter[int] = Counter()
    positive_face_index_min: int | None = None
    positive_face_index_max: int | None = None
    negative_face_indices = 0
    invalid_vertices = 0
    invalid_faces = 0
    repeated_face_indices = 0

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                try:
                    xyz = [float(value) for value in fields[1:4]]
                except (ValueError, IndexError):
                    invalid_vertices += 1
                    continue
                if len(xyz) != 3 or not all(math.isfinite(value) for value in xyz):
                    invalid_vertices += 1
                    continue
                for axis, value in enumerate(xyz):
                    minimum[axis] = min(minimum[axis], value)
                    maximum[axis] = max(maximum[axis], value)
                vertices += 1
            elif line.startswith("vt "):
                texture_vertices += 1
            elif line.startswith("vn "):
                normals += 1
            elif line.startswith("f "):
                fields = line.split()[1:]
                indices = []
                try:
                    indices = [int(field.split("/", 1)[0]) for field in fields]
                except (ValueError, IndexError):
                    invalid_faces += 1
                    continue
                if len(indices) < 3 or any(index == 0 for index in indices):
                    invalid_faces += 1
                    continue
                if len(set(indices)) != len(indices):
                    repeated_face_indices += 1
                face_arities[len(indices)] += 1
                faces += 1
                for index in indices:
                    if index < 0:
                        negative_face_indices += 1
                        if -index > vertices:
                            invalid_faces += 1
                    else:
                        positive_face_index_min = (
                            index if positive_face_index_min is None
                            else min(positive_face_index_min, index)
                        )
                        positive_face_index_max = (
                            index if positive_face_index_max is None
                            else max(positive_face_index_max, index)
                        )

    if vertices == 0 or faces == 0:
        raise ValueError(f"OBJ lacks valid vertices or faces: {path}")
    if positive_face_index_max is not None and positive_face_index_max > vertices:
        invalid_faces += 1
    diagonal = math.sqrt(sum((hi - lo) ** 2 for lo, hi in zip(minimum, maximum)))
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "vertices": vertices,
        "texture_vertices": texture_vertices,
        "normals": normals,
        "faces": faces,
        "face_arities": {str(key): value for key, value in sorted(face_arities.items())},
        "bounds": {"minimum": minimum, "maximum": maximum},
        "bounding_box_diagonal_source_units": diagonal,
        "positive_face_index_min": positive_face_index_min,
        "positive_face_index_max": positive_face_index_max,
        "negative_face_indices": negative_face_indices,
        "invalid_vertices": invalid_vertices,
        "invalid_faces": invalid_faces,
        "repeated_face_indices": repeated_face_indices,
    }


def parse_pairs(path: Path) -> tuple[list[dict], list[str]]:
    pairs = []
    errors = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or any(len(field) != 3 or not field.isdigit() for field in fields):
            errors.append(f"invalid_pair_line:{path.name}:{line_number}:{line}")
            continue
        pairs.append({"source": fields[0], "target": fields[1], "line": line_number})
    return pairs, errors


def audit_resolution(root: Path, resolution: str) -> tuple[dict, list[str], list[str]]:
    errors = []
    warnings = []
    resolution_root = root / f"SHREC19_{resolution}"
    model_paths = sorted((resolution_root / "models").glob("scan_*.obj"))
    model_ids = [path.stem.removeprefix("scan_") for path in model_paths]
    if model_ids != EXPECTED_MODEL_IDS:
        errors.append(f"{resolution}:unexpected_model_ids")

    models = {path.stem: audit_obj(path) for path in model_paths}
    invalid_meshes = [
        name for name, item in models.items()
        if item["invalid_vertices"] or item["invalid_faces"]
    ]
    if invalid_meshes:
        errors.append(f"{resolution}:invalid_meshes:{','.join(invalid_meshes)}")

    test_sets = {}
    for index in range(4):
        path = resolution_root / "test-sets" / f"test-set{index}.txt"
        if not path.is_file():
            errors.append(f"{resolution}:missing_test_set:{index}")
            continue
        pairs, pair_errors = parse_pairs(path)
        errors.extend(f"{resolution}:{item}" for item in pair_errors)
        pair_keys = [f"{item['source']},{item['target']}" for item in pairs]
        counts = Counter(pair_keys)
        duplicates = {key: count for key, count in sorted(counts.items()) if count > 1}
        unique_pair_keys = list(dict.fromkeys(pair_keys))
        missing_models = sorted({
            model_id
            for pair in pairs
            for model_id in (pair["source"], pair["target"])
            if f"scan_{model_id}" not in models
        })
        if len(unique_pair_keys) != EXPECTED_UNIQUE_PAIRS[index]:
            errors.append(
                f"{resolution}:test_set_{index}:expected_{EXPECTED_UNIQUE_PAIRS[index]}_"
                f"unique_pairs_observed_{len(unique_pair_keys)}"
            )
        if duplicates:
            warnings.append(f"{resolution}:test_set_{index}:duplicate_rows:{duplicates}")
        if missing_models:
            errors.append(f"{resolution}:test_set_{index}:missing_models:{missing_models}")
        test_sets[str(index)] = {
            "raw_rows": len(pairs),
            "unique_pairs": len(unique_pair_keys),
            "duplicates": duplicates,
            "missing_models": missing_models,
            "pairs": pairs,
        }

    ground_truth_candidates = sorted(
        str(path.relative_to(resolution_root))
        for path in resolution_root.rglob("*")
        if path.is_file() and path.suffix.lower() not in {".obj", ".txt", ".bat"}
    )
    return ({
        "root": str(resolution_root),
        "readme_sha256": sha256(resolution_root / "README.txt"),
        "model_count": len(models),
        "model_ids": model_ids,
        "models": models,
        "test_sets": test_sets,
        "raw_pair_rows": sum(item["raw_rows"] for item in test_sets.values()),
        "unique_pairs_within_test_sets": sum(
            item["unique_pairs"] for item in test_sets.values()
        ),
        "ground_truth_candidate_files": ground_truth_candidates,
    }, errors, warnings)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    archive = args.archive.resolve()
    errors = []
    warnings = []
    resolutions = {}
    for resolution in ("hires", "lores"):
        summary, resolution_errors, resolution_warnings = audit_resolution(root, resolution)
        resolutions[resolution] = summary
        errors.extend(resolution_errors)
        warnings.extend(resolution_warnings)

    for index in range(4):
        hires_pairs = resolutions["hires"]["test_sets"][str(index)]["pairs"]
        lores_pairs = resolutions["lores"]["test_sets"][str(index)]["pairs"]
        if hires_pairs != lores_pairs:
            errors.append(f"pair_definition_differs_between_resolutions:test_set_{index}")

    ground_truth_files = [
        item
        for resolution in resolutions.values()
        for item in resolution["ground_truth_candidate_files"]
    ]
    if not ground_truth_files:
        warnings.append("official_public_archive_contains_no_ground_truth_correspondence_files")

    status = "SHREC2019_AUDIT_FAIL" if errors else (
        "SHREC2019_AUDIT_PASS_WITH_WARNINGS" if warnings else "SHREC2019_AUDIT_PASS"
    )
    report = {
        "schema_version": "1.0",
        "status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SHREC'19 Shape Correspondence with Isometric and Non-Isometric Deformations",
        "official_metadata": {
            "doi": "10.17035/d.2019.0072003316",
            "repository_page": "https://research-data.cardiff.ac.uk/articles/dataset/27052618",
            "repository_file_id": "49265038",
            "license": "CC BY 4.0",
            "declared_scanner": "Artec3D Space Spider",
            "declared_deformations": [
                "articulating", "bending", "stretching", "topological/geometric change"
            ],
        },
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "root": str(root),
        "resolutions": resolutions,
        "errors": errors,
        "warnings": warnings,
        "audit_findings": {
            "mesh_files": sum(item["model_count"] for item in resolutions.values()),
            "physical_scans_per_resolution": resolutions["hires"]["model_count"],
            "raw_pair_rows_per_resolution": resolutions["hires"]["raw_pair_rows"],
            "unique_pairs_per_resolution": resolutions["hires"]["unique_pairs_within_test_sets"],
            "official_duplicate_pair_row": "043,045 in test-set0",
            "ground_truth_files_in_public_archive": len(ground_truth_files),
        },
        "evidence_boundary": {
            "real_scanned_meshes": True,
            "controlled_deformation_categories": True,
            "same_physical_specimen_dual_epoch_for_every_pair": "not_documented_and_not_guaranteed",
            "paper_reports_marker_based_correspondence_ground_truth": True,
            "correspondence_ground_truth_available_in_public_archive": False,
            "dense_3d_displacement_ground_truth_available": False,
            "absolute_length_unit_documented_in_archive": False,
            "physical_independent_unit_count": "not_documented; scans and pairs must not be treated as independent specimens",
            "allowed_current_use": [
                "unlabelled real-mesh domain stress testing",
                "qualitative registration failure analysis",
                "geometry and topology robustness diagnostics without accuracy claims",
            ],
            "not_allowed_current_claims": [
                "real displacement accuracy",
                "real simultaneous interval coverage",
                "real correspondence accuracy against unavailable ground truth",
                "76 independent physical validation units",
            ],
            "measurement_paper_role": (
                "exploratory cross-domain robustness material only; it does not close the missing "
                "real dual-epoch displacement-ground-truth gap"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": status,
        "archive_bytes": report["archive"]["bytes"],
        "archive_sha256": report["archive"]["sha256"],
        "mesh_files": report["audit_findings"]["mesh_files"],
        "scans_per_resolution": report["audit_findings"]["physical_scans_per_resolution"],
        "raw_pair_rows": report["audit_findings"]["raw_pair_rows_per_resolution"],
        "unique_pairs": report["audit_findings"]["unique_pairs_per_resolution"],
        "ground_truth_files": report["audit_findings"]["ground_truth_files_in_public_archive"],
        "errors": errors,
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
