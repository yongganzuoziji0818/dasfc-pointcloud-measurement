"""Audit the public SHREC 2020a real non-rigid scan benchmark on cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obj_counts(path: Path) -> dict:
    vertices = 0
    texture_vertices = 0
    normals = 0
    faces = 0
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = np.fromstring(line[2:], sep=" ", dtype=np.float64)
                if values.size < 3 or not np.isfinite(values[:3]).all():
                    raise ValueError(f"invalid vertex in {path}")
                minimum = np.minimum(minimum, values[:3])
                maximum = np.maximum(maximum, values[:3])
                vertices += 1
            elif line.startswith("vt "):
                texture_vertices += 1
            elif line.startswith("vn "):
                normals += 1
            elif line.startswith("f "):
                faces += 1
    if vertices == 0 or faces == 0:
        raise ValueError(f"OBJ lacks vertices or faces: {path}")
    return {
        "vertices": vertices,
        "texture_vertices": texture_vertices,
        "normals": normals,
        "faces": faces,
        "bounds": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
        "bounding_box_diagonal": float(np.linalg.norm(maximum - minimum)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def mat_summary(path: Path) -> dict:
    variables = [
        {"name": name, "shape": list(shape), "matlab_class": matlab_class}
        for name, shape, matlab_class in whosmat(path)
    ]
    loaded = loadmat(path)
    simplified = loadmat(path, simplify_cells=True)
    numeric = {}
    for name, value in loaded.items():
        if name.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if np.issubdtype(value.dtype, np.number):
            finite = np.isfinite(value)
            numeric[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_fraction": float(finite.mean()) if value.size else None,
                "minimum": float(value[finite].min()) if finite.any() else None,
                "maximum": float(value[finite].max()) if finite.any() else None,
            }
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "variables": variables,
        "numeric": numeric,
        "mesh_structures": {
            name: {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                if isinstance(value, np.ndarray)
                else {"python_type": type(value).__name__}
                for key, value in simplified[name].items()
            }
            for name in ("M", "N")
            if isinstance(simplified.get(name), dict)
        },
    }


def correspondence_schema_summary(path: Path) -> dict:
    data = loadmat(path, simplify_cells=True)
    source = np.asarray(data["M"]["VERT"], dtype=np.float64)
    target = np.asarray(data["N"]["VERT"], dtype=np.float64)
    target_faces_raw = np.asarray(data["N"]["TRIV"], dtype=np.int64)
    barycentric = np.asarray(data["baryc_corr"], dtype=np.float64)
    sparse = np.asarray(data["corr"], dtype=np.int64)
    candidates = []
    for face_base in (0, 1):
        target_faces = target_faces_raw - face_base
        if target_faces.min() < 0 or target_faces.max() >= target.shape[0]:
            continue
        face_index = np.rint(barycentric[:, 0]).astype(np.int64) - face_base
        weights = barycentric[:, 1:]
        valid = (
            (np.abs(barycentric[:, 0] - np.rint(barycentric[:, 0])) <= 1e-8)
            & (face_index >= 0)
            & (face_index < target_faces.shape[0])
            & np.isfinite(weights).all(axis=1)
            & (np.abs(weights.sum(axis=1) - 1.0) <= 1e-5)
            & (weights.min(axis=1) >= -1e-5)
            & (weights.max(axis=1) <= 1.0 + 1e-5)
        )
        reconstructed = np.full((source.shape[0], 3), np.nan, dtype=np.float64)
        reconstructed[valid] = np.einsum(
            "nij,ni->nj", target[target_faces[face_index[valid]]], weights[valid]
        )
        for source_column in (0, 1):
            target_column = 1 - source_column
            for corr_base in (0, 1):
                source_index = sparse[:, source_column] - corr_base
                target_index = sparse[:, target_column] - corr_base
                in_bounds = (
                    (source_index >= 0)
                    & (source_index < source.shape[0])
                    & (target_index >= 0)
                    & (target_index < target.shape[0])
                )
                mapped = in_bounds & valid[np.clip(source_index, 0, source.shape[0] - 1)]
                if mapped.any():
                    errors = np.linalg.norm(
                        reconstructed[source_index[mapped]] - target[target_index[mapped]],
                        axis=1,
                    )
                    median_error = float(np.median(errors))
                    maximum_error = float(errors.max())
                else:
                    median_error = None
                    maximum_error = None
                candidates.append({
                    "face_index_base": face_base,
                    "corr_source_column_zero_based": source_column,
                    "corr_index_base": corr_base,
                    "valid_barycentric_rows": int(valid.sum()),
                    "sparse_rows": int(sparse.shape[0]),
                    "sparse_rows_mapped_to_valid_barycentric": int(mapped.sum()),
                    "median_target_vertex_error_source_units": median_error,
                    "maximum_target_vertex_error_source_units": maximum_error,
                })
    selected = min(
        candidates,
        key=lambda item: (
            -item["sparse_rows_mapped_to_valid_barycentric"],
            float("inf")
            if item["median_target_vertex_error_source_units"] is None
            else item["median_target_vertex_error_source_units"],
        ),
    )
    return {
        "interpretation": (
            "Rows index source vertices; column 0 is a target face index and columns "
            "1:4 are barycentric weights. Sentinel rows are invalid."
        ),
        "selected_candidate": selected,
        "candidate_diagnostics": candidates,
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    model_dir = root / "models"
    truth_dir = root / "SHREC20a_hires_gts"
    pair_file = root / "test-sets" / "test-set0.txt"
    models = sorted(model_dir.glob("scan*.obj"))
    truths = sorted(truth_dir.glob("scan*_scan00.mat"))
    pairs = []
    for raw_line in pair_file.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            source, target = [item.strip() for item in raw_line.split(",")]
            pairs.append({"source": source, "target": target})
    model_summaries = {path.stem: obj_counts(path) for path in models}
    truth_summaries = {path.stem: mat_summary(path) for path in truths}
    correspondence_schemas = {
        path.stem: correspondence_schema_summary(path) for path in truths
    }
    errors = []
    if len(models) != 12:
        errors.append(f"expected_12_models_observed_{len(models)}")
    if len(truths) != 11:
        errors.append(f"expected_11_ground_truths_observed_{len(truths)}")
    if len(pairs) != 11 or any(pair["target"] != "scan00" for pair in pairs):
        errors.append("unexpected_test_pair_definition")
    for pair in pairs:
        if pair["source"] not in model_summaries or pair["target"] not in model_summaries:
            errors.append(f"missing_model_for_pair:{pair['source']}:{pair['target']}")
        if f"{pair['source']}_{pair['target']}" not in truth_summaries:
            errors.append(f"missing_truth_for_pair:{pair['source']}:{pair['target']}")
    report = {
        "schema_version": "1.0",
        "status": "SHREC2020A_AUDIT_PASS" if not errors else "SHREC2020A_AUDIT_FAIL",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "models": model_summaries,
        "ground_truth": truth_summaries,
        "correspondence_schema": correspondence_schemas,
        "test_pairs": pairs,
        "errors": errors,
        "evidence_boundary": {
            "physical_objects": 1,
            "partial_to_full_scan_pairs": len(pairs),
            "scanner": "Artec3D Space Spider, as documented by the benchmark page",
            "absolute_unit_documented_in_archive": False,
            "allowed_scale_for_analysis": "dimensionless full-scan bounding-box normalization",
            "confirmatory_coverage_inference_allowed": False,
            "reason": (
                "All eleven pairs are repeated deformations/scans of one stuffed rabbit; "
                "they are not eleven independent physical specimens."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "models": len(models),
        "ground_truth_files": len(truths),
        "pairs": len(pairs),
        "ground_truth_variables": {
            key: value["variables"] for key, value in truth_summaries.items()
        },
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
