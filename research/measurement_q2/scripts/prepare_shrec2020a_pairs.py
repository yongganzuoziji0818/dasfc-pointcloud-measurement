"""Convert released SHREC 2020a marker truth into compact cloud pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zero_based_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.min() >= 1 and faces.max() <= vertex_count:
        faces = faces - 1
    if faces.min() < 0 or faces.max() >= vertex_count:
        raise ValueError("target triangle indices are outside the vertex array")
    return faces


def decode_barycentric_truth(
    values: np.ndarray,
    target_faces: np.ndarray,
    target: np.ndarray,
    sparse_correspondence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    candidates = []
    weights = values[:, 1:]
    raw_face = values[:, 0]
    for face_index_base in (0, 1):
        face_index = np.rint(raw_face).astype(np.int64) - face_index_base
        valid = (
            (np.abs(raw_face - np.rint(raw_face)) <= 1e-8)
            & (face_index >= 0)
            & (face_index < target_faces.shape[0])
            & np.isfinite(weights).all(axis=1)
            & (np.abs(weights.sum(axis=1) - 1.0) <= 1e-5)
            & (weights.min(axis=1) >= -1e-5)
            & (weights.max(axis=1) <= 1.0 + 1e-5)
        )
        reconstructed = np.full((values.shape[0], 3), np.nan, dtype=np.float64)
        reconstructed[valid] = np.einsum(
            "nij,ni->nj", target[target_faces[face_index[valid]]], weights[valid]
        )
        for source_column in (0, 1):
            target_column = 1 - source_column
            for correspondence_index_base in (0, 1):
                source_index = (
                    sparse_correspondence[:, source_column]
                    - correspondence_index_base
                )
                target_index = (
                    sparse_correspondence[:, target_column]
                    - correspondence_index_base
                )
                in_bounds = (
                    (source_index >= 0)
                    & (source_index < values.shape[0])
                    & (target_index >= 0)
                    & (target_index < target.shape[0])
                )
                clipped_source = np.clip(source_index, 0, values.shape[0] - 1)
                mapped = in_bounds & valid[clipped_source]
                errors = np.linalg.norm(
                    reconstructed[source_index[mapped]] - target[target_index[mapped]],
                    axis=1,
                ) if mapped.any() else np.empty(0, dtype=np.float64)
                candidates.append({
                    "face_index_base": face_index_base,
                    "correspondence_source_column_zero_based": source_column,
                    "correspondence_index_base": correspondence_index_base,
                    "face_index": face_index,
                    "valid": valid,
                    "valid_count": int(valid.sum()),
                    "mapped_sparse_count": int(mapped.sum()),
                    "median_target_vertex_error": (
                        float(np.median(errors)) if errors.size else float("inf")
                    ),
                    "maximum_target_vertex_error": (
                        float(errors.max()) if errors.size else float("inf")
                    ),
                })
    selected = min(
        candidates,
        key=lambda item: (
            -item["mapped_sparse_count"],
            item["median_target_vertex_error"],
            item["face_index_base"],
        ),
    )
    if selected["mapped_sparse_count"] != sparse_correspondence.shape[0]:
        raise ValueError(
            "not every released sparse marker correspondence maps to a valid "
            f"barycentric row: {selected['mapped_sparse_count']}/"
            f"{sparse_correspondence.shape[0]}"
        )
    if selected["valid_count"] < 200:
        raise ValueError("too few released marker correspondences")
    diagnostics = {
        "semantics": "released sparse texture-marker truth; sentinel rows invalid",
        "face_column_zero_based": 0,
        "weight_columns_zero_based": [1, 2, 3],
        "face_index_base_in_file": selected["face_index_base"],
        "correspondence_source_column_zero_based": selected[
            "correspondence_source_column_zero_based"
        ],
        "correspondence_index_base_in_file": selected["correspondence_index_base"],
        "released_sparse_rows": int(sparse_correspondence.shape[0]),
        "unique_valid_barycentric_rows": selected["valid_count"],
        "median_target_vertex_error_source_units": selected[
            "median_target_vertex_error"
        ],
        "maximum_target_vertex_error_source_units": selected[
            "maximum_target_vertex_error"
        ],
        "candidate_diagnostics": [
            {
                "face_index_base": item["face_index_base"],
                "correspondence_source_column_zero_based": item[
                    "correspondence_source_column_zero_based"
                ],
                "correspondence_index_base": item["correspondence_index_base"],
                "valid_count": item["valid_count"],
                "mapped_sparse_count": item["mapped_sparse_count"],
                "median_target_vertex_error": item["median_target_vertex_error"],
            }
            for item in candidates
        ],
    }
    return (
        selected["face_index"],
        weights,
        selected["valid"],
        diagnostics,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite converted pairs: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    truth_dir = args.root / "SHREC20a_hires_gts"
    records = []
    for path in sorted(truth_dir.glob("scan*_scan00.mat")):
        data = loadmat(path, simplify_cells=True)
        source = np.asarray(data["M"]["VERT"], dtype=np.float64)
        target = np.asarray(data["N"]["VERT"], dtype=np.float64)
        target_faces = zero_based_faces(data["N"]["TRIV"], target.shape[0])
        barycentric = np.asarray(data["baryc_corr"], dtype=np.float64)
        sparse_correspondence = np.asarray(data["corr"], dtype=np.int64)
        if barycentric.shape != (source.shape[0], 4):
            raise ValueError(f"unexpected barycentric truth shape in {path}")
        face_index, weights, valid, decoding = decode_barycentric_truth(
            barycentric, target_faces, target, sparse_correspondence
        )
        ground_truth = np.full(source.shape, np.nan, dtype=np.float64)
        triangles = target_faces[face_index[valid]]
        ground_truth[valid] = np.einsum(
            "nij,ni->nj", target[triangles], weights[valid]
        )
        target_min = target.min(axis=0)
        target_max = target.max(axis=0)
        centre = 0.5 * (target_min + target_max)
        diagonal = float(np.linalg.norm(target_max - target_min))
        if diagonal <= 0 or not np.isfinite(diagonal):
            raise ValueError(f"invalid target bounding box in {path}")
        output_path = args.output_dir / f"{path.stem}.npz"
        np.savez_compressed(
            output_path,
            source=(source - centre) / diagonal,
            target=(target - centre) / diagonal,
            ground_truth_target=(ground_truth - centre) / diagonal,
            ground_truth_valid=valid,
            released_sparse_correspondence=sparse_correspondence,
            target_faces=target_faces,
            normalization_centre=centre,
            normalization_diagonal=diagonal,
        )
        records.append({
            "pair_id": path.stem,
            "source_vertices": int(source.shape[0]),
            "target_vertices": int(target.shape[0]),
            "target_faces": int(target_faces.shape[0]),
            "valid_marker_correspondences": int(valid.sum()),
            "valid_fraction": float(valid.mean()),
            "barycentric_decoding": decoding,
            "normalization_diagonal_source_units": diagonal,
            "output": output_path.name,
            "sha256": sha256(output_path),
        })
    manifest = {
        "schema_version": "1.0",
        "status": "SHREC2020A_SPARSE_MARKER_PAIRS_PREPARED",
        "pair_count": len(records),
        "normalization": "subtract full-scan bbox centre; divide by full-scan bbox diagonal",
        "independent_physical_object_count": 1,
        "ground_truth_semantics": (
            "approximately 300 released texture-marker correspondences per scan pair; "
            "not dense truth over all source vertices"
        ),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": manifest["status"],
        "pair_count": manifest["pair_count"],
        "minimum_marker_count": min(
            item["valid_marker_correspondences"] for item in records
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
