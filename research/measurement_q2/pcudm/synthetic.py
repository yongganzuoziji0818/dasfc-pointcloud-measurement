"""Physics-controlled panel structure and sensor observation generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SyntheticDomain:
    """Observation-domain parameters in millimetres and degrees."""

    name: str = "nominal"
    noise_mm: float = 0.8
    heteroscedasticity: float = 1.0
    dropout: float = 0.08
    outlier_fraction: float = 0.02
    occlusion_fraction: float = 0.10
    pose_translation_mm: float = 8.0
    pose_rotation_deg: float = 1.0
    density_jitter: float = 0.10
    coordinate_jitter_fraction: float = 0.12
    support_candidate_contamination: float = 0.20
    support_candidate_miss: float = 0.10


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    domain: str
    reference: np.ndarray
    target: np.ndarray
    panel_ids: np.ndarray
    support_true: np.ndarray
    support_candidates: np.ndarray
    displacement_true: np.ndarray
    normal_displacement_true: np.ndarray
    rigid_rotation_true: np.ndarray
    rigid_translation_true: np.ndarray
    target_clean_corresponding: np.ndarray
    noise_scale_true: np.ndarray
    valid_field_mask: np.ndarray
    metadata: dict


def _panel_grid(
    rng: np.random.Generator,
    panel_count: int,
    points_x: int,
    points_z: int,
    panel_width_mm: float,
    height_mm: float,
    joint_gap_mm: float,
    coordinate_jitter_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    panel_ids = []
    local_coordinates = []
    for panel in range(panel_count):
        start = panel * (panel_width_mm + joint_gap_mm)
        local_x = np.linspace(0.0, panel_width_mm, points_x, dtype=np.float64)
        z = np.linspace(0.0, height_mm, points_z, dtype=np.float64)
        xx, zz = np.meshgrid(local_x, z, indexing="xy")
        local_x_normalized = xx / panel_width_mm
        local_z_normalized = zz / height_mm
        interior = (
            (local_x_normalized > 0.0)
            & (local_x_normalized < 1.0)
            & (local_z_normalized > 0.0)
            & (local_z_normalized < 1.0)
        )
        if coordinate_jitter_fraction > 0:
            step_x = panel_width_mm / max(points_x - 1, 1)
            step_z = height_mm / max(points_z - 1, 1)
            xx = xx.copy()
            zz = zz.copy()
            xx[interior] += rng.normal(
                scale=coordinate_jitter_fraction * step_x, size=int(interior.sum())
            )
            zz[interior] += rng.normal(
                scale=coordinate_jitter_fraction * step_z, size=int(interior.sum())
            )
            xx = np.clip(xx, 0.0, panel_width_mm)
            zz = np.clip(zz, 0.0, height_mm)
            local_x_normalized = xx / panel_width_mm
            local_z_normalized = zz / height_mm
        yy = np.zeros_like(xx)
        points.append(np.column_stack((xx.ravel() + start, yy.ravel(), zz.ravel())))
        panel_ids.append(np.full(xx.size, panel, dtype=np.int32))
        local_coordinates.append(
            np.column_stack((local_x_normalized.ravel(), local_z_normalized.ravel()))
        )
    return (
        np.concatenate(points, axis=0),
        np.concatenate(panel_ids, axis=0),
        np.concatenate(local_coordinates, axis=0),
    )


def _displacement_field(
    local: np.ndarray,
    panel_ids: np.ndarray,
    rng: np.random.Generator,
    amplitude_mm: float,
    joint_slip_mm: float,
    deformation_family: str,
) -> np.ndarray:
    x = local[:, 0]
    z = local[:, 1]
    offsets = joint_slip_mm * (panel_ids - np.mean(np.unique(panel_ids)))
    # Support lines are physically restrained. Tapering preserves zero displacement
    # at the bottom and vertical panel edges while still allowing interior joint slip.
    taper = np.sin(np.pi * x) * np.sin(np.pi * z)
    panel_count = int(panel_ids.max()) + 1
    if deformation_family == "modal_bulge":
        panel_phase = rng.uniform(-0.35, 0.35, panel_count)
        panel_gain = rng.uniform(0.75, 1.25, panel_count)
        bending = amplitude_mm * panel_gain[panel_ids] * taper
        second_mode = 0.28 * amplitude_mm * np.sin(
            2.0 * np.pi * z + panel_phase[panel_ids]
        )
        second_mode *= np.sin(np.pi * x)
        center_x = rng.uniform(0.25, 0.75, panel_count)
        center_z = rng.uniform(0.35, 0.75, panel_count)
        bulge = 0.25 * amplitude_mm * np.exp(
            -(
                (x - center_x[panel_ids]) ** 2 / 0.025
                + (z - center_z[panel_ids]) ** 2 / 0.05
            )
        )
        normal = bending + second_mode + bulge + offsets * taper
    elif deformation_family == "rbf_kink":
        normal = np.zeros_like(x)
        for panel in range(panel_count):
            index = panel_ids == panel
            panel_x = x[index]
            panel_z = z[index]
            panel_field = np.zeros(index.sum(), dtype=np.float64)
            for _ in range(3):
                centre_x = rng.uniform(0.15, 0.85)
                centre_z = rng.uniform(0.15, 0.85)
                width_x = rng.uniform(0.035, 0.14)
                width_z = rng.uniform(0.045, 0.20)
                signed_gain = rng.uniform(-0.70, 0.70) * amplitude_mm
                panel_field += signed_gain * np.exp(
                    -(
                        (panel_x - centre_x) ** 2 / width_x
                        + (panel_z - centre_z) ** 2 / width_z
                    )
                )
            hinge_x = rng.uniform(0.30, 0.70)
            hinge_sign = rng.choice((-1.0, 1.0))
            hinge = hinge_sign * 0.55 * amplitude_mm * np.maximum(
                panel_x - hinge_x, 0.0
            )
            ripple_phase = rng.uniform(-np.pi, np.pi)
            ripple = 0.18 * amplitude_mm * np.sin(
                3.0 * np.pi * panel_x + ripple_phase
            ) * np.sin(2.5 * np.pi * panel_z)
            panel_field = (panel_field + hinge + ripple) * taper[index]
            normal[index] = panel_field
        normal += offsets * taper
    else:
        raise ValueError(
            "deformation_family must be 'modal_bulge' or 'rbf_kink'"
        )
    displacement = np.zeros((local.shape[0], 3), dtype=np.float64)
    displacement[:, 1] = normal
    return displacement


def generate_case(
    seed: int,
    domain: SyntheticDomain | None = None,
    *,
    panel_count: int = 3,
    points_x: int = 24,
    points_z: int = 20,
    panel_width_mm: float = 1000.0,
    height_mm: float = 1800.0,
    joint_gap_mm: float = 25.0,
    amplitude_mm: float = 18.0,
    joint_slip_mm: float = 3.0,
    deformation_family: str = "modal_bulge",
) -> SyntheticCase:
    """Generate one independent reference/observation scan pair.

    The target is formed by applying structural displacement first and an unknown
    global rigid transform second. Noise, structured occlusion, dropout, density
    variation, and outliers are then applied to the observed target only.
    """

    domain = domain or SyntheticDomain()
    rng = np.random.default_rng(seed)
    reference, panel_ids, local = _panel_grid(
        rng,
        panel_count,
        points_x,
        points_z,
        panel_width_mm,
        height_mm,
        joint_gap_mm,
        domain.coordinate_jitter_fraction,
    )
    displacement = _displacement_field(
        local,
        panel_ids,
        rng,
        amplitude_mm=amplitude_mm,
        joint_slip_mm=joint_slip_mm,
        deformation_family=deformation_family,
    )

    rotation_vector = rng.normal(size=3)
    rotation_vector /= max(np.linalg.norm(rotation_vector), 1e-12)
    rotation_vector *= np.deg2rad(rng.uniform(0.25, domain.pose_rotation_deg))
    rigid_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    direction = rng.normal(size=3)
    direction /= max(np.linalg.norm(direction), 1e-12)
    rigid_translation = direction * rng.uniform(
        0.25 * domain.pose_translation_mm, domain.pose_translation_mm
    )

    deformed = reference + displacement
    clean_target = deformed @ rigid_rotation.T + rigid_translation

    x_norm = local[:, 0]
    z_norm = local[:, 1]
    range_factor = 0.4 + 0.6 * z_norm
    incidence_factor = 1.0 + 0.7 * np.abs(x_norm - 0.5)
    noise_scale = domain.noise_mm * (
        1.0 + domain.heteroscedasticity * range_factor * incidence_factor
    )
    noisy_target = clean_target + rng.normal(size=clean_target.shape) * noise_scale[:, None]

    keep = rng.random(reference.shape[0]) >= domain.dropout
    if domain.occlusion_fraction > 0:
        occlusion_panel = int(rng.integers(0, panel_count))
        center = rng.uniform(0.25, 0.75, size=2)
        half_width = np.sqrt(domain.occlusion_fraction) * 0.35
        occluded = (
            (panel_ids == occlusion_panel)
            & (np.abs(local[:, 0] - center[0]) < half_width)
            & (np.abs(local[:, 1] - center[1]) < half_width)
        )
        keep &= ~occluded

    # Density jitter creates a second, independent thinning process with a smooth
    # vertical gradient instead of uniform random deletion only.
    density_keep = 1.0 - domain.density_jitter * (0.25 + 0.75 * z_norm)
    keep &= rng.random(reference.shape[0]) < density_keep
    target = noisy_target[keep]

    n_outliers = int(np.ceil(domain.outlier_fraction * max(target.shape[0], 1)))
    if n_outliers:
        minimum = clean_target.min(axis=0) - np.array([100.0, 80.0, 100.0])
        maximum = clean_target.max(axis=0) + np.array([100.0, 80.0, 100.0])
        outliers = rng.uniform(minimum, maximum, size=(n_outliers, 3))
        target = np.concatenate((target, outliers), axis=0)
        target = target[rng.permutation(target.shape[0])]

    support_true = (
        (local[:, 0] <= 0.08)
        | (local[:, 0] >= 0.92)
        | (local[:, 1] <= 0.08)
    )
    support = support_true.copy()
    true_index = np.flatnonzero(support_true)
    missed = rng.random(true_index.size) < domain.support_candidate_miss
    support[true_index[missed]] = False
    false_pool = np.flatnonzero(~support_true)
    false_count = min(
        false_pool.size,
        int(round(domain.support_candidate_contamination * max(true_index.size, 1))),
    )
    if false_count:
        support[rng.choice(false_pool, size=false_count, replace=False)] = True
    valid = ~(
        (local[:, 0] < 0.02)
        | (local[:, 0] > 0.98)
        | (local[:, 1] < 0.02)
    )
    return SyntheticCase(
        case_id=f"{domain.name}-seed-{seed}",
        domain=domain.name,
        reference=reference,
        target=target,
        panel_ids=panel_ids,
        support_true=support_true,
        support_candidates=support,
        displacement_true=displacement,
        normal_displacement_true=displacement[:, 1],
        rigid_rotation_true=rigid_rotation,
        rigid_translation_true=rigid_translation,
        target_clean_corresponding=clean_target,
        noise_scale_true=noise_scale,
        valid_field_mask=valid,
        metadata={
            "seed": seed,
            "panel_count": panel_count,
            "points_per_reference": int(reference.shape[0]),
            "points_per_target": int(target.shape[0]),
            "amplitude_mm": amplitude_mm,
            "joint_slip_mm": joint_slip_mm,
            "deformation_family": deformation_family,
            "domain": domain.__dict__,
        },
    )
