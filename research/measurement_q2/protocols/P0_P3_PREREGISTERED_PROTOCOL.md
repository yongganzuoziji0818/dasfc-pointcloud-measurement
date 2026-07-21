# P0-P3 external-evidence extension protocol

## Status and evidence boundary

- Protocol status: `AUTHORIZED_DESIGN_FROZEN_EXECUTION_NOT_STARTED`.
- Authorization: the user approved P0, P1, P2, and P3 on 2026-07-19.
- Scientific execution is cloud-only under `/workspace/sound-barrier-measurement`.
- No confidential railway data, real sound-barrier scans, or real dual-epoch railway validation are available.
- This extension may strengthen external validity, calibration evidence, and cross-front-end robustness. It must not be described as real sound-barrier validation.

## P0: public physics-based state-pair benchmark

### Source and feasibility gate

The primary source is STEEL-3dPointClouds (DesignSafe project `PRJ-5743`, DOI `10.17603/ds2-1deq-4z12`). The portal reports one `STEEL-3dPointClouds.hdf5` file of 4.5 TB. The current persistent disk has about 935 GB free, so a whole-file download is prohibited.

P0 may proceed only if all of the following G1-access checks pass:

1. the public object supports bounded random/range reads without downloading the full file;
2. HDF5 metadata and selected chunks can be read deterministically on the cloud;
3. the selected subset, cache, and intermediate artifacts fit within a 300 GB project quota;
4. the source object identity, remote path, reported size, modification time, and per-block hashes are recorded;
5. no account credential or confidential file is embedded in code or evidence.

If G1-access fails, STEEL is recorded as an infeasible candidate and no STEEL result is claimed. P1-P3 may still run on already authorized synthetic evidence, but they do not replace P0.

The 2026-07-19 access audit found that the portal reports 4,963,883,174,920 bytes, rejects direct transfer with HTTP 413, and the TACC large-transfer host is not reachable from the current cloud endpoint. STEEL is therefore `ACCESS_BLOCKED_NO_NEAR_DATA_ACCOUNT` for this execution stage. It may be resumed only through a DesignSafe/TACC near-data job or another source-authorized subset; it must not be silently replaced by generated STEEL-like data.

### P0b feasible real-scan supplement

SHREC 2020a is added as a bounded public real-scan supplement before formal extension runs. It contains one stuffed rabbit captured with an Artec3D Space Spider, one full scan, eleven partial/deformed scans, and eleven released files with approximately 300 texture-marker correspondences per pair. The archive stores these sparse correspondences as valid barycentric rows inside source-vertex-length arrays and uses sentinel rows elsewhere; they are not dense truth. Coordinates are normalized by the full-scan bounding-box diagonal because the archive does not document an absolute unit.

All eleven pairs belong to one physical object. They may support released-marker correspondence errors, front-end failure analysis, and dimensionless uncertainty ranking, but they are one specimen cluster and cannot support confirmatory 95% coverage inference or a real-structure generalization claim. This supplement does not satisfy the confirmatory P0 trajectory-count gate.

### Independent unit and leakage rule

The candidate independent unit is a complete numerical member/load trajectory, not a state increment, node, point, or state pair. The exact trajectory key must be derived from HDF5 groups and attributes before any split is made.

- All states from one trajectory remain in exactly one of tuning, validation, calibration, or test.
- The main analysis uses at most one predeclared state pair per trajectory.
- Extra state pairs from the same trajectory are sensitivity observations only and are analyzed with trajectory-cluster resampling.
- No test trajectory may affect scale fitting, front-end selection, calibration quantiles, rejection thresholds, or sample eligibility thresholds.

### State-pair rule

After schema audit, the first state is the earliest valid reference state. The target is the latest valid state that satisfies all frozen quality checks and remains within the dataset's documented response range. If capacity-loss or normalized-history metadata permits pre-outcome binning, trajectories are stratified into four fixed severity bins before splitting; otherwise only source-defined member/load families are used. State selection is deterministic and recorded before field estimation.

### Eligibility and stopping rule

A trajectory is eligible only when reference and target coordinates have identical ordered topology, finite values, at least 200 field locations, and a documented unit conversion. All excluded trajectories remain in the denominator of the audit table with a reason.

- Confirmatory target: at least 300 eligible independent trajectories and at least 60 test trajectories.
- Preferred target: at least 600 eligible trajectories and at least 150 test trajectories.
- If fewer than 300 eligible trajectories exist, P0 is exploratory and cannot support a confirmatory external-coverage claim.
- Any split leakage, topology mismatch, undocumented unit, or outcome-dependent state selection stops P0 before formal estimation.

## P1: cross-front-end robustness

The uncertainty methods are compared under three frozen, non-learning measurement front ends:

1. `cascade_strong`: the current support-aware trimmed point-to-point cascade;
2. `multiscale_trimmed_ptp`: deterministic coarse-to-fine trimmed point-to-point ICP followed by the same structural field fit;
3. `robust_ptpl`: robust point-to-plane ICP followed by the same structural field fit.

The same trajectory splits, field mask, unit conversion, calibration budget, and uncertainty baselines are used for all front ends. A front-end failure stays in the denominator. Front-end-specific tuning uses validation trajectories only. No front end is removed because it performs poorly.

Primary robustness claim: the sign of the DAS-FC minus homoscedastic pair-max interval-score effect must be negative for all three front ends, and the trajectory-cluster bootstrap 95% confidence interval must exclude zero for at least two front ends. Coverage and efficiency are reported jointly; a narrower but under-covering method does not pass.

## P2: repeated calibration and split sensitivity

- Repetitions: 30, with frozen seeds `2026071900` through `2026071929`.
- Resampling unit: complete trajectory.
- Split proportions, applied within pre-outcome strata where available: 20% tuning, 15% validation, 35% calibration, and 30% test.
- The same split is reused across methods and front ends within a repetition.
- Calibration-size sensitivity uses `n = 30, 60, 100, 150` trajectories **per calibration group** when the eligible pool permits it. Values larger than the available calibration set are reported as unavailable, not silently replaced.

For every repetition and calibration size, report simultaneous coverage, mean width, interval score, failure count, and the paired DAS-FC-minus-baseline effect. The summary reports the median, interquartile range, full range, the predeclared 2.5th--97.5th percentile stability interval, and the number of repetitions with the expected effect sign. Repetitions reuse one frozen trajectory pool and are therefore not treated as independent observations. A positive manuscript claim requires at least 27 of 30 repetitions to retain the expected sign and the stability interval to exclude zero.

## P3: closest uncertainty baselines

All methods share point estimates and eligible fields within each front end:

1. homoscedastic pair-max split conformal;
2. raw-local pair-max split conformal;
3. learned-unshrunk pair-max split conformal;
4. pointwise marginal split conformal, explicitly labelled as a non-simultaneous negative control;
5. classical Bonferroni Gaussian band;
6. two-stage classical max-t simultaneous band;
7. full grouped DAS-FC;
8. pooled/no-group DAS-FC ablation.

The max-t nuisance location and scale are fitted on tuning trajectories; its maximum standardized-residual quantile is calibrated on calibration trajectories. Bonferroni uses the same tuning estimates and fixed family-wise alpha. Methods that produce infinite or undefined bands remain reported.

## Outcomes and inference

The independent observation for all confirmatory inference is one trajectory. Primary outcomes are:

- trajectory-level full-field simultaneous coverage at nominal 95%;
- mean interval width;
- interval score;
- normal-displacement MAE and RMSE;
- failure/non-convergence rate;
- runtime and peak resident memory.

Paired method contrasts use trajectory-cluster bootstrap confidence intervals and paired sign-flip tests. Holm correction is applied to the three predeclared DAS-FC interval-score contrasts against homoscedastic, max-t, and raw-local bands. Domain/family summaries and worst-family results are mandatory.

## Formal stop conditions

The external extension does not support a positive Measurement claim if any of the following occurs:

1. test leakage or pseudoreplication is detected;
2. external known-group DAS-FC coverage is materially incompatible with 0.95, defined as a Wilson 95% interval excluding 0.95 or point coverage below 0.90;
3. the primary DAS-FC-minus-homoscedastic interval-score bootstrap interval crosses zero;
4. the effect disappears under two or more front ends;
5. the P2 sign-stability threshold is not met;
6. results become positive only after deleting failed trajectories, changing state pairs, or choosing a favorable split.

Negative results are retained and reported. No new module, threshold, dataset, or split may be introduced after viewing formal test outcomes without a separately authorized protocol amendment.
