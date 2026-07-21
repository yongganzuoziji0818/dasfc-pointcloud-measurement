# SHREC'19 resolution-robustness exploratory protocol v1

## Material Passport

- Origin Skill: experiment-agent + experimental-design.
- Origin Mode: plan, followed by cloud-only run and validate.
- Origin Date: 2026-07-21.
- Verification Status: UNVERIFIED before execution.
- Version Label: `SHREC2019-RESOLUTION-ROBUSTNESS-V1`.
- Authorization: the user explicitly requested SHREC'19 algorithm execution
  through the point at which evidence can support a manuscript-results rewrite.
- Data boundary: official public SHREC'19 only; no confidential railway data.

## Research question

When four already frozen registration front ends are applied to the same 76
unique SHREC'19 deformation pair definitions, how sensitive are their observable
fit, rigid transform, and bidirectional consistency diagnostics to the public
high- versus low-resolution representations?

This is a finite-benchmark, unlabelled real-mesh robustness question. It is not a
test of displacement accuracy, correspondence accuracy, or conformal coverage.

## Units and dependence

The archive does not identify an independent physical-specimen count. Scan IDs
occur in multiple graph edges, and the high/low meshes are technical
representations of the same scan. Consequently:

- one unique `(test_set, source, target)` pair is the reporting row;
- high/low resolution and forward/reverse direction are repeated measurements;
- test set is a prespecified deformation stratum;
- 76 pair rows are not called 76 independent specimens;
- no p-values or population confidence intervals are computed;
- leave-one-scan-ID-out ranges are sensitivity analyses, not confidence
  intervals.

## Frozen design

The official duplicate `043,045` row in test set 0 is removed by retaining its
first occurrence. The resulting counts are 14, 26, 19, and 17 unique pairs.
Every pair is run using:

1. published coordinates (identity negative control),
2. Cascade-Strong,
3. multiscale trimmed point-to-point ICP,
4. robust point-to-plane ICP.

For Cascade-Strong, the implementation is the already used SHREC2020A front-end
path; only its estimated rigid rotation and translation enter this experiment.
Its panel-oriented displacement field is neither interpreted nor reported on
the arbitrary SHREC'19 hand/mannequin geometry.

Each method is evaluated at high and low resolution in both directions. For a
pair, a single common center and robust diagonal are computed from the
high-resolution source and target using their combined marginal 10th and 90th
percentiles. The identical affine normalization is applied to both resolutions
and directions. Native coordinates are never labelled millimetres.

The smoke run uses the first frozen unique pair in each stratum:
`0:040:043`, `1:000:001`, `2:030:011`, and `3:000:009`. It is an engineering
check only and cannot enter the manuscript. The formal run uses all 76 pairs.

## Endpoints

Co-primary descriptive endpoints are:

- absolute high-versus-low gap in symmetric mean nearest-surface fit;
- high-versus-low translation gap in the common dimensionless coordinate frame;
- high-versus-low rotation geodesic gap in degrees.

Secondary endpoints are bidirectional cycle defects; symmetric mean, median,
and p95 surface mismatch; exact test-set summaries; dataset-level Spearman
diagnostic associations; algorithm failure counts; and leave-one-scan-ID-out
sensitivity ranges. Published coordinates are not ranked on transform or cycle
stability because an identity transform produces zero gaps by construction.

## Resource and execution plan

The cloud resource snapshot reports 112 physical/224 logical cores, about 969 GB
available RAM, 892 GB available persistent storage, and Python 3.12.3 with
NumPy 2.4.6, SciPy 1.17.1, and Open3D 0.19.0. The experiment is CPU-bound and
does not use the L40S. Runs are sequential by pair with 16 cKDTree workers and
the activation script's eight-thread OMP/MKL cap.

Every smoke, formal, and reproduction run uses a new directory, dedicated log,
hard timeout, exit code, terminal sentinel, and source/config/data/output hashes.
There is no automatic retry. Deterministic reproduction must match bytewise
after excluding run-specific timestamps and durations, or match every scientific
metric exactly after canonicalization.

## Manuscript gate

The results may be integrated only after:

1. all 1,216 formal method-resolution-direction rows are attempted;
2. structural input/output validation passes;
3. a separately versioned deterministic reproduction completes;
4. all statistical fallacy categories are reviewed;
5. figures and prose say “resolution robustness” or “failure diagnostics,” not
   “real displacement validation.”

The direction of the result is not a gate. Negative or mixed findings remain
reportable if the frozen design and claim boundary are respected.
