# ETH Rockfall Simulator physical-validation protocol

## Registration

- Protocol ID: `ROCKFALL-PHYSICAL-V1`
- Status: `FROZEN_BEFORE_DATA_AUDIT_AND_ALGORITHM_EXECUTION`
- Freeze date: 2026-07-21
- Dataset revision: `42a3947d960c8163157c915dea847cda96904a3d`
- Dataset license: CC BY 4.0
- Scientific role: zero-cost real, physical, dual-epoch TLS validation with an
  independent Leica TS60 reference system.

This protocol is an additive branch. It does not modify, replace, tune, or
rerun the frozen P0--P3 synthetic evidence.

## Data and epoch map

The publisher declares four physical epochs. The only permitted epoch map is:

| TS epoch | TLS epoch |
|---|---|
| E0 | epoch_1 |
| E1 | epoch_2 |
| E2 | epoch_3 |
| E3 | epoch_4 |

The three consecutive events are interpreted as publisher-labelled physical
motions: vertical translation, translation plus rotation, and rotation. The
three additional non-consecutive pairs may be reported only as dependent
secondary contrasts. All pairs belong to one apparatus and one acquisition
campaign.

## Frozen gates

1. `D0_INTEGRITY`: exact Git revision, complete LFS materialization, per-file
   SHA-256, license and documentation captured.
2. `D1_LAYOUT`: four readable TLS epochs and TS coordinates for T0--T4 at
   E0--E3. Empty/non-finite point clouds or ambiguous epoch identity fail.
3. `D2_REFERENCE`: T0 behaves as a stable control; T1--T4 support a rigid
   transform for each consecutive event. The reference residuals and all
   anomalies are retained.
4. `D3_FRAME`: the relationship between TLS and TS coordinates is either
   documented or empirically recoverable without using the tested algorithm's
   displacement output. If this gate fails, vector/component-wise comparisons
   and dense reference propagation are prohibited.
5. `D4_SMOKE`: one predeclared consecutive event completes with frozen code and
   produces finite outputs without changing any estimator parameter.
6. `D5_FORMAL`: all three consecutive events and all frozen comparators are
   attempted once. Failed events remain outcomes. No silent retry is allowed.

## Estimands and endpoints

### Confirmatory real-data endpoints

The confirmatory unit is a physical scan-pair event, not a point, patch, prism,
or algorithmic correspondence.

For each of the three consecutive events, report:

1. availability and identity of the independent TS reference;
2. absolute displacement-vector error at each legitimately mapped T1--T4
   reference location, in millimetres, if `D3_FRAME` passes;
3. event-level maximum reference-location error;
4. whether the frozen interval covers all four reference vectors
   simultaneously;
5. mean interval width and interval score at the four reference locations;
6. abstention/fallback status and the reason code.

If `D3_FRAME` does not pass, only coordinate-frame invariant quantities may be
reported: rigid rotation angle, inter-target distance preservation, Kabsch
residual, and displacement-magnitude summaries that can be proven invariant
for the evaluated correspondence. Such outputs are exploratory and cannot be
substituted for the confirmatory vector endpoints.

### Secondary endpoints

- point-estimation performance of the frozen `cascade_strong` frontend;
- frozen classical registration comparators already used in P3;
- resolution/downsampling sensitivity at predeclared voxel sizes;
- stable-region false-motion and moving-region detection diagnostics;
- all six epoch pairs as dependent descriptive contrasts.

No population p-values are allowed. With one apparatus and three consecutive
events, report event rows, medians/ranges, exact counts, and Wilson intervals
only as descriptive finite-sample summaries. Points and prisms are nested
within event and must never inflate the independent sample size.

## Calibration and model-freeze boundary

- No Rockfall epoch may train, tune, select, shrink, or recalibrate the point
  estimator or uncertainty model.
- Synthetic P0--P3 choices, calibration quantiles, fallback rules, grid rules,
  and comparator parameters remain frozen.
- Rockfall is an external physical test branch. Any unit conversion or
  coordinate transform must be determined from publisher metadata, stable
  controls, or independently identified reference geometry before viewing
  algorithm errors.
- If the frozen method cannot operate on the released geometry without a new
  scientific choice, the branch records `NOT_EVALUABLE_UNDER_FROZEN_METHOD`;
  it is not repaired post hoc.

## Claim boundary

Permitted after all relevant gates pass:

- the study includes real physical dual-epoch TLS measurements;
- Leica TS60 observations provide an independent sparse reference chain;
- real-event point accuracy and simultaneous coverage at the four reference
  targets were evaluated, if and only if `D3_FRAME` passes;
- failure, fallback, and domain-shift behavior was measured on the apparatus.

Prohibited:

- claiming three events are three independent structures;
- claiming dense real non-rigid structural displacement truth unless the
  release itself supplies and validates such truth;
- treating millions of points as independent coverage trials;
- claiming confirmatory real-world nominal 95% simultaneous coverage from
  this one apparatus;
- claiming railway sound-barrier field validation;
- hiding a failed coordinate-frame or target-mapping gate.

## Manuscript decision rule

The real-data hard absence is considered closed only if `D0`--`D3` pass and at
least one consecutive scan pair yields an independently referenced physical
displacement endpoint. The broader nominal-coverage claim remains
simulation-confirmed and must be labelled as such. If `D3_FRAME` fails, the
dataset remains useful for real dual-epoch robustness but does not close the
reference-truth absence.

