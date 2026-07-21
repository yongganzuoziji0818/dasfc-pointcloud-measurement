# ETH Rockfall D4/D5 frozen algorithm protocol

## Registration

- Experiment ID: `ROCKFALL-DASFC-PHYSICAL-V1`
- Freeze date: 2026-07-21
- Parent protocol: `ROCKFALL-PHYSICAL-V1.1`
- D3 input: `ROCKFALL-FRAME-MAPPING-V1`, passed before this freeze
- Status: `FROZEN_BEFORE_ALGORITHM_EXECUTION`

Rockfall is an unseen external physical domain. It does not train, tune,
recalibrate, or select any estimator, scale model, quantile, crop after outcome
inspection, or front end.

## Fixed geometric preparation

1. Build one canonical coordinate frame from the E0 mapped T1--T4 TLS marker
   centres. The horizontal axis joins the T1/T4 and T2/T3 column midpoints; the
   vertical axis joins the T3/T4 and T1/T2 row midpoints after orthogonalisation;
   the normal completes a right-handed frame.
2. Apply that same frame to every TLS epoch and convert metres to millimetres.
3. Retain points within 120 mm of the E0 marker rectangle in canonical X/Z and
   within 100 mm of the marker plane. Deterministically voxelise at 5 mm.
4. The moving-panel region is the E0 marker rectangle expanded by 50 mm in X/Z.
   Points outside it but inside the context crop are support candidates. No
   target-epoch truth or estimator error changes this mask.
5. At each prism, interpolate the predicted normal field from the 16 nearest
   moving-region reference points in X/Z using inverse-distance weights.

## Frozen estimators and uncertainty

The three front ends are exactly the P1 frozen set:

1. `cascade_strong`;
2. `multiscale_trimmed_ptp` followed by the same field fit;
3. `robust_ptpl` followed by the same field fit.

Estimator parameters remain `mode=cascade`, 14 ICP iterations, trim fraction
0.82, ridge 0.01, 20 scale neighbours, and 0.35 mm scale floor. Multiscale
thresholds remain 100, 30, and 10 mm.

Because Rockfall is unseen, each front end uses its already frozen P1
homoscedastic pooled fallback quantile and the event's median raw scale. The
three quantiles are read only from hash-locked P1 reports. No Rockfall result
may alter them.

## D4 smoke and D5 formal execution

- D4 is one engineering-only E0--E1 `cascade_strong` run. It passes only if the
  crop contains at least 1,000 moving and 1,000 support reference points, the
  estimator converges, all four prism outputs are finite, and all declared
  artifacts are written. D4 outcomes are not manuscript evidence.
- After D4 passes, D5 attempts all three consecutive events with all three
  front ends exactly once. Every failure remains in the denominator. No silent
  retry or threshold change is allowed.

## Endpoints and claim boundary

For T1--T4 and each event/front end, report the independently mapped TS60 3D
reference vector, predicted 3D vector, Euclidean vector error, signed canonical
normal error, fallback interval radius, target-wise coverage, and event-level
simultaneous coverage across all four targets. Also report event maximum vector
error, normal MAE, mean interval width, interval score, valid fraction, runtime,
and failure state.

The conformal output is a scalar normal-displacement interval. It is not a
calibrated 3D vector region. Three events from one apparatus support finite-case
physical validation only; they cannot establish population-level nominal 95%
coverage. No p-values or pseudo-replication over points/prisms are permitted.

