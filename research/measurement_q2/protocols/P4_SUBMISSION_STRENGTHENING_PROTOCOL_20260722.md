# P4 submission-strengthening protocol

## Registration

- Protocol ID: `DASFC-P4-SUBMISSION-STRENGTHENING-V1`
- Freeze date: 2026-07-22
- Status at registration: `FROZEN_BEFORE_P4_EXECUTION`
- Execution environment: cloud only, under `/workspace/sound-barrier-measurement`
- Role: post-confirmatory sensitivity and diagnostic analysis; no new nominal-coverage claim

P4 does not alter the estimator, synthetic generator, calibration groups, split IDs,
shrinkage coefficients, conformal quantiles, fallback policy, Rockfall frame mapping,
or any P0--P3/D5 result. It reads frozen inputs and writes a new non-overwriting run.

## P4-A: Rockfall reference-tolerance sensitivity

The published TS60 coordinates remain the central sparse references. Frame-mapping
residuals are not treated as a GUM uncertainty budget. Two deterministic tolerance
brackets assess whether the finite event counts depend on treating each mapped
normal component as exact:

1. `target_projected_sum`: for each target and consecutive event, project the
   source- and target-epoch TS-to-TLS residual vectors onto the frozen canonical
   normal and sum their absolute magnitudes;
2. `event_rms_rss`: for each event, use the root-sum-square of the two epoch-level
   mapping RMS residuals as a common conservative tolerance.

For each original scalar interval and each tolerance bracket, report:

- central-reference containment (the frozen D5 result);
- interval intersection with the reference-tolerance set;
- full containment of the reference-tolerance set;
- lower and upper bounds on absolute normal error.

The independent reporting unit remains the scan-pair event (`n=3`). Targets are
nested locations. Only exact target and event counts are reported; no p-values,
confidence intervals, or population-coverage estimates are permitted.

## P4-B: synthetic-to-Rockfall observable diagnostic audit

The audit compares truth-free scan-pair diagnostics under the frozen
`cascade_strong` front end. The synthetic reference distribution contains all 240
tuning pairs from the four known formal-v1 domains. The external rows are the three
consecutive Rockfall events, using the frozen D5 crop, support mask, coordinate
frame, voxel size, and cascade estimator.

For every pair, let `D` be the Euclidean diagonal between the combined reference
and target 10th and 90th marginal percentiles. The seven fixed features are:

1. median raw scale divided by `D`;
2. 95th-percentile raw scale divided by `D`;
3. median match distance divided by `D`;
4. 95th-percentile match distance divided by `D`;
5. mean support probability;
6. valid fraction within the declared analysis mask;
7. 95th-percentile absolute normal displacement divided by `D`.

Synthetic medians and `1.4826 x MAD` values define robust standardisation. If a
MAD is numerically zero, the frozen fallback is `max(0.1 x sample SD, 1e-12)`.
Each Rockfall event receives a robust RMS z-distance, nearest synthetic robust
distance, per-feature empirical percentile, and count outside the synthetic
1st--99th percentile envelope. These are descriptive applicability diagnostics,
not a classifier, hypothesis test, or restored coverage guarantee. No threshold
is tuned from Rockfall outcomes.

## P4-C: deterministic full-field failure illustration

The illustrative case is selected mechanically from the frozen 240 known-domain
formal test pairs. Sort eligible case IDs lexicographically and take the first case
for which:

- pointwise point coverage is at least 0.90;
- pointwise simultaneous coverage is false; and
- DAS-FC simultaneous coverage is true.

The formal-v1 tuning set is rerun deterministically only to reconstruct the frozen
structural scale learner. The stored formal-v1 shrinkage coefficient and conformal
quantiles are then applied to the selected case. The reconstructed point coverage,
simultaneous-coverage indicators, and interval widths must match the frozen report
within `1e-9`; otherwise P4 fails and no figure is manuscript-eligible.

The figure shows `abs(error) / interval radius` at every valid field location for
the pointwise and DAS-FC intervals, with the failure threshold fixed at one. The
selection rule must be disclosed in the caption. The panel is illustrative and
does not add an independent replicate.

## Execution and acceptance

P4 passes only if all source/input hashes match the frozen configuration, all 240
synthetic tuning pairs and three Rockfall events complete, the field reconstruction
gate passes, all tabular/JSON/figure artifacts are finite, and PDF/SVG/600-dpi PNG
figures are written. A failed run is retained and is not silently retried.

Allowed claims are limited to: (a) sensitivity of the three finite Rockfall event
counts to two disclosed frame-mapping tolerance brackets; (b) observable diagnostic
distance between the formal synthetic tuning distribution and three Rockfall
events; and (c) a deterministic illustration of marginal-versus-simultaneous
coverage. P4 cannot establish dense physical truth, traceability, population
coverage, independent-apparatus replication, or railway-field performance.
