# Supplementary methods and reproducibility specification

## S1. Scope and evidence boundary

This supplement specifies the implemented DAS-FC procedure. Confirmatory nominal-coverage results use synthetic scan pairs because known dense truth is observable there. The public 3DPrintedShapes v1 analysis uses single-state mesh--scan surface disagreement. The SHREC'19 analysis uses real deformation pairs but only observable cross-resolution fit and transform-consistency endpoints because the retrieved public archive contained no reference correspondence or displacement files. The ETH Rockfall Simulator analysis supplies a sparse independent-reference check for three consecutive physical events. It provides bounded physical corroboration, not dense-field truth, population nominal coverage, independent-apparatus replication, or railway-field performance.

## S2. Frozen point-estimation front end

All interval methods share `PCUDMFieldEstimator(mode="cascade", icp_iterations=14)`. Translation is initialized from the midpoint difference of the coordinate-wise 10th and 90th percentiles. Trimmed point-to-point ICP retains the lowest 82% of eligible match distances. Panel-local normal coordinates are fitted with a nine-term basis using ridge penalty 0.01 and four robust reweighting passes. Raw scale uses a 20-neighbour median absolute residual, 0.20 times match distance, and a 0.35 mm floor. Data units are millimetres.

## S3. Structural scale learner

At most 256 valid locations are sampled deterministically from each tuning pair. Each pair contributes total weight one. Seven predictors are used: log raw scale, log-one-plus match distance, support probability, absolute estimated displacement, two panel-local coordinates, and normalized panel index. The response is `log(abs(error) + 0.10 mm)`.

The implementation uses `HistGradientBoostingRegressor` with quantile loss at 0.80, learning rate 0.06, 160 iterations, 15 leaf nodes, minimum 30 samples per leaf, L2 regularization 1.0, and random state 20260719. Exponentiated predictions have a 0.10 mm floor.

## S4. Shrinkage and calibration

For each predeclared known domain, validation selects $\lambda$ from $\{0,0.25,0.50,0.75,1\}$ by minimizing $2q_{d,\lambda}\overline{s}_{d,\lambda}$; exact ties select the smaller value. Calibration uses the pair score

\[
R_j=\max_x\frac{|u_j(x)-\hat u_j(x)|}{\max\{s_j(x),10^{-6}\}}.
\]

For nominal level $1-\alpha$, the stored quantile is the $k_d$-th ordered score with

\[
k_d=\min\{n_d,\lceil(n_d+1)(1-\alpha)\rceil\}.
\]

Here $\alpha=0.05$, $n_d=150$, and $k_d=144$. The groupwise simultaneous-coverage statement requires exchangeability within the protocol-defined group and requires every scale and shrinkage choice to be fixed before calibration.

## S5. Operational domain rule

The calibration group is assigned from a registered acquisition protocol and its metadata, not from test error or post hoc visual inspection. The protocol must identify the scanner/acquisition condition and map it to a group represented in tuning, validation, and calibration. If no registered mapping exists, the pair is outside scope. The implementation fixes $\lambda=0$, uses the pooled homoscedastic calibration quantile, and labels the result empirical/outside guarantee. It does not automatically discover a new domain.

## S6. Selective reporting and AURC

The scale score is the mean full-field reported scale. The residual score is median nearest-match distance. The OOD score is the root-mean-square standardized distance of seven scan-level diagnostics from tuning means and standard deviations. The combined score is the maximum of validation-median/MAD standardized components.

For scan-pair risks $e_j$ ordered from low to high rejection score, selective risk at retained fraction $m/n$ is $m^{-1}\sum_{i=1}^{m}e_{(i)}$. AURC is computed by trapezoidal integration over retained fractions $m/n$. Lower AURC indicates that low-score retained cases have lower error. The combined score failed formal v1; scale-only was frozen and tested on a disjoint 480-pair v2 set. Neither rule has been validated for real double-epoch displacement.

## S7. Reproducibility contract

Tuning, validation, calibration, formal v1, rejector v2, grid sensitivity, and deformation-family stress use disjoint registered case IDs and seed families. Deterministic reproduction compares configuration, seeds, calibration states, 3,360 formal method rows, 480 rejector rows, summaries, and AURC values. Duration fields are excluded. The canonical scientific-output SHA-256 is `a9503bfa6791da55326c1650fd2d565463bd04dae893ee1732738312b08c7d26`.

## S8. Metrological terminology

The measurand is the normal-displacement field between two structural states at frozen valid field locations. In synthetic data, estimation error is observable because truth is known. DAS-FC produces split-conformal coverage intervals; these are not automatically GUM standard uncertainty, expanded uncertainty, or a traceability statement. In the public single-state data, the endpoint is symmetric surface disagreement because no double-epoch displacement truth exists. In the Rockfall study, Leica TS60 prism displacements are independent sparse references for scalar normal-component inclusion and 3D target error; they do not define a dense truth field or a calibrated three-dimensional vector region.

## S9. SHREC'19 resolution-robustness protocol

The official SHREC'19 archive (DOI `10.17035/d.2019.0072003316`; SHA-256 `5e1a8dc86701c9bde0aeeee17aa4ffdbfc8b3bf1e344229bf2546e0476d541fe`) was audited before use. It contained 50 high-resolution and 50 low-resolution OBJ meshes, four pair lists, and no reference correspondence or displacement files. One repeated `043,045` row in test set 0 was deduplicated by first occurrence, leaving 76 unique test-set/source/target graph edges. The counts by test set were 14, 26, 19, and 17. Native coordinates were not assigned physical units.

For each graph edge, one centre and robust diagonal were computed from the combined 10th and 90th marginal quantiles of the high-resolution source and target. This affine normalization was applied unchanged to high and low resolution and to forward and reverse directions. Published coordinates, Cascade-Strong, multiscale trimmed point-to-point ICP, and robust point-to-plane ICP were run in a frozen order. The 76 pairs x 4 front ends x 2 resolutions x 2 directions produced 1,216 formal rows. Failures were retained by protocol; none occurred.

Pair-level descriptive endpoints were the mean absolute high--low gap in symmetric mean nearest-surface fit across directions, the mean high--low translation-vector norm, and the mean geodesic rotation gap. Bidirectional cycle defects, surface-fit summaries, diagnostic Spearman coefficients, exact test-set strata, and leave-one-scan-ID-out median ranges were secondary. No $p$-values or population confidence intervals were computed because scan IDs recur across graph edges and independent physical-specimen identity is undocumented. The identity control's zero transform and cycle gaps are mathematical consequences and were not ranked as registration stability.

An initial v1 run failed the predeclared exact-reproduction gate because parallel Open3D reductions differed at approximately $10^{-15}$, although all row keys and inputs matched. That failure is retained. Protocol v1.1 changed only execution determinism by setting OMP, MKL, OpenBLAS, NumExpr, and Open3D thread counts to one; algorithms, pairs, and endpoints were unchanged. Two independent v1.1 executions matched exactly after canonical removal of timestamps, durations, and resolved roots. The canonical execution and analysis SHA-256 values were `86737dee6e23c90e67488b05512f270493474e93998a9b0ba488f85b755c3376` and `985c6008644db9983a361fc35aedc45616dd2485aac7e11acb9640ec03174385`.

## S10. ETH Rockfall sparse physical-reference protocol

The ETH Rockfall Simulator data were frozen at revision `42a3947d960c8163157c915dea847cda96904a3d`. The controlled sequence contains four successive Leica RTC360 epochs and five Leica TS60 prism targets. T0 was treated as stable for frame verification; T1--T4 were the displacement-reference targets. Before any algorithm output was accessed, a rigid TS-to-TLS mapping was selected at E0 and checked at E1--E3. Epoch root-mean-square mapping residuals were 1.951, 2.099, 1.808, and 2.119 mm; maximum residuals were 3.442, 3.639, 2.440, and 2.778 mm. The T0 TLS drift was 1.952 mm. Mapping residuals were retained as frame-quality diagnostics and were not subtracted from prediction errors.

The D5 formal study evaluated E0--E1, E1--E2, and E2--E3 with Cascade-Strong, multiscale trimmed point-to-point, and robust point-to-plane front ends. The algorithm protocol, fallback behavior, thresholds, dataset revision, frame mapping, and three P1 calibration reports were hash-locked. Rockfall data were not used for tuning or calibration. All events were assigned to the predeclared unseen external physical domain and therefore used the pooled homoscedastic fallback outside the formal known-domain guarantee.

For each event and front end, the primary physical endpoints were simultaneous inclusion of all four TS60 normal-component references, the number of covered targets, mean and maximum 3D vector error, normal-component MAE, mean interval width, and mean interval score. The scan-pair event was the independent reporting unit ($n=3$). Four target rows per event were nested validation locations. Analysis reported event-level medians, ranges, and exact counts only; it computed no $p$-values, significance marks, or population confidence intervals. A complete 11-type statistical-fallacy scan explicitly checked pseudoreplication, selection, multiplicity, and causal-language risks.

The first D5 formal attempt failed during CSV serialization because result rows had heterogeneous optional fields; its failure sentinel and manifest are retained. Attempt 02 changed only CSV field-name construction to the ordered union across rows. It completed all nine event--frontend and 36 target rows. A separate non-overwriting analysis run verified the D5 manifest, table schemas, row keys, JSON/CSV coverage consistency, dataset revision, no-tuning flag, and claim-boundary flags before generating descriptive tables and Figure 7.
