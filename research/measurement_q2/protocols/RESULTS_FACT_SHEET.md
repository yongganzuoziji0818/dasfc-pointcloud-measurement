# Frozen results fact sheet

## Confirmatory synthetic evidence

| Item | Frozen result | Allowed interpretation |
|---|---:|---|
| Independent known test units | 240 scan pairs, 4 domains | Confirmatory unit is the scan pair |
| DAS-FC minus homoscedastic interval score | -2.1348 mm; 95% CI [-2.3617, -1.9394] | DAS-FC improves coverage-efficiency score under shared point estimates |
| Paired effect and multiplicity | dz=-1.136; Holm p=1.00e-44 | Large paired effect; use with bootstrap/randomization evidence |
| DAS-FC simultaneous coverage | 0.958; Wilson 95% CI [0.925, 0.977] | Known calibrated domains only |
| DAS-FC / homoscedastic width | 11.173 / 13.351 mm | Improvement is not caused by wider intervals |
| Domain consistency | 4/4 domains favor DAS-FC | No observed aggregate sign reversal |
| Pointwise conformal baseline | point coverage 0.946; whole-field coverage 0.008 | Marginal point calibration does not solve simultaneous field coverage |
| Point-estimation runtime | mean 0.208 s; p95 0.250 s per scan pair | Algorithmic benchmark on cloud CPU, excluding I/O/UI |
| Non-convergence | 0/3360 method rows | No formal cases removed |

## Robustness and negative results

| Audit | Result | Boundary |
|---|---|---|
| Independent rejector v2 | scale AURC 2.2115; lower than residual 2.5359, OOD 2.5152, combined 2.4686 | Confirms ranking, not a universal risk oracle |
| Four-grid sensitivity | all 12x9--30x22 gates pass; coverage 0.950--0.971 | Supports discretization stability over tested grids |
| Default unseen domains | fallback coverage 0.829 vs 0.275; score 30.984 vs 17.675 | Coverage recovery has severe width/score cost |
| `rbf_kink` family | MAE 1.590 mm, 0 failures; fallback coverage 0.971 vs 0.688 | Mixed; fallback score worsens by 8.927 mm |
| Combined rejector v1 | AURC 2.4100 vs best single 2.2056 | Failed hypothesis; not a contribution |
| Exact reproduction | canonical scientific SHA-256 matches exactly | Durations excluded from deterministic hash |

## Public real-point-cloud evidence

- Corpus: 3DPrintedShapes, 38 physical specimens, three devices, 114 acquisitions; one structural corpus.
- Published-coordinate mesh/scan disagreement means: iPad 9.622 mm, FARO 7.491 mm, Creaform 5.551 mm; Friedman chi-square 63.0, Kendall W 0.829.
- Unmodified transfer: multiscale ICP 4.222 mm versus Cascade-Strong 4.539 mm mean surface disagreement. Cascade minus ICP = +0.316 mm, specimen-cluster bootstrap 95% CI [+0.185, +0.476].
- Frozen risk diagnostics correlate with real surface disagreement, but match-q95 ranks risk better than learned scale (AURC 2.859 vs 2.972; difference +0.113, 95% CI [+0.070, +0.161]).
- These data support scanner-domain shift and failure-ranking evidence only. They do not contain double-epoch displacement truth.

## Prohibited numerical substitutions

Do not treat field locations as sample size; do not call 114 acquisitions independent specimens; do not merge raw and processed representations; do not report real surface disagreement as deformation error; and do not replace the negative real ICP comparison with the favorable synthetic comparison.

