# Final integrity report — P4 submission package

Date: 22 July 2026  
Decision: **PASS — ready for author approval and journal upload**

## Audited package

- Main manuscript: `anonymous_manuscript.tex` and the 35-page anonymous PDF.
- Supplement: `Supplementary_Methods.tex` and the 5-page anonymous PDF.
- Figures: eight vector PDFs, each cited and rendered.
- Bibliography: 26/26 records cited; 26/26 records independently rechecked; 20/20 citation-context clusters supported.
- Statistical claims: 113/113 registered claims resolve to frozen evidence.
- Submission QA: every technical and administrative check passed; no unresolved placeholders or identity leakage in the anonymous manuscript.

Key machine-readable receipts:

- `manuscript/integrity_audit_p4_archive_final_20260722.json` — SHA-256 `f00c12c94e341d9d5b8fb5f04ace4ef04c37ddc8dd106e403357cda41e2e6f52`.
- `submission/reference_audit_p4_final_20260722.json` — SHA-256 `9ffddfb009383c53a9caa3b9a925f1e3398622c6341f0a8770bd3877bfc48fba`.
- `submission/submission_qa_p4_archive_final_20260722.json` — SHA-256 `8cf821eec4031b38ccd351dd4c6d3742955bb123819298d68ceaf62698c42582`.
- `evidence/formal/p4_submission_strengthening_attempt02_20260722/P4.ok` — immutable successful-run sentinel; the earlier bootstrap failure remains preserved separately.

## Seven-mode AI research integrity audit

| Mode | Verdict | Evidence and boundary |
|---|:---:|---|
| 1. Fabricated or unexecuted results | CLEAR | P0–P4 result claims resolve to saved reports, CSV/JSON outputs, exits, audit reports, and hash manifests. P4 Attempt 02 has zero execution and audit exit codes. Failed and superseded attempts were retained rather than rewritten as successes. |
| 2. Hallucinated or mismatched references | CLEAR | All 26 bibliography records were checked against primary publisher, proceedings, standards-body, repository, OpenReview, arXiv, or author-institution sources. All 20 citation-context clusters were reviewed for semantic support. No unresolved record remains. |
| 3. Unsupported numerical claims | CLEAR | The claim ledger and integrity audit close 113/113 registered statistics. P4 diagnostic distances, tolerance counts, and the selected 624-location field reconstruct from the saved P4 report and source tables. |
| 4. Statistical or unit-of-analysis inflation | CLEAR | The scan pair is the independent unit. Grid locations, SHREC graph edges, Rockfall prisms, repeated splits, and repeated scanner observations are explicitly treated as nested, dependent, or descriptive as appropriate. No target-level pseudo-replication is used. |
| 5. Hidden negative or failed evidence | CLEAR | The failed combined rejector, public-data loss to ICP, mixed SHREC ordering, Rockfall Cascade failure, P4 bootstrap precondition failure, and domain-shift limitations remain disclosed. No failed attempt was overwritten. |
| 6. Method/result drift | CLEAR | Frozen configs, estimators, thresholds, seeds, report hashes, and P4 input manifests match the described methods. P4 added only post-confirmatory applicability and reference-sensitivity analyses; it did not retune the estimator or calibration policy. |
| 7. Overclaiming beyond evidence | CLEAR | The confirmatory guarantee is restricted to predefined synthetic groups. Public single-state scans, SHREC, and Rockfall are labelled consistency, stress-test, or sparse-reference evidence. The paper expressly excludes dense physical truth, population coverage, railway deployment, GUM uncertainty budgets, and universal robustness. |

## Figure and table fidelity

| Item | Visual/table claim | Trace and verdict |
|---|---|---|
| Figure 1 | DAS-FC workflow and applicability boundary | Method specification and fallback contract; caption states the guarantee boundary. PASS |
| Figure 2 | Known-domain coverage, width, score, and domain effects | Formal-v1 result tables and paired statistics; identical point estimates disclosed. PASS |
| Figure 3 | Grid, unseen-domain, and selective-risk sensitivity | Frozen grid/v2/rbf-kink reports; stress outcomes are not labelled nominal guarantees. PASS |
| Figure 4 | Three-device single-state consistency | Public 3DPrintedShapes reports; caption excludes double-epoch truth and coverage validation. PASS |
| Figure 5 | P0b–P3 extensions | Frozen frontend, split, baseline, and marker reports; repeated-pool and single-object limits disclosed. PASS |
| Figure 6 | SHREC'19 cross-resolution stress test | 1,216 rows and 76 pair rows; no false independence, significance, or truth claim. PASS |
| Figure 7 | Rockfall sparse reference study | Three scan-pair events with four nested prisms; caption excludes dense truth and population inference. PASS |
| Figure 8 | P4 applicability, tolerance sensitivity, and field illustration | P4 Attempt 02 reports and deterministic selection ledger; panels are explicitly descriptive. PASS |
| Table 1 | Known-domain interval comparison | Formal-v1 aggregate metrics under common point estimates. PASS |
| Table 2 | Prospective evidence gates | P0b–P3 frozen protocols and observed gates; allowed inference is separately stated. PASS |
| Table 3 | Target-level novelty comparison | Supported by conformal and registration references; presented as conceptual comparison, not an empirical ranking. PASS |

## Boundary of this audit

This check verifies disclosure and claim-to-provenance fidelity. It does not judge whether the experiment was correctly designed, run, statistically adequate, or reproducible by ARS.

The remaining scientific limitation is deliberate and visible: no public dataset used here supplies dense independently measured real double-epoch displacement truth. The submission is therefore ready as a simulation-led measurement-method paper with bounded physical corroboration, not as a deployment-validation paper.
