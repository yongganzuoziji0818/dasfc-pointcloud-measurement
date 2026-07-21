# DAS-FC point-cloud measurement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21477662.svg)](https://doi.org/10.5281/zenodo.21477662)

This repository is the public research package for **domain-adaptive shrinkage full-field conformal calibration (DAS-FC)**, a scan-pair-aligned interval procedure for structural point-cloud normal-displacement fields.

The central measurement target is simultaneous coverage of every frozen valid location in one field produced from one scan pair. The uncertainty methods share the same point estimator so interval comparisons are not confounded by point-estimation accuracy.

## Evidence scope

- Synthetic known-domain experiments establish the nominal full-field coverage and efficiency results.
- Unseen-domain, grid, deformation-family, repeated-allocation, and registration-frontend studies test robustness and failure behavior.
- Public 3DPrintedShapes and SHREC scans test device and cross-resolution behavior without claiming displacement truth.
- Three ETH Rockfall Simulator scan-pair events provide independent Leica TS60 references at four sparse targets per event.

The Rockfall evidence closes the absence of physical displacement truth only at sparse endpoints. It does **not** establish dense-field truth, population-level 0.95 coverage, independent-apparatus replication, railway-field validation, or a GUM-compliant uncertainty budget.

## Repository layout

```text
research/measurement_q2/
  pcudm/            Core estimator, calibration, scale model, and baselines
  scripts/          Experiment, analysis, audit, and validation programs
  configs/          Frozen JSON protocols
  protocols/        Preregistered and algorithm protocols
  results/          Compact synthetic and public-data result tables
  reproducibility/  Environment, manifests, and manuscript QA reports
  manuscript/       Manuscript source, references, and seven figures
  tests/            Core unit tests
```

Raw public datasets and full run directories are intentionally excluded. Dataset locations, fixed revisions, checksums, and allowed inferences are documented in `protocols/` and the manuscript.

## Environment

Python 3.12 is the frozen environment family.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s research/measurement_q2/tests -v
```

On Windows, activate with `.venv\Scripts\activate`.

## Synthetic smoke run

```bash
python research/measurement_q2/scripts/run_das_fc_formal.py \
  --config research/measurement_q2/configs/formal_v1.json \
  --output-dir runs/formal_smoke \
  --smoke
```

Formal configurations are frozen evidence records. New experiments should write to new output directories and must not overwrite archived results.

## Data and confidentiality

This package contains no Railway Sciences confidential data, private point clouds, cloud-server addresses, credentials, private keys, or local author-administration records. Public datasets retain their original licenses and must be downloaded from their official sources.

## Citation and archive

Citation metadata are provided in `CITATION.cff`. Release `v1.0.0` is permanently archived at https://doi.org/10.5281/zenodo.21477662.

## License status

No open-source license has yet been granted for this repository. Copyright remains with the authors. Public visibility and scientific citation do not by themselves grant permission to redistribute or create derivative software. Dataset licenses are separate from the code license.
