# CrisisDynNet core analysis code — 18 November 2026

This source snapshot contains the core trajectory, prediction and subtype-dynamics implementation for the university-student distress and crisis-risk study. The seven source files listed in `SOURCE_MANIFEST.json` are byte-for-byte copies of the supplied analysis package. Packaging adds documentation and a data-schema export; it changes no fitted model, analysis result or algorithm.

## Contents

| File | Role |
| --- | --- |
| `common/crisisdynnet.py` | Daily panels, trajectory signatures, dynamic mixture model, prediction windows, gated fusion GRU |
| `common/baselines.py` | Clustering and prediction comparison methods |
| `pipeline/08_cohort_framework.py` | Cohort analysis, participant-grouped evaluation, ablations and warning summaries |
| `pipeline/14_subtype_dynamics.py` | Subtype profiles, turning points and scenario-specific warning lead times |
| `config.py` | Paths, seed and analysis settings |
| `tools/check_data.py` | Input table/schema checks |
| `requirements.txt` | Original dependency declarations (versions are mostly unpinned) |
| `DATA_SCHEMA.json` | Input table names, required columns and range checks exported by the supplied checker |
| `SOURCE_MANIFEST.json` | SHA-256 hashes of the seven original files |

## Setup and execution

Use a fresh checkout or extracted snapshot and Python 3.11. From this directory:

```powershell
python -m pip install -r requirements.txt
$env:NHB_DATA_DIR = 'D:\path\to\authorized_cohort_tables'
python tools/check_data.py --schema
python tools/check_data.py
python pipeline/08_cohort_framework.py
python pipeline/14_subtype_dynamics.py
```

`NHB_DATA_DIR` selects the directory containing the study's eleven CSV tables. Without it, the default is `../data` relative to this code directory. The second analysis consumes the first analysis's output. Results, figures, caches and logs are written under this directory's `outputs/`. Running the analyses again in the same directory can overwrite generated outputs; use a fresh copy for a separate run.

The original requirements also list packages used by the larger historical pipeline. PyTorch and hmmlearn support neural-network and HMM comparisons. The input checker validates all eleven tables. The cohort analysis loads nine tables; subtype-dynamics analysis loads six and consumes the earlier results. `DATA_SCHEMA.json` is the machine-readable input contract.

## Data and interpretation

Participant-level mental-health and physiological data require the study's controlled-access process described in the manuscript. This source release contains code and schema metadata only. The aggregate figure workbooks at the repository root do not replace participant-level analysis inputs.

The snapshot implements retrospective internal evaluation. Daily normalization and trajectory features use full follow-up information, and some preprocessing is fitted before grouped prediction folds. The warning threshold is estimated from pooled held-out non-event scores. Interpretation of its performance must reflect these implementation choices; prospective deployment requires time-restricted inputs and training-fold preprocessing and threshold selection. The four-subtype analysis is fixed separately from the reported model-selection score. Several estimators retain their original stochastic defaults and dependencies are not fully version-locked.

Historical synthetic-cohort generators, truth caches, participant-level inputs, fitted outputs and manuscript generators are outside this core release. The preserved analysis contains optional truth-cache checks; a fresh copy without those caches skips them. The current release has been checked for Python syntax, imports and schema export; a full cohort analysis has not been rerun for publication of this snapshot. This upload records source availability on 18 November 2026 and does not establish a prospective preregistration date.

## Version identification

Use the fixed Git commit URL supplied with the submission's code-availability record. `SOURCE_MANIFEST.json` supports independent comparison with the original seven files.
