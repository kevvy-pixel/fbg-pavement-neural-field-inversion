# Experiment packages

This directory preserves the standalone MATLAB packages used after the original
Step19 model. They are intentionally kept as versioned packages because helper
functions and configurations changed between stages.

## Layout

- `ablations/`: negative, diagnostic, low-budget, or superseded experiments.
- `calibration/`: the retained sparse-calibration workflow.

The main zero-shot model remains
`Step19_BoundaryAware_FiLM_AD_DNF.m` in the repository root. The experiment
packages do not replace that backbone.

## Final decision

- No measurements from a new condition: use Step19-SOTA.
- Six sparse measurements, accuracy first: use Step19-SOTA + Ridge-6.
- Six sparse measurements, risk first: use Step19-SOTA + Gated Ridge-6.
- Residual-FiLM, the spatial boundary expert, Ridge-4, and the four-point gate
  are retained as ablations.

See [`../docs/EXPERIMENT_INDEX_CN.md`](../docs/EXPERIMENT_INDEX_CN.md) for the
version-to-directory mapping, entry points, and conclusions.

## Data policy

`Response_Matrix_Table.csv` is private and ignored at every directory depth.
Generated `outputs/`, result directories, model snapshots, logs, and demo data
are also ignored. The small `LEGACY_STEP19_REFERENCE.csv` files are published
because they contain only reference metrics, not the private response matrix.
