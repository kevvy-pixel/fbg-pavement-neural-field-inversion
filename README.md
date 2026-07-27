# FBG Pavement Neural-Field Inversion

MATLAB code for reconstructing continuous pavement response fields from FBG
measurements with coordinate neural networks, automatic-differentiation strain
supervision, FiLM conditioning, and boundary-aware features.

中文项目名称：**基于运动学约束神经场的 FBG 路面响应反演与早期塌陷预警**

This repository extends the data-processing and response-modeling workflow in
[`fbg-multilayer-pavement-modeling`](https://github.com/kevvy-pixel/fbg-multilayer-pavement-modeling).

## Recommended model

The recommended entry point is:

```matlab
run('run_main.m')
```

This runs `Step19_BoundaryAware_FiLM_AD_DNF.m`, the selected main model.
Step19 uses a case-wise split to test cross-condition generalization and adds:

- a displacement neural field;
- strain supervision through second-order automatic differentiation;
- FiLM conditioning on loading conditions;
- explicit load-to-boundary distance features;
- moderate load and boundary sample weighting.

The learned displacement is a **displacement proxy**, not a directly measured
absolute displacement. Absolute displacement calibration requires independent
LVDT, DIC, or displacement-sensor measurements.

## Code map

| Script | Role | Split | Status |
|---|---|---|---|
| `Step13_NN_FBG_Field_Reconstruction.m` | Direct strain/response MLP | case | Baseline |
| `Step14_Displacement_Neural_Field_AD_Strain.m` | Displacement field with AD strain supervision | case | Exploratory |
| `Step15_Asphalt_LoadBalanced_Displacement_Neural_Field.m` | Asphalt-only, load-balanced AD-DNF | case | Ablation |
| `Step16_Asphalt_EdgeAnchored_AD_DNF.m` | Weak edge anchoring | case | Ablation |
| `Step17_FiLM_Conditioned_AD_DNF.m` | FiLM-conditioned AD-DNF backbone | case | Method backbone |
| `Step18_PointMode_Ablation_FiLM_AD_DNF.m` | Within-condition spatial interpolation | point | Diagnostic |
| `Step19_BoundaryAware_FiLM_AD_DNF.m` | Boundary-aware FiLM AD-DNF | case | **Main model** |
| `Step21_BoundaryResidualExpert_FiLM_AD_DNF.m` | Boundary residual expert | case / optional mixed-edge | Negative ablation |

The iteration report also discusses Step20A and Step20B diagnostics. Their
standalone source files are not part of the supplied code snapshot, so this
repository does not claim to reproduce those two runs.

## Requirements

- MATLAB with script-local-function support;
- Deep Learning Toolbox;
- sufficient memory for second-order automatic differentiation.

The experiments use random seed `2026`. The supplied scripts run on CPU unless
they are explicitly adapted to move parameters and batches to `gpuArray`.

## Private data

The experimental dataset is intentionally excluded. Place the authorized file
below in the repository root:

```text
Response_Matrix_Table.csv
```

Validate its schema before training:

```matlab
run('check_data_schema.m')
```

The required columns and conventions are documented in
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

## Running experiments

Run MATLAB from the repository root. For the main model:

```matlab
run('run_main.m')
```

For the direct-MLP baseline:

```matlab
run('Step13_NN_FBG_Field_Reconstruction.m')
```

For an ablation, run the corresponding `Step*.m` file directly. Every script
creates its own result directory containing figures, tables, models, and logs.
These generated artifacts are ignored by Git.

## Reported model selection

The supplied iteration report identifies Step19 as the most stable main model:

- all-data \(R^2 = 0.9164\);
- non-extreme-boundary \(R^2 = 0.9586\);
- non-extreme-boundary RMSE = \(40.63\,\mu\varepsilon\).

The 95 cm extreme-boundary condition remains a known extrapolation limitation.
Step21 only slightly improved that condition while reducing overall
generalization, so it is retained as a negative ablation rather than the main
model.

See [`docs/MODEL_EVOLUTION.md`](docs/MODEL_EVOLUTION.md) for the method history
and interpretation boundaries.

## Data and licensing

The source code and experimental dataset are separate works. The dataset is
maintained privately and is not distributed through this repository.

No open-source license has been selected for this code snapshot. Public
visibility alone does not grant permission to reuse or redistribute the code.
