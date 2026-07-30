# Package information

- Package: `FBG_NeuralField_ResidualMoE_Package`
- Public repository inspected: `kevvy-pixel/fbg-multilayer-pavement-modeling`
- Repository commit inspected: `17a78be9d0c6ab13fc428dee221c4ee35e276735`
- Input interface: `Response_Matrix_Table.csv` produced by
  `run_02_build_response_matrix.m`
- Original repository files modified: none
- Private experimental data included: none

## Implemented stages

| Stage | Purpose |
| --- | --- |
| `00_data_validation` | Validate columns, aggregate repeated positions, define the case split, and compare measured 90/95 cm shapes |
| `01_plain_film` | Plain FiLM-conditioned displacement neural field baseline |
| `02_residual_film` | Residual MLP blocks with FiLM conditioning |
| `03_spatial_edge_expert` | Coordinate-aware soft gate and local edge residual expert |
| `04_fewshot_95cm` | Freeze the global field and calibrate the edge adapter with selected 95 cm points |
| `05_comparison` | Compare R² and RMSE across all models and calibration sizes |

## Important methodological boundary

The public repository does not contain the private Step19-Step21 neural-field
source files. This package therefore implements the reported ideas as a new,
standalone extension compatible with the public response-table format. It is
not a byte-for-byte modification of the unavailable private scripts.

## Verification performed before packaging

- Parsed every `.m` file with a MATLAB tree-sitter grammar;
- Confirmed zero parser error nodes;
- Checked the nested-structure parameter design against MATLAB documentation
  for `dlgradient`, `adamupdate`, and `dlupdate`;
- Confirmed all input/output paths are relative to the package root unless a
  full CSV path is explicitly supplied;
- Added automatic writable-workspace selection for read-only extraction,
  download-cache, and protected installation directories;
- Stabilized the AD-DNF numerical scale by reducing high-frequency encoding,
  displacement output scale, and learning rates;
- Added element-wise gradient clipping and best-validation checkpoint restore;
- Initialized the edge expert as an exact zero residual with a nearly closed
  gate, and froze the global Residual-FiLM field during edge-expert training;
- Included a synthetic demo generator for first-run validation.

MATLAB itself is not installed in the packaging environment. Run `RUN_DEMO`
locally before starting a long real-data experiment.
