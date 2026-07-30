# Package information

- Package: `FBG_NeuralField_FixedFewshot_v4`
- Public repository inspected: `kevvy-pixel/fbg-multilayer-pavement-modeling`
- Repository commit inspected: `17a78be9d0c6ab13fc428dee221c4ee35e276735`
- Input: `Response_Matrix_Table.csv`
- Original repository files modified: none
- Private experimental data included: none

## Implemented stages

| Stage | Purpose |
| --- | --- |
| `00_data_validation` | Validate data and define the held-out target case |
| `01_plain_film` | Plain FiLM neural-field baseline |
| `02_residual_film` | Residual FiLM neural field |
| `03_spatial_edge_expert` | Coordinate-aware soft gate and edge expert |
| `04_fixed_holdout_fewshot_95cm` | Repeated fixed-test comparison of four calibration methods |
| `05_comparison` | Full-target comparison of the three main models |

## Version 4 additions

- A fixed evaluation set within every repeat and every k;
- Nested calibration points for `k=0,2,4,6,8`;
- Repeated spatial-coverage splits with mean and standard deviation;
- Scalar-bias and output-head baselines;
- An explicit complexity decision table;
- Full-batch updates for the tiny few-shot calibration sets;
- `RUN_FIXED_FEWSHOT` to reuse an earlier run's trained models.

The public repository does not contain the private Step19-Step21 neural-field
source. This remains a standalone extension compatible with the public
response-table format.

MATLAB is not installed in the packaging environment. Static parsing and
package validation are performed here; run `RUN_DEMO` locally before a long
experiment.
