# Package information

- Package: `FBG_NeuralField_GlobalBenchmark_v6`
- Input: `Response_Matrix_Table.csv`
- Evaluation unit: complete experimental case
- quick/research folds: 3/5
- Original repository files modified: none
- Experimental data included: none

## Global benchmark

- Plain-FiLM, Residual-FiLM and spatial edge expert are retrained on identical
  case-level folds;
- every case is held out exactly once;
- response and load normalization are refit from training cases only;
- ridge calibration uses 4/6 target-case coordinates and evaluates on common
  untouched coordinates;
- both case-macro and point-micro metrics are exported;
- training, convergence, inference, calibration and parameter-count results
  are exported separately.

MATLAB is not installed in the packaging environment. All `.m` files are
statically parsed before delivery. `RUN_DEMO` is the local dynamic check.
