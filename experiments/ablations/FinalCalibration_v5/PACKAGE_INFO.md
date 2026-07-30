# Package information

- Package: `FBG_NeuralField_FinalCalibration_v5`
- Input: `Response_Matrix_Table.csv`
- Original repository files modified: none
- Experimental data included: none

## Final protocol

- 10 unique calibration pools in quick, 20 in research;
- fixed 20-coordinate test set within each repeat;
- nested `k=0,2,4,6,8`;
- Residual zero-shot, scalar bias, ridge affine, and output-head methods;
- paired RMSE win rates and unique-split audit;
- complex methods must beat both scalar bias and zero-shot baselines.

The spatial edge expert remains available in `RUN_ALL` as an architectural
ablation but is excluded from repeated target calibration because v4 showed
negligible adaptation.

MATLAB is not installed in the packaging environment. All `.m` files are
statically parsed before delivery; run `RUN_DEMO` locally for dynamic checks.
