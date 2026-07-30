# Package information

- Package: `FBG_NeuralField_SOTARidgeRobustness_v9`
- MATLAB: R2024a
- Existing-run entry: `RUN_NEXT_EXPERIMENT.m`
- Core experiment: `RUN_SOTA_RIDGE_ROBUSTNESS.m`
- Default repeats: 20 unique spatial calibration pools per case
- Design: nested Ridge-4/Ridge-6 with a common fixed test set
- Additional ablation: calibration-only Ridge-6 LOOCV gate
- Neural-network retraining: none for an existing v7 run
