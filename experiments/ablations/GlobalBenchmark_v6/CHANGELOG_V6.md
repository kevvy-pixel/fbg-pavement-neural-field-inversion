# v6 changes

- Adds complete-case 3/5-fold cross-validation;
- refits response/load scales within every training fold;
- evaluates Plain, Residual, Edge, Ridge-4 and Ridge-6 on common coordinates;
- exports separate full-coordinate zero-shot metrics;
- records neural training, time-to-best and common-target convergence;
- benchmarks inference and closed-form calibration time;
- counts backbone and adaptation parameters;
- prevents duplicate `FBG_NeuralField_Workspace` path nesting;
- preserves every validation checkpoint in training histories.
- accepts both `split` and legacy `case_split` case-metric column names.
