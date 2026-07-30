# v5 changes

- Guarantees unique calibration-pool coordinate sets;
- increases quick/research repeats from 5/10 to 10/20;
- adds regularized affine calibration;
- removes repeated edge-expert calibration to save CPU time;
- adds paired improvement and RMSE win-rate statistics;
- adds per-k automatic method selection;
- requires complex methods to beat both bias-only and zero-shot baselines;
- reuses only `02_residual_film/model.mat` from an earlier run.
