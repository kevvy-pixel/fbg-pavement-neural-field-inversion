# v8 changes

- Adds `Step19-SOTA+Ridge-4`.
- Adds a pre-registered no-leakage calibration gate.
- Uses leave-one-coordinate-out cross-validation within the four calibration
  coordinates.
- Requires at least 5% relative and 2 microstrain absolute calibration-CV RMSE
  improvement.
- Rejects affine scales close to the configured hard bounds.
- Adds per-case gate decisions, LOOCV predictions, paired improvements, 95 cm
  results, and gate audit outputs.
- Can post-process a completed v7 run without neural-network retraining.
