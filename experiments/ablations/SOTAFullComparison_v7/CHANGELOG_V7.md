# v7 changes

- Restored the Step19 research capacity: spatial width 96, condition width 64,
  and five FiLM blocks.
- Restored x-axis mirror augmentation and boundary-aware sample weighting.
- Uses the reported boundary feature length 0.180 and extreme threshold 0.210.
- Gives SOTA and Residual identical initialization tensors, minibatch sequence,
  folds, loss, and 2400-iteration budget.
- Removed the spatial-edge expert from the main experiment.
- Adds Ridge-4 only after the common zero-shot backbone comparison.
- Adds five-fold out-of-fold, paired case-level, 95 cm, runtime, and complexity
  summaries.
- Adds completed-stage resume support.
