# v9 changes

- Adds 20 unique spatial calibration splits per held-out case.
- Uses nested four- and six-coordinate calibration sets.
- Uses one common fixed test set per repeat for SOTA, Ridge-4, and Ridge-6.
- Adds global metrics by repeat and mean/standard-deviation summaries.
- Adds case-level calibration win rates and 5th/95th percentile gains.
- Adds repeated 95 cm robustness results.
- Adds a six-point LOOCV gate ablation.
- Reuses completed Step19-SOTA predictions without neural-network retraining.
