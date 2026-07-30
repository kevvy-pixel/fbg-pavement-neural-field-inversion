# v4 changes

The old few-shot table changed the test set as k increased. Version 4 creates
one calibration pool per repeat and holds the remaining coordinates fixed for
all methods and calibration sizes.

New baselines:

- `residual_zero`;
- `bias_only`;
- `output_head`;
- `edge_expert`.

New aggregate outputs include per-repeat metrics, mean and standard deviation,
all fixed-test predictions, and a complexity decision table. The standalone
`RUN_FIXED_FEWSHOT` entry point reuses models from a previous successful run.
