# Model evolution and experiment roles

## Research objective

The code reconstructs a continuous pavement response field from discrete FBG
measurements. The main physics-inspired formulation predicts a displacement
proxy field

\[
w_\theta = w_\theta(x,y,c)
\]

and obtains directional strain supervision through automatic differentiation:

\[
\hat{\varepsilon}_x=-z\frac{\partial^2w_\theta}{\partial x^2},
\qquad
\hat{\varepsilon}_y=-z\frac{\partial^2w_\theta}{\partial y^2}.
\]

Longitudinal FBGs use \(\hat{\varepsilon}_x\), while transverse FBGs use
\(\hat{\varepsilon}_y\).

## Experiment sequence

1. **Step13 - direct-response baseline.** A coordinate MLP directly fits the
   measured strain or wavelength response. It establishes whether the processed
   data support neural reconstruction.
2. **Step14 - displacement neural field.** The output changes from direct
   strain to a displacement proxy whose second derivatives supervise strain.
   Asphalt and soil are both included, making this an exploratory physical
   model.
3. **Step15 - asphalt-only and load-balanced.** Soil is removed from training,
   and load weighting reduces domination by large-load cases.
4. **Step16 - weak edge anchoring.** Weak boundary collocation fixes gauge
   freedom without asserting a strict simply supported or clamped boundary.
5. **Step17 - FiLM-conditioned AD-DNF.** Loading conditions modulate the spatial
   network through FiLM. Spatial features are constructed inside the tracked
   automatic-differentiation graph.
6. **Step18 - point-mode diagnostic.** A point-wise split tests interpolation
   within observed loading conditions rather than zero-shot condition
   generalization.
7. **Step19 - boundary-aware main model.** Boundary-distance features and
   moderate boundary weighting improve ordinary and non-extreme-boundary
   reconstruction while retaining a strict case split.
8. **Step21 - boundary residual expert.** A gated residual branch is tested as
   a negative ablation. Its small improvement on the extreme boundary does not
   offset its lower overall generalization.

## Reported comparison

| Model | Split | Test R2 | All R2 | 95 cm R2 | Interpretation |
|---|---:|---:|---:|---:|---|
| Step13 | case | 0.8886 | 0.9457 | not reported | Stable direct-fit baseline |
| Step19 | case | 0.7810 | 0.9164 | 0.3703 | Selected main model |
| Step19 excluding extreme boundary | case | - | 0.9586 | excluded | Stable ordinary-condition result |
| Step21 | case | 0.7591 | 0.8932 | 0.3973 | Negative ablation |

The accompanying report also lists Step20A and Step20B. Their standalone source
files were not present in the supplied code snapshot, so they are not included
as reproducible repository entries.

## Later experiment packages

The standalone packages under `experiments/` test whether the 95 cm limitation
should be addressed by a more complex shared neural network or by low-dimensional
adaptation from a few measurements:

| Version | Package | Question | Decision |
|---|---|---|---|
| Initial | `ablations/ResidualMoE_initial` | Do residual blocks or a spatial edge expert help? | Keep as structural ablations |
| v4 | `ablations/FixedFewshot_v4` | How do calibration methods compare on one fixed sensor split? | Fixed-split diagnostic |
| v5 | `ablations/FinalCalibration_v5` | Can the calibration workflow be made reproducible? | Superseded workflow version |
| v6 | `ablations/GlobalBenchmark_v6` | Are the structural ideas promising under a fast global benchmark? | Screening evidence only |
| v7 | `ablations/SOTAFullComparison_v7` | Does Residual-FiLM beat Step19 at the same full training budget? | No; strict negative ablation |
| v8 | `ablations/SOTACalibrationGate_v8` | Do Ridge-4 and a four-point LOOCV gate help? | Ridge-4 ablation; reject the gate |
| v9 | `calibration/SOTARidgeRobustness_v9` | Are Ridge-4/6 gains robust to sensor selection? | Retain Ridge-6 and six-point gate |

The fair v7 comparison reverses the apparent ranking from the early fast
screening: Step19-SOTA reaches `R2=0.8467, RMSE=79.51 microstrain`, whereas
Residual-FiLM reaches `R2=0.7106, RMSE=109.26 microstrain`. The structural
modification is therefore not selected.

The v9 analysis uses 20 repeated nested spatial splits. Ridge-6 reduces mean
overall RMSE from `86.63` to `80.44 microstrain` (7.15%) and improves the
95 cm condition by `15.78 microstrain` on average with an 85% win rate. The
six-point gate gives up some mean gain in exchange for lower run-to-run
variation and an explicit fallback to SOTA.

## Selected deployment policy

- **No new-condition measurements:** Step19-SOTA.
- **Six measurements, accuracy priority:** Step19-SOTA + Ridge-6.
- **Six measurements, risk priority:** Step19-SOTA + Gated Ridge-6.
- **Four measurements:** Ridge-4 only as a low-sensor option/ablation; do not
  use the four-point LOOCV gate.

## Interpretation limits

- The predicted displacement is a proxy field constrained by strain
  derivatives, not an independently measured absolute displacement.
- Case-wise and point-wise splits answer different questions and should not be
  compared as if they measured the same generalization task.
- The 95 cm extreme-boundary condition is outside the reliable zero-shot
  operating region of the current main model.
- A small number of measured extreme-boundary points or stronger experimentally
  justified boundary information is preferable to adding unsupported hard
  boundary conditions.
- The 20 repeated spatial splits measure sensitivity to calibration-point
  selection; they are not 20 independent physical loading experiments.
- Ridge calibration corrects only a global affine scale and offset. It cannot
  repair local peak-location or response-shape errors.
