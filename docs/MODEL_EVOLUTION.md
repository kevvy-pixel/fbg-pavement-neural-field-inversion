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
