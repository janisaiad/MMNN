# Fixed-Geometry ICL for LSM — presentation brief

## The question

Can a fixed nonlinear GP/softmax geometry support a tied in-context solver for
the actual 2D Linear Sampling Method, and can the differentiable solver then be
used for experiment design?

## What was run

- Active deterministic 2D multistatic scattering, not random-source LSM.
- 32 fixed plane-wave sources and 32 receivers; complex `F_D` matrices.
- Fixed angular von Mises/softmax kernel.
- Tied complex PCG; only an SPD attention preconditioner is learned.
- 1,500 training steps, three seeds, 576 held-out tasks per condition.
- Ellipses/disks in training; kites and two obstacles held out.
- Separate experiment: learn six distinct source/receiver angles.

## Main fixed-geometry result at 15% noise

| Method | Ellipse AP | Kite AP | Two-disk AP | Kite residual |
|---|---:|---:|---:|---:|
| Trained PCG, 8 loops | 0.9915 | 0.9881 | 0.9136 | 0.727 |
| Trained PCG, 20 loops | 0.9917 | 0.9889 | 0.9131 | 0.0102 |
| Direct Tikhonov | 0.9917 | 0.9889 | 0.9131 | 0.00002 |

Localization saturates early; solver fidelity does not. This is why the
residual must be shown next to the reconstructions.

## Experiment-design result at 12% noise

| Six-angle geometry | Ellipse AP | Kite AP | Two-disk AP |
|---|---:|---:|---:|
| Learned | 0.958 | 0.963 | 0.879 |
| Random feasible | 0.897 | 0.909 | 0.814 |
| Uniform | 0.260 | 0.263 | 0.244 |

The uniform six-angle array has a maximum point-spread sidelobe of 0.999. The
learned design reduces it to 0.562 by breaking periodic grating-lobe aliases.

## The modelling distinction

- The softmax nonlinearity is chosen from the desired angular kernel; it is not
  a learned temperature.
- The model does not predict the image or the LSM indicator.
- It does not learn a noise-dependent spatial regularization map.
- Learning is restricted to a structure-preserving SPD preconditioner and then
  to the acquisition angles.

## Honest limitation

This is a synthetic Born/Foldy PoC. The next decisive test is zero-shot transfer
to an independent sound-soft boundary-integral simulator. Random-source `C_D`
and moving-small-scatterer `C_tilde_D` experiments have not yet been run.
