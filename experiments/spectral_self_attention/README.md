# Spectral selection beyond arXiv:2604.26085

This directory contains an independent, reproducible extension of *Spectral
Selection in Symmetric Self-Attention Dynamics*.  It uses the exact spherical
ODE from equation (2.2), represented in an eigenbasis of the symmetric matrix
`V`.

The finite taxonomy in `taxonomy.py` is exhaustive up to changes that cannot
alter the qualitative spectral pattern: permutation/rotation of the eigenbasis,
continuous changes that preserve eigenvalue order, inertia, extreme-value
dominance, and multiplicity.  Parameter sweeps then vary the quantities that can
produce bifurcations inside a class: spectral ratios, attention sharpness, token
number, sign-group imbalance, and initialization.

The implementation treats repeated eigenvalues as eigenspaces rather than
arbitrarily declaring one basis vector selected.  It also evaluates the full
pure-mode Jacobian in Theorem 5.2, which gives a numerical oracle for every
spectrum, mode, beta, and sign split.

The main theoretical and empirical synthesis is
`refs/spectral_self_attention/EXTENDED_SPECTRAL_SELECTION.md`.  In addition to the
paper's pure modes, `mixed_equilibria.py` implements the exact balanced three-group
family discovered in the sweep.

The exhaustive equilibrium characterization is in
`refs/spectral_self_attention/ALL_EQUILIBRIA.md`.  Its spectral-Gram theorem is
implemented by `equilibrium_catalogue.py`; `run_equilibrium_catalogue.py` audits
the small planar systems.

Reproduce the completed artifacts with:

```bash
uv run python -m experiments.spectral_self_attention.run_sweep --profile full --workers 8
uv run python -m experiments.spectral_self_attention.mixed_equilibria
uv run python -m experiments.spectral_self_attention.long_time_audit
uv run python -m experiments.spectral_self_attention.make_figures
uv run python -m experiments.spectral_self_attention.multihead_phase
uv run python -m experiments.spectral_self_attention.mean_field_extensions
uv run python -m experiments.spectral_self_attention.one_step_muon
uv run python -m pytest -q tests/test_spectral_self_attention.py
```

`multihead_phase.py` freezes five tokens and colours the initial position of a
sixth token by its eventual destination on the circle and on the two-sphere. It
also measures whether random unit tokens and independent symmetric heads become
effectively decoupled as the ambient dimension grows. The outputs are in
`data/spectral_self_attention/multihead/`.

`mean_field_extensions.py` audits the finite-sum/continuum-integral limit,
replaces the exponential by three other positive kernels, compares normalized
and unnormalized attention, trains a terminal control around the mixed polygon,
and constructs a rotating two-head cluster. Its outputs are in
`data/spectral_self_attention/mean_field_extensions/`.

`one_step_muon.py` compares the one-step gradient mechanism with an exact
polar-factor Muon step and its five-step Newton--Schulz approximation. The three
updates are matched in continuous `L2` norm and written to
`data/spectral_self_attention/one_step_muon/`.
