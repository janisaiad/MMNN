#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-f8e23c2
live_root=/root/MMNN/ICL-PDE/data/transformers

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-spectral-complement

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/launch_finite_basis_pde_spectral_complement.py" \
  --root "$live_root/finite_basis_pde_spectral_complement_v1" \
  --wait-for "$live_root/finite_basis_pde_spectral_controls_v1/SPECTRAL_CAMPAIGN_COMPLETE" \
  --poll-seconds 60
