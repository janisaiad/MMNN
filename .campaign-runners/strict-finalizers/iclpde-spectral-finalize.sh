#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-b963fcc
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_spectral_controls_v1
training_source_dir=/root/MMNN/.campaign-worktrees/icl-pde-225589f/experiments/transformers

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-spectral-finalize-strict

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --manifest "$campaign_root/SPECTRAL_CONTROL_MANIFEST.json" \
  --training-source-dir "$training_source_dir" \
  --poll-seconds 60 \
  --audit-device cuda \
  --completion-marker SPECTRAL_CAMPAIGN_COMPLETE
