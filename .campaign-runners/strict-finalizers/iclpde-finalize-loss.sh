#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-b963fcc
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_loss_controls_v1

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-finalize-loss-strict

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --manifest "$campaign_root/CONTROL_MANIFEST.json" \
  --poll-seconds 60 \
  --audit-device cuda
