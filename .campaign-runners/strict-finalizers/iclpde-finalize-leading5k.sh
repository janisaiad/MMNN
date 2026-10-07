#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-b963fcc
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_leading_5k_control_v1

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-finalize-leading5k-strict

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --manifest "$campaign_root/MANIFEST_k8_leading_5k_control.json" \
  --poll-seconds 60 \
  --audit-device cuda
