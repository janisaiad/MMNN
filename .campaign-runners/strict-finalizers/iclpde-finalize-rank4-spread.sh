#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-b963fcc
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_rank4_spread_v1

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-finalize-rank4-spread-strict

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --manifest "$campaign_root/MANIFEST_k8_rank4_spread.json" \
  --poll-seconds 60 \
  --audit-device cuda
