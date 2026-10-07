#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-symmetric-complement-8d948c9
live_root=/root/MMNN/ICL-PDE/data/transformers
parent_root="$live_root/finite_basis_pde_spectral_controls_v1"
complement_root="$live_root/finite_basis_pde_spectral_complement_v1"
comparison_root="$live_root/finite_basis_pde_symmetric_attention_comparison_v1"

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-symmetric-attention-comparison

while [[ ! -s "$parent_root/SPECTRAL_CAMPAIGN_COMPLETE" || \
         ! -s "$complement_root/SPECTRAL_COMPLEMENT_CAMPAIGN_COMPLETE" ]]; do
  sleep 60
done

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/compare_symmetric_attention_preconditioners.py" \
  --spectral-controls-root "$parent_root" \
  --spectral-complement-root "$complement_root" \
  --outdir "$comparison_root" \
  --audit-device cpu \
  --audit-batch-size 32
