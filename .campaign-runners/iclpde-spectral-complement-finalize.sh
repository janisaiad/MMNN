#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-symmetric-complement-8d948c9
live_root=/root/MMNN/ICL-PDE/data/transformers
campaign_root="$live_root/finite_basis_pde_spectral_complement_v1"

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-spectral-complement-finalize

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --expected-runs 6 \
  --manifest "$campaign_root/SPECTRAL_COMPLEMENT_MANIFEST.json" \
  --after-marker "$campaign_root/SPECTRAL_COMPLEMENT_TRAINING_COMPLETE" \
  --poll-seconds 60 \
  --audit-device cuda \
  --completion-marker SPECTRAL_COMPLEMENT_CAMPAIGN_COMPLETE
