#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-frozen-encoder-a97dd57
expected_commit=a97dd5716a0f5cb0833f9dbdf36428597cf74f36
data_root=/root/MMNN/ICL-PDE/data/transformers
campaign_root="$data_root/finite_basis_pde_frozen_encoder_symmetric_control_v1"
prerequisite="$data_root/finite_basis_pde_symmetric_attention_v2/SYMMETRIC_ATTENTION_V2_CAMPAIGN_COMPLETE"

actual_commit="$(git -C "$code_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "pinned frozen-encoder source mismatch: $actual_commit" >&2
  exit 1
fi
if ! git -C "$code_root" diff --quiet --ignore-space-at-eol HEAD -- 2>/dev/null; then
  echo "pinned frozen-encoder worktree has substantive tracked changes" >&2
  exit 1
fi
if [[ -n "$(git -C "$code_root" ls-files --others --exclude-standard)" ]]; then
  echo "pinned frozen-encoder worktree has untracked files" >&2
  exit 1
fi

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-frozen-encoder-symmetric-control

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/launch_finite_basis_pde_frozen_encoder_control.py" \
  --root "$campaign_root" \
  --wait-for "$prerequisite" \
  --poll-seconds 60
