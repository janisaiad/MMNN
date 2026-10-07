#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-frozen-encoder-a97dd57
expected_commit=a97dd5716a0f5cb0833f9dbdf36428597cf74f36
training_source_dir="$code_root/experiments/transformers"
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_frozen_encoder_symmetric_control_v1
training_marker="$campaign_root/FROZEN_ENCODER_DECODER_TRAINING_COMPLETE"

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
export MPLCONFIGDIR=/tmp/mpl-iclpde-frozen-encoder-symmetric-control-finalize

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --expected-runs 18 \
  --manifest "$campaign_root/FROZEN_ENCODER_DECODER_MANIFEST.json" \
  --after-marker "$training_marker" \
  --training-source-dir "$training_source_dir" \
  --audit-device cuda \
  --completion-marker FROZEN_ENCODER_SYMMETRIC_CONTROL_COMPLETE \
  --poll-seconds 60
