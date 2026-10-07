#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-symmetric-v2-210e98f
expected_commit=210e98f365d85d9987524543feec34b0eaa3ca4e
training_source_dir="$code_root/experiments/transformers"
campaign_root=/root/MMNN/ICL-PDE/data/transformers/finite_basis_pde_symmetric_attention_v2
training_marker="$campaign_root/SYMMETRIC_ATTENTION_V2_TRAINING_COMPLETE"

actual_commit="$(git -C "$code_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "pinned symmetric-attention source mismatch: $actual_commit" >&2
  exit 1
fi
if ! git -C "$code_root" diff --quiet --ignore-space-at-eol HEAD -- 2>/dev/null; then
  echo "pinned symmetric-attention worktree has substantive tracked changes" >&2
  exit 1
fi
if [[ -n "$(git -C "$code_root" ls-files --others --exclude-standard)" ]]; then
  echo "pinned symmetric-attention worktree has untracked files" >&2
  exit 1
fi

while [[ ! -s "$training_marker" ]]; do
  sleep 60
done

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-symmetric-attention-v2-finalize

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/finalize_finite_basis_pde_campaign.py" \
  --root "$campaign_root" \
  --expected-runs 18 \
  --manifest "$campaign_root/SYMMETRIC_ATTENTION_MANIFEST.json" \
  --training-source-dir "$training_source_dir" \
  --after-marker "$training_marker" \
  --poll-seconds 60 \
  --audit-device cuda \
  --completion-marker SYMMETRIC_ATTENTION_V2_CAMPAIGN_COMPLETE
