#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-symmetric-v2-210e98f
expected_commit=210e98f365d85d9987524543feec34b0eaa3ca4e
data_root=/root/MMNN/ICL-PDE/data/transformers
campaign_root="$data_root/finite_basis_pde_symmetric_attention_v2"
comparison_root="$data_root/finite_basis_pde_symmetric_attention_comparison_v2"
scientific_marker="$campaign_root/SYMMETRIC_ATTENTION_V2_CAMPAIGN_COMPLETE"

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

while [[ ! -s "$scientific_marker" ]]; do
  sleep 60
done

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-symmetric-attention-v2-comparison

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/compare_symmetric_attention_preconditioners.py" \
  --symmetric-attention-root "$campaign_root" \
  --outdir "$comparison_root" \
  --audit-device cpu \
  --audit-batch-size 32
