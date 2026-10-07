#!/usr/bin/env bash
set -euo pipefail

code_root=/root/MMNN/.campaign-worktrees/icl-pde-symmetric-v2-210e98f
expected_commit=210e98f365d85d9987524543feec34b0eaa3ca4e
data_root=/root/MMNN/ICL-PDE/data/transformers
campaign_root="$data_root/finite_basis_pde_symmetric_attention_v2"
prerequisite="$data_root/finite_basis_pde_rank4_spread_cold_v1/k8_m0_16_query_attention_rank4_spread_cold_seed2/COMPLETE"

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

cd "$code_root"
export MPLCONFIGDIR=/tmp/mpl-iclpde-symmetric-attention-v2

exec /root/MMNN/.venv/bin/python \
  "$code_root/experiments/transformers/launch_finite_basis_pde_symmetric_attention_v2.py" \
  --root "$campaign_root" \
  --wait-for "$prerequisite" \
  --poll-seconds 60
