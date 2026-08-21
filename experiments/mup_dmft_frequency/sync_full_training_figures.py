"""Copy every generated vector figure used by the paper into its source tree."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "figures"
DESTINATION = ROOT.parents[1] / "refs" / "mup_dmft_frequency" / "figures"
STEMS = (
    "hierarchy_and_kernel_evolution",
    "dynamic_saddle_spectrum",
    "frequency_rank_controls",
    "mup_width_collapse",
    "full_training_depth_hierarchy",
    "muon_hierarchy_clocks",
    "muon_powerlaw_front",
    "muon_paired_endpoints",
    "muon_mup_width_transfer",
    "full_training_dynamic_diagnostics",
    "full_training_step_convergence",
)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = [stem for stem in STEMS if not (SOURCE / f"{stem}.pdf").exists()]
    if missing:
        raise FileNotFoundError(f"missing generated figures: {missing}")
    for stem in STEMS:
        source = SOURCE / f"{stem}.pdf"
        destination = DESTINATION / source.name
        shutil.copy2(source, destination)
        print(destination)


if __name__ == "__main__":
    main()
