#!/usr/bin/env python3
"""Reproduce the RF-LR symmetry experiments and rebuild the paper assets."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY_FIGURES = ROOT.parent / "icml_sgdadamlandscapedynamical" / "figures"
PAPER_ASSETS = ROOT / "results" / "paper_assets"


EXPERIMENT_SCRIPTS = [
    "good_sumcos_low_loss.py",
    "all_iclr_sumcos_rerun.py",
    "symmetry_grid_long.py",
    "analyze_symmetry_grid.py",
    "multidim_symmetry_batch1.py",
    "weightspace_distribution_analysis.py",
    "run_exact_fullrank_counterparts.py",
    "build_paper_assets.py",
]


LEGACY_ASSETS = {
    "mlp_asymmetry_layer7.png": "paper_mlp_asymmetry_layer7.png",
    "mlp_asymmetry_layer16.png": "paper_mlp_asymmetry_layer16.png",
}


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def copy_legacy_assets() -> None:
    PAPER_ASSETS.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in LEGACY_ASSETS.items():
        source = LEGACY_FIGURES / source_name
        target = PAPER_ASSETS / target_name
        if not source.exists():
            raise FileNotFoundError(f"Missing legacy figure: {source}")
        shutil.copy2(source, target)
        print(f"copied {source} -> {target}", flush=True)


def run_experiments(dry_run: bool, fast: bool) -> None:
    scripts = ["build_paper_assets.py"] if fast else EXPERIMENT_SCRIPTS
    for script in scripts:
        path = ROOT / script
        if not path.exists():
            raise FileNotFoundError(f"Missing reproduction script: {path}")
        run_command([sys.executable, script], dry_run=dry_run)


def compile_paper(dry_run: bool) -> None:
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "workshop_main.tex"],
        ["bibtex", "workshop_main"],
        ["pdflatex", "-interaction=nonstopmode", "workshop_main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "workshop_main.tex"],
    ]
    for command in commands:
        run_command(command, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce experiments and rebuild the RF-LR symmetry paper.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--fast", action="store_true", help="Only rebuild paper assets from existing results.")
    parser.add_argument("--skip-compile", action="store_true", help="Do not run LaTeX after rebuilding assets.")
    args = parser.parse_args()

    copy_legacy_assets()
    run_experiments(dry_run=args.dry_run, fast=args.fast)
    copy_legacy_assets()
    if not args.skip_compile:
        compile_paper(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
