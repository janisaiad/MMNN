#!/usr/bin/env python3
"""
we load all baseline sweep results, identify best losses, build a table and a concise md summary (which worked / which did not).
we also list all worked configs sorted by loss and plot a histogram of min losses per factor.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

_BASE = Path(__file__).resolve().parent
RESULTS_DIR = _BASE / "results_baseline_sweep"
OUT_MD = _BASE / "baseline_sweep_summary.md"
OUT_CSV = _BASE / "baseline_sweep_results.csv"
OUT_HIST_PNG = _BASE / "baseline_sweep_loss_histogram.png"
RESULTS_DIR_SUMCOS = _BASE / "results_baseline_sweep_sumcos"
OUT_MD_SUMCOS = _BASE / "baseline_sweep_sumcos_summary.md"
OUT_CSV_SUMCOS = _BASE / "baseline_sweep_sumcos_results.csv"
OUT_HIST_PNG_SUMCOS = _BASE / "baseline_sweep_sumcos_loss_histogram.png"

# we consider "worked" if final test error is below this and finite
TEST_ERR_WORKED_THRESHOLD = 0.01
# we consider "failed" if test error above this or NaN/Inf
TEST_ERR_FAILED_ABOVE = 0.5


def load_all():
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        lf = d / "losses.json"
        if not lf.exists():
            continue
        try:
            with open(lf) as f:
                data = json.load(f)
        except Exception:
            continue
        cfg = data.get("config") or {}
        name = cfg.get("name") or d.name
        final_test = data.get("final_test_error")
        final_train = data.get("final_train_error")
        epochs_run = data.get("epochs_run", 0)
        all_losses = data.get("all_losses") or []
        if final_test is None and all_losses:
            final_train = all_losses[-1] if all_losses else None
        min_loss = float(min(all_losses)) if all_losses else np.nan
        finite = isinstance(final_test, (int, float)) and np.isfinite(final_test)
        test_err = float(final_test) if finite else np.nan
        worked = finite and test_err < TEST_ERR_WORKED_THRESHOLD
        failed = not finite or test_err >= TEST_ERR_FAILED_ABOVE
        rows.append({
            "name": name,
            "factor": int(cfg.get("factor", 0)),
            "N": int(cfg.get("n_train", 0)),
            "bs": int(cfg.get("batch_size", 0)),
            "L": int(cfg.get("num_layers", 0)),
            "final_test_error": test_err,
            "final_train_error": float(final_train) if final_train is not None and np.isfinite(final_train) else np.nan,
            "min_loss": min_loss,
            "epochs_run": int(epochs_run),
            "worked": worked,
            "failed": failed,
        })
    return rows


def main(results_dir=None, out_md=None, out_csv=None, out_hist_png=None):
    global RESULTS_DIR, OUT_MD, OUT_CSV, OUT_HIST_PNG
    if results_dir is not None:
        RESULTS_DIR = results_dir
    if out_md is not None:
        OUT_MD = out_md
    if out_csv is not None:
        OUT_CSV = out_csv
    if out_hist_png is not None:
        OUT_HIST_PNG = out_hist_png
    rows = load_all()
    if not rows:
        print("no results found in", RESULTS_DIR)
        return

    # sort by final_test_error (best first), NaN last
    def key(r):
        e = r["final_test_error"]
        return (np.isnan(e), e if np.isfinite(e) else float("inf"))

    rows_sorted = sorted(rows, key=key)

    # best 20
    best = [r for r in rows_sorted if np.isfinite(r["final_test_error"])][:20]
    failed = [r for r in rows if r["failed"]]
    worked = [r for r in rows if r["worked"]]
    worked_sorted = sorted(worked, key=lambda r: r["final_test_error"])

    # build summary table by (factor, N, bs): count worked / total, best L
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in rows:
        k = (r["factor"], r["N"], r["bs"])
        by_config[k].append(r)

    lines = []
    title = "Baseline sweep (sumcos)" if "sumcos" in str(OUT_MD) else "Baseline sweep summary"
    lines.append(f"# {title}")
    lines.append("")
    target_desc = "Target: $\\sum_{k=1}^{\\mathrm{factor}} \\cos(2\\pi k x)$" if "sumcos" in str(OUT_MD) else "Target: cos(2 π factor x)"
    lines.append(f"{target_desc}, N = base×factor, bs ∈ {{1,2,4,8,16}}, L ∈ {{1..2×factor}}. "
                 "**Worked** = final test error < 0.01; **Failed** = test error ≥ 0.5 or NaN/Inf.")
    lines.append("")
    lines.append("## Best configs (by final test error)")
    lines.append("")
    lines.append("| config | factor | N | bs | L | final_test_err | final_train_err | epochs |")
    lines.append("|--------|--------|---|-----|---|----------------|-----------------|--------|")
    for r in best:
        lines.append(
            f"| {r['name']} | {r['factor']} | {r['N']} | {r['bs']} | {r['L']} | "
            f"{r['final_test_error']:.4e} | {r['final_train_error']:.4e} | {r['epochs_run']} |"
        )
    lines.append("")
    lines.append("## Worked vs did not")
    lines.append("")
    lines.append(f"- **Worked** (test err < {TEST_ERR_WORKED_THRESHOLD}): {len(worked)} configs.")
    lines.append(f"- **Failed** (test err ≥ {TEST_ERR_FAILED_ABOVE} or NaN/Inf): {len(failed)} configs.")
    lines.append(f"- Total completed: {len(rows)}.")
    lines.append("")
    lines.append("## All worked configs (sorted by final test error)")
    lines.append("")
    lines.append("| # | config | factor | N | bs | L | final_test_err | final_train_err | min_loss | epochs |")
    lines.append("|---|--------|--------|---|-----|---|----------------|-----------------|----------|--------|")
    for i, r in enumerate(worked_sorted, 1):
        min_l = r.get("min_loss")
        min_l_str = f"{min_l:.4e}" if min_l is not None and np.isfinite(min_l) else "–"
        lines.append(
            f"| {i} | {r['name']} | {r['factor']} | {r['N']} | {r['bs']} | {r['L']} | "
            f"{r['final_test_error']:.4e} | {r['final_train_error']:.4e} | {min_l_str} | {r['epochs_run']} |"
        )
    lines.append("")
    lines.append("### Worked (representative)")
    lines.append("")
    for r in worked[:30]:
        lines.append(f"- `{r['name']}` test_err={r['final_test_error']:.4e} L={r['L']} N={r['N']} bs={r['bs']}")
    if len(worked) > 30:
        lines.append(f"- ... and {len(worked) - 30} more.")
    lines.append("")
    lines.append("### Did not work (representative)")
    lines.append("")
    for r in failed[:30]:
        err = r["final_test_error"]
        err_str = f"{err:.4e}" if np.isfinite(err) else "NaN/Inf"
        lines.append(f"- `{r['name']}` test_err={err_str} L={r['L']} N={r['N']} bs={r['bs']}")
    if len(failed) > 30:
        lines.append(f"- ... and {len(failed) - 30} more.")
    lines.append("")
    lines.append("## Table by factor (mean test error, count worked/total)")
    lines.append("")
    factors = sorted(set(r["factor"] for r in rows))
    lines.append("| factor | N range | total | worked | failed | best test err |")
    lines.append("|--------|---------|-------|--------|--------|---------------|")
    for f in factors:
        sub = [r for r in rows if r["factor"] == f]
        worked_f = [r for r in sub if r["worked"]]
        best_err = min((r["final_test_error"] for r in sub if np.isfinite(r["final_test_error"])), default=np.nan)
        n_vals = [r["N"] for r in sub]
        n_range = f"{min(n_vals)}–{max(n_vals)}" if n_vals else "–"
        best_str = f"{best_err:.4e}" if np.isfinite(best_err) else "–"
        lines.append(f"| {f} | {n_range} | {len(sub)} | {len(worked_f)} | {len(sub)-len(worked_f)} | {best_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("Full results CSV: `baseline_sweep_results.csv`. Histogram: `baseline_sweep_loss_histogram.png`.")

    md_text = "\n".join(lines)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(md_text)
    print("wrote", OUT_MD)

    # histogram of min_loss per factor (we use former LaTeX-style formatting from meanfield_cosine_multifreq_experiment)
    plt.rcParams["figure.figsize"] = [6, 6]
    plt.rcParams["font.size"] = 18
    plt.rcParams["font.weight"] = "normal"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["mathtext.rm"] = "serif"
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.size"] = 22
    mpl.rcParams["axes.formatter.limits"] = (-6, 6)
    mpl.rcParams["axes.formatter.use_mathtext"] = True
    mpl.rcParams["font.family"] = "STIXGeneral"
    mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    mpl.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    mpl.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    mpl.rcParams["xtick.minor.visible"] = True
    mpl.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True

    factors = sorted(set(r["factor"] for r in rows))
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes_flat = axes.ravel()
    for idx, f in enumerate(factors):
        ax = axes_flat[idx]
        sub = [r for r in rows if r["factor"] == f and np.isfinite(r.get("min_loss", np.nan))]
        min_losses = [r["min_loss"] for r in sub]
        if not min_losses:
            ax.text(0.5, 0.5, f"factor {f}\nno finite min_loss", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # we bin in log space so bins are uniform on log scale (auto count)
            pos = [x for x in min_losses if x > 0]
            if not pos:
                ax.text(0.5, 0.5, f"factor {f}\nno positive min_loss", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                n_bins = max(15, min(50, int(1 + np.log2(len(pos)))))
                log_lo = np.log10(min(pos))
                log_hi = np.log10(max(pos))
                bins_log = np.logspace(log_lo, log_hi, num=n_bins + 1)
                ax.hist(pos, bins=bins_log, color="steelblue", alpha=0.8, edgecolor="white")
                ax.set_xscale("log")
                ax.set_xlabel("min loss (train)")
                ax.set_ylabel("count")
                ax.set_title(f"factor {f} (n={len(min_losses)})")
    if len(factors) < 6:
        axes_flat[len(factors)].set_visible(False)
    axes_flat[5].set_visible(False)
    plt.suptitle("Histogram of min loss per factor (min over epochs)")
    plt.tight_layout()
    plt.savefig(OUT_HIST_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", OUT_HIST_PNG)

    # write CSV
    import csv
    cols = ["name", "factor", "N", "bs", "L", "final_test_error", "final_train_error", "min_loss", "epochs_run", "worked", "failed"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            row = {k: r[k] for k in cols}
            for k in ("final_test_error", "final_train_error", "min_loss"):
                if k in row and (row[k] is None or (isinstance(row[k], float) and np.isnan(row[k]))):
                    row[k] = ""
            w.writerow(row)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Analyze baseline sweep results; optionally use sumcos results.")
    ap.add_argument("--sumcos", action="store_true", help="use results_baseline_sweep_sumcos (target = sum_{k=1}^{factor} cos(2 pi k x))")
    args = ap.parse_args()
    if args.sumcos:
        main(results_dir=RESULTS_DIR_SUMCOS, out_md=OUT_MD_SUMCOS, out_csv=OUT_CSV_SUMCOS, out_hist_png=OUT_HIST_PNG_SUMCOS)
    else:
        main()
