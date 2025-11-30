"""
Generate all paper figures from computed data and light simulations.

Outputs:
- refs/paper/figures/*.png, *.pdf
- A summary JSON with figure paths for embedding
"""

import json
import math
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("refs/paper/figures")
DATA_DIR = Path("refs/paper/data")
FIG_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(123456)


def relu_kernel_sigma1(rho: np.ndarray) -> np.ndarray:
    rho = np.clip(rho, -1.0, 1.0)  # we clip values to valid range #
    theta = np.arccos(rho)  # we compute angles #
    return (1.0 / np.pi) * (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + rho * (np.pi - theta))  # we return base kernel #


def relu_kernel_theta1(rho: np.ndarray) -> np.ndarray:
    rho = np.clip(rho, -1.0, 1.0)  # we clip values to valid range #
    theta = np.arccos(rho)  # we compute angles #
    return (1.0 / (2.0 * np.pi)) * ((np.pi - theta) * np.cos(theta) + np.sin(theta))  # we return derivative kernel #


def det_ntk_3layer(rho: np.ndarray) -> np.ndarray:
    theta1 = relu_kernel_theta1(rho)  # we compute derivative kernel #
    sigma1 = relu_kernel_sigma1(rho)  # we compute base kernel #
    return theta1 * (1.0 - np.arccos(np.clip(rho, -1.0, 1.0)) / np.pi) + sigma1 + 1.0  # we compute 3-layer ntk #


def savefig(fig, name: str):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")  # we save png #
    fig.savefig(pdf_path, bbox_inches="tight")  # we save pdf #
    plt.close(fig)  # we close the figure to free memory #
    return {"png": str(png_path), "pdf": str(pdf_path)}  # we return paths #


def plot1_rank_concentration():
    r_vals = np.unique(np.round(np.logspace(np.log10(5), np.log10(200), 40)).astype(int))  # we choose ranks #
    epsilons = [0.1, 0.2, 0.5, 1.0]  # we choose thresholds #
    n_mc = 20000  # we choose mc samples #
    fig, ax = plt.subplots(figsize=(8, 6))  # we create fig #
    for eps in epsilons:
        probs = []  # we initialize list #
        for r in r_vals:
            x = RNG.standard_normal((n_mc, r))  # we sample x #
            y = RNG.standard_normal((n_mc, r))  # we sample y #
            w = (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)) / r  # we compute w #
            probs.append(np.mean(np.abs(w - 1.0) >= eps))  # we estimate prob #
        probs = np.array(probs)  # we convert to array #
        ax.semilogy(r_vals, probs, "o-", ms=3, label=f"empirical, ε={eps}")  # we plot empirical #
        theo = 4.0 * np.exp(-r_vals * (eps**2) / 8.0)  # we compute bound #
        ax.semilogy(r_vals, theo, "--", label=f"theory, ε={eps}")  # we plot theory #
    ax.semilogy(r_vals, 1.0 / np.sqrt(r_vals), "k:", lw=2, label="classical O(1/√r)")  # we plot reference #
    ax.set_xlabel("rank r")  # we set label #
    ax.set_ylabel("P(|W-1| ≥ ε)")  # we set label #
    ax.set_title("Exponential concentration in rank")  # we set title #
    ax.grid(True, alpha=0.3)  # we add grid #
    ax.legend(ncol=2, fontsize=9)  # we add legend #
    return savefig(fig, "fig_plot1_rank_concentration")  # we save fig #


def approx_sample_fisher(rho_true: float, r: int, size: int) -> np.ndarray:
    mu = np.arctanh(np.clip(rho_true, -0.99, 0.99))  # we transform mean #
    sigma2 = 1.0 / max(r - 3, 1)  # we approximate variance #
    z = RNG.normal(loc=mu, scale=np.sqrt(sigma2), size=size)  # we sample normal #
    return np.tanh(z)  # we inverse transform #


def plot2_ntk_concentration():
    # prefer real data from grid if present; fallback to approximate sampling #
    candidates = sorted(list(DATA_DIR.glob("*_ntk_rho.npz")))  # we search saved ntk-rho files #
    if len(candidates) > 0:
        data = np.load(candidates[0], allow_pickle=True)  # we load first candidate #
        rho_vals = data["rho_vals"]  # we read rho #
        mean_vals = data["ntk_mean"]  # we read mean #
        std_vals = data["ntk_std"]  # we read std #
        if "k_infty" in data:
            det = data["k_infty"]  # we read deterministic #
        else:
            det = det_ntk_3layer(rho_vals)  # we compute deterministic #
        fig, ax = plt.subplots(figsize=(8, 6))  # we create fig #
        ax.plot(rho_vals, det, "k-", lw=2, label="$K_\\infty(\\rho)$")  # we plot deterministic #
        ax.plot(rho_vals, mean_vals, color="C0", lw=2, label="empirical mean")  # we plot empirical #
        ax.fill_between(rho_vals, mean_vals - 2 * std_vals, mean_vals + 2 * std_vals, color="C0", alpha=0.25, label="±2σ")  # we band #
        ax.set_xlabel("$\\rho$")  # we label x #
        ax.set_ylabel("NTK value")  # we label y #
        ax.set_title("Empirical NTK vs deterministic kernel across $\\rho$")  # we title #
        ax.grid(True, alpha=0.3)  # we grid #
        ax.legend(fontsize=10)  # we legend #
        return savefig(fig, "fig_plot2_ntk_concentration")  # we save #
    # fallback: approximate Fisher–Kibble demo for multiple r #
    r_list = [10, 30, 100]  # we set ranks #
    rho_vals = np.linspace(-1.0, 1.0, 121)  # we set rho grid #
    n_mc = 2000  # we set mc samples #
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)  # we create panes #
    det = det_ntk_3layer(rho_vals)  # we compute deterministic curve #
    for ax, r in zip(axes, r_list):
        means = []  # we init #
        stds = []  # we init #
        for rho in rho_vals:
            rho_hat = approx_sample_fisher(float(rho), r, n_mc)  # we sample fisher #
            x_norm = np.linalg.norm(RNG.standard_normal((n_mc, r)), axis=1)  # we sample norm #
            y_norm = np.linalg.norm(RNG.standard_normal((n_mc, r)), axis=1)  # we sample norm #
            w_r = (x_norm * y_norm) / r  # we compute w_r #
            theta1 = relu_kernel_theta1(rho_hat)  # we compute theta1 #
            sigma1 = relu_kernel_sigma1(rho_hat)  # we compute sigma1 #
            ntk = theta1 * (1.0 - np.arccos(np.clip(rho_hat, -1.0, 1.0)) / np.pi) + w_r * sigma1 + 1.0  # we compute ntk #
            means.append(np.mean(ntk))  # we store mean #
            stds.append(np.std(ntk))  # we store std #
        means = np.array(means)  # we convert #
        stds = np.array(stds)  # we convert #
        ax.plot(rho_vals, det, "k-", lw=2, label="$K_\\infty(\\rho)$")  # we plot det #
        ax.plot(rho_vals, means, "b--", lw=1.5, label="mean empirical")  # we plot mean #
        ax.fill_between(rho_vals, means - 2 * stds, means + 2 * stds, color="C0", alpha=0.25, label="±2σ")  # we fill #
        ax.set_title(f"rank r={r}")  # we set title #
        ax.grid(True, alpha=0.3)  # we grid #
        ax.set_xlabel("$\\rho$")  # we label #
    axes[0].set_ylabel("NTK value")  # we set ylabel #
    axes[0].legend(fontsize=9)  # we legend #
    fig.suptitle("Three-layer NTK concentration (approximate Fisher-Kibble)")  # we title #
    return savefig(fig, "fig_plot2_ntk_concentration")  # we save #


def plot3_spectral_decay(d: int = 10):
    k = np.arange(1, 1001)  # we set k grid #
    depths = [1, 2, 3, 5, 10]  # we set depths #
    fig, ax = plt.subplots(figsize=(8, 6))  # we create fig #
    ax.loglog(k, k ** (-d), "k--", lw=2, alpha=0.6, label=f"reference k^-{d}")  # we plot ref #
    colors = plt.cm.plasma(np.linspace(0, 1, len(depths)))  # we colors #
    for L, c in zip(depths, colors):
        ax.loglog(k, k ** (-d), color=c, lw=1.5, label=f"L={L}")  # we plot same decay #
    ax.set_xlabel("spherical harmonic index k")  # we label #
    ax.set_ylabel("eigenvalue μ_k")  # we label #
    ax.set_title("Spectral decay: RKHS equivalence across depths (conceptual)")  # we title #
    ax.grid(True, which="both", alpha=0.3)  # we grid #
    ax.legend(ncol=2, fontsize=9)  # we legend #
    return savefig(fig, "fig_plot3_spectral_decay")  # we save #


def plot4_mp_spectrum():
    master_index = DATA_DIR / "grid_master_index.json"  # we set index path #
    if not master_index.exists():  # we check existence #
        # fallback: no grid data; create placeholder #
        fig, ax = plt.subplots(figsize=(8, 6))  # we fig #
        ax.text(0.5, 0.5, "No grid_master_index.json found", ha="center", va="center", transform=ax.transAxes)  # we annotate #
        ax.set_axis_off()  # we hide #
        return savefig(fig, "fig_plot4_mp_spectrum_placeholder")  # we save #
    with open(master_index, "r") as f:
        idx = json.load(f)  # we load index #
    # group by gamma approximately #
    entries = idx.get("index", [])  # we get entries #
    # pick up to three different gamma ratios by reading corresponding metadata #
    gamma_to_files = {}  # we init dict #
    for e in entries:
        meta_path = Path(e["meta"])  # we meta path #
        try:
            with open(meta_path, "r") as mf:
                meta = json.load(mf)  # we read #
            gamma = float(meta.get("gamma_ratio", np.nan))  # we get gamma #
            if np.isnan(gamma):  # we skip bad #
                continue
            gamma_key = f"{gamma:.2f}"  # we key #
            gamma_to_files.setdefault(gamma_key, []).append((e["file"], meta))  # we append #
        except Exception:
            continue
    # choose up to three gammas #
    chosen = sorted(gamma_to_files.keys(), key=lambda g: abs(float(g) - 1.0))[:3]  # we choose #
    if not chosen:
        fig, ax = plt.subplots(figsize=(8, 6))  # we fig #
        ax.text(0.5, 0.5, "No valid gamma configs found", ha="center", va="center", transform=ax.transAxes)  # we annotate #
        ax.set_axis_off()  # we hide #
        return savefig(fig, "fig_plot4_mp_spectrum_placeholder")  # we save #
    fig, axes = plt.subplots(1, len(chosen), figsize=(6 * len(chosen), 5), sharey=True)  # we subplots #
    if len(chosen) == 1:
        axes = [axes]  # we normalize #
    for ax, gk in zip(axes, chosen):
        files = gamma_to_files[gk]  # we get files #
        # use the first config #
        spec_path = Path(files[0][0])  # we get spec path #
        meta = files[0][1]  # we get meta #
        data = np.load(spec_path, allow_pickle=True)  # we load spectra #
        ev_mean = data["eigenvalues_mean"]  # we get mean spectrum #
        ax.hist(ev_mean, bins=50, density=True, alpha=0.6, color="steelblue", label="bulk spectrum (mean)")  # we hist #
        # overlay MP support as vertical lines #
        mp = meta.get("mp_params", {})  # we get mp params #
        if mp:
            support = mp.get("support", [None, None])  # we support #
            if support[0] is not None:
                ax.axvline(support[0], color="k", ls="--", lw=2, label="MP support")  # we left edge #
            if support[1] is not None:
                ax.axvline(support[1], color="k", ls="--", lw=2)  # we right edge #
        # spike #
        ax.axvline(np.max(ev_mean), color="r", ls=":", lw=2, label="spike (max)")  # we spike #
        ax.set_title(f"γ = {gk}")  # we title #
        ax.set_xlabel("eigenvalue λ")  # we label #
        ax.grid(True, alpha=0.3)  # we grid #
    axes[0].set_ylabel("density")  # we ylabel #
    axes[0].legend(fontsize=9)  # we legend #
    return savefig(fig, "fig_plot4_mp_spectrum")  # we save #


def plot5_efficiency():
    N_vals = np.logspace(1, 3, 100)  # we N grid #
    ranks = [10, 30, 100]  # we ranks #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # we panes #
    for ax, r in zip(axes, ranks):
        params_rflr = r * N_vals  # we compute rN #
        params_mlp = N_vals ** 2  # we compute N^2 #
        ax.loglog(N_vals, params_rflr, "b-", lw=2, label=f"RF-LR (r={r})")  # we plot #
        ax.loglog(N_vals, params_mlp, "r--", lw=2, label="MLP")  # we plot #
        ax.set_title(f"rank r={r}")  # we title #
        ax.set_xlabel("width N")  # we label #
        ax.grid(True, which="both", alpha=0.3)  # we grid #
        ax.legend(fontsize=9)  # we legend #
    axes[0].set_ylabel("parameter count")  # we ylabel #
    return savefig(fig, "fig_plot5_efficiency")  # we save #


def plot6_fisher_kibble():
    r = 30  # we set rank #
    rho_true_vals = [0.0, 0.5, 0.9]  # we set rhos #
    n_samples = 20000  # we set samples #
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # we panes #
    for ax, rho_true in zip(axes, rho_true_vals):
        rho_hat = approx_sample_fisher(rho_true, r, n_samples)  # we sample fisher #
        x_norm = np.linalg.norm(RNG.standard_normal((n_samples, r)), axis=1)  # we sample norm #
        y_norm = np.linalg.norm(RNG.standard_normal((n_samples, r)), axis=1)  # we sample norm #
        w_r = (x_norm * y_norm) / r  # we compute w_r #
        ax.scatter(rho_hat, w_r, s=2, alpha=0.2)  # we scatter #
        ax.axhline(1.0, color="r", ls="--", alpha=0.6)  # we mean line #
        ax.set_title(f"ρ={rho_true}")  # we title #
        ax.set_xlabel("Fisher ρ̂")  # we label #
        ax.grid(True, alpha=0.3)  # we grid #
    axes[0].set_ylabel("w_r = ||x_1||·||y_1||/r")  # we ylabel #
    return savefig(fig, "fig_plot6_fisher_kibble")  # we save #


def plot7_puiseux():
    r = 30  # we set rank #
    rho_vals = np.linspace(0.9, 1.0, 400)  # we rho grid #
    n_mc = 20000  # we samples #
    mean_vals = []  # we init #
    det_vals = det_ntk_3layer(rho_vals)  # we det #
    for rho in rho_vals:
        rho_hat = approx_sample_fisher(float(rho), r, n_mc)  # we fisher #
        x_norm = np.linalg.norm(RNG.standard_normal((n_mc, r)), axis=1)  # we norm #
        y_norm = np.linalg.norm(RNG.standard_normal((n_mc, r)), axis=1)  # we norm #
        w_r = (x_norm * y_norm) / r  # we w_r #
        theta1 = relu_kernel_theta1(rho_hat)  # we theta1 #
        sigma1 = relu_kernel_sigma1(rho_hat)  # we sigma1 #
        ntk = theta1 * (1.0 - np.arccos(np.clip(rho_hat, -1.0, 1.0)) / np.pi) + w_r * sigma1 + 1.0  # we ntk #
        mean_vals.append(np.mean(ntk))  # we append #
    mean_vals = np.array(mean_vals)  # we convert #
    diff = np.abs(mean_vals - det_vals)  # we diff #
    t = 1.0 - rho_vals  # we t #
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [2, 1]})  # we fig #
    ax1.plot(rho_vals, det_vals, "k-", lw=2, label="deterministic")  # we plot det #
    ax1.plot(rho_vals, mean_vals, "b--", lw=2, label="mean over Fisher-Kibble (approx)")  # we plot mean #
    ax1.set_xlabel("ρ")  # we label #
    ax1.set_ylabel("NTK")  # we label #
    ax1.set_title("Mean NTK vs Deterministic (near ρ=1)")  # we title #
    ax1.legend(fontsize=9)  # we legend #
    ax1.grid(True, alpha=0.3)  # we grid #
    ax2.loglog(t, diff, "r-", lw=2, label="|difference|")  # we loglog #
    ax2.loglog(t, np.sqrt(t), "k--", lw=1, alpha=0.6, label="t^{1/2} reference")  # we ref #
    ax2.set_xlabel("t = 1-ρ")  # we label #
    ax2.set_ylabel("|difference|")  # we label #
    ax2.grid(True, which="both", alpha=0.3)  # we grid #
    ax2.legend(fontsize=9)  # we legend #
    return savefig(fig, "fig_plot7_puiseux")  # we save #


def plot8_recursion():
    L_vals = np.arange(1, 11)  # we L grid #
    rhos = [0.0, 0.5, 0.9]  # we rho list #
    fig, ax = plt.subplots(figsize=(8, 6))  # we fig #
    for rho in rhos:
        # simple recursion: Θ^{(0)}=0; for ℓ>=1: Θ^{(ℓ)} = 1 + Θ^{(ℓ-1)}·dotΣ^{(ℓ)} + Σ^{(ℓ)}; we assume stationarity across ℓ for demo #
        Sigma = relu_kernel_sigma1(np.array([rho]))[0]  # we sigma #
        Theta_dot = relu_kernel_theta1(np.array([rho]))[0]  # we theta dot #
        K = 0.0  # we init #
        vals = []  # we list #
        for L in L_vals:
            K = 1.0 + K * Theta_dot + Sigma  # we update #
            vals.append(K)  # we append #
        ax.plot(L_vals, vals, "o-", label=f"ρ={rho}")  # we plot #
    ax.set_xlabel("depth L")  # we label #
    ax.set_ylabel("NTK value")  # we label #
    ax.set_title("NTK recursion visualization (conceptual)")  # we title #
    ax.grid(True, alpha=0.3)  # we grid #
    ax.legend()  # we legend #
    return savefig(fig, "fig_plot8_recursion")  # we save #


def plot9_concentration_vs_params():
    P = np.logspace(2, 6, 100)  # we param grid #
    # assume r ∝ sqrt(P) for RF-LR with fixed N #
    std_rflr = P ** (-1.0)  # we assume 1/r^2 with r ~ sqrt(P) => 1/P #
    std_mlp = P ** (-0.5)  # we assume 1/sqrt(P) #
    fig, ax = plt.subplots(figsize=(8, 6))  # we fig #
    ax.loglog(P, std_rflr, "b-", lw=2, label="RF-LR (theory)")  # we plot #
    ax.loglog(P, std_mlp, "r--", lw=2, label="MLP (theory)")  # we plot #
    ax.set_xlabel("parameter count P")  # we label #
    ax.set_ylabel("NTK standard deviation")  # we label #
    ax.set_title("Concentration vs parameter count")  # we title #
    ax.grid(True, which="both", alpha=0.3)  # we grid #
    ax.legend()  # we legend #
    return savefig(fig, "fig_plot9_concentration")  # we save #


def plot10_training_placeholder():
    # placeholder: synthetic power-law decays #
    flops = np.logspace(6, 10, 200)  # we grid #
    loss_rflr = flops ** (-0.7)  # we synthetic #
    loss_mlp = flops ** (-0.5)  # we synthetic #
    fig, ax = plt.subplots(figsize=(8, 6))  # we fig #
    ax.loglog(flops, loss_rflr, "b-", lw=2, label="RF-LR (synthetic)")  # we plot #
    ax.loglog(flops, loss_mlp, "r--", lw=2, label="MLP (synthetic)")  # we plot #
    ax.set_xlabel("FLOPs")  # we label #
    ax.set_ylabel("loss")  # we label #
    ax.set_title("Training dynamics vs FLOPs (placeholder)")  # we title #
    ax.grid(True, which="both", alpha=0.3)  # we grid #
    ax.legend()  # we legend #
    return savefig(fig, "fig_plot10_training")  # we save #


def main():
    generated = {}  # we init map #
    generated["plot1"] = plot1_rank_concentration()  # we gen #
    generated["plot2"] = plot2_ntk_concentration()  # we gen #
    generated["plot3"] = plot3_spectral_decay()  # we gen #
    generated["plot4"] = plot4_mp_spectrum()  # we gen #
    generated["plot5"] = plot5_efficiency()  # we gen #
    generated["plot6"] = plot6_fisher_kibble()  # we gen #
    generated["plot7"] = plot7_puiseux()  # we gen #
    generated["plot8"] = plot8_recursion()  # we gen #
    generated["plot9"] = plot9_concentration_vs_params()  # we gen #
    generated["plot10"] = plot10_training_placeholder()  # we gen #
    out = {
        "created_at": datetime.now().isoformat(),
        "figures": generated,
    }  # we build out #
    with open(DATA_DIR / "figures_index.json", "w") as f:
        json.dump(out, f, indent=2)  # we save index #
    print("All figures generated.")  # we print #
    print(json.dumps(out, indent=2))  # we echo summary #


if __name__ == "__main__":
    main()  # we run main #


