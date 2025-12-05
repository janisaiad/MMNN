"""
Generate individual Marchenko-Pastur plots for EACH configuration.
Each plot shows bulk eigenvalue histogram with fitted MP density.

Outputs to: figures/paper/mp_individual/
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

# matplotlib configuration
plt.rcParams['figure.figsize'] = [10, 7]
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 16
mpl.rcParams['savefig.dpi'] = 200

# directories
DATA_DIR = Path("refs/paper/data")
OUTPUT_DIR = Path("figures/paper/mp_individual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# color scheme
cmap = mpl.colormaps['Dark2']
colors = cmap.colors


def marchenko_pastur_density(x, gamma, center=1.0, scale=1.0):
    """
    marchenko-pastur density with adjustable center and scale
    
    rho(lambda) = (1/(2*pi*gamma)) * sqrt((b-lambda)*(lambda-a)) / lambda
    where a = center + scale*(1-sqrt(gamma))^2, b = center + scale*(1+sqrt(gamma))^2
    """
    x = np.atleast_1d(x)  # we ensure array #
    a = center + scale * (1 - np.sqrt(gamma))**2  # we compute lower edge #
    b = center + scale * (1 + np.sqrt(gamma))**2  # we compute upper edge #
    
    density = np.zeros_like(x, dtype=float)  # we initialize density #
    mask = (x >= a) & (x <= b)  # we mask support #
    
    if np.any(mask):
        x_in = x[mask]  # we extract x in support #
        density[mask] = (1.0 / (2.0 * np.pi * gamma)) * \
                        np.sqrt(np.maximum(0, (b - x_in) * (x_in - a))) / np.maximum(x_in, 1e-10)  # we compute density #
    
    return density  # we return density #


def fit_mp_to_bulk(bulk_eigenvalues, gamma_ratio):
    """
    fit marchenko-pastur parameters to bulk eigenvalues
    
    returns fitted center, scale, and goodness of fit
    """
    # initial guess: center at median, scale from range
    center_init = np.median(bulk_eigenvalues)  # we guess center #
    range_bulk = bulk_eigenvalues.max() - bulk_eigenvalues.min()  # we compute range #
    theoretical_range = 4 * np.sqrt(gamma_ratio)  # we compute theoretical range #
    scale_init = range_bulk / theoretical_range if theoretical_range > 0 else 1.0  # we guess scale #
    
    # create histogram for fitting
    n_bins_fit = 50  # we set bins for fitting #
    counts, bin_edges = np.histogram(bulk_eigenvalues, bins=n_bins_fit, density=True)  # we compute histogram #
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # we compute centers #
    
    # filter out zero counts
    mask_nonzero = counts > 0  # we mask nonzero #
    x_fit = bin_centers[mask_nonzero]  # we get x for fitting #
    y_fit = counts[mask_nonzero]  # we get y for fitting #
    
    if len(x_fit) < 10:
        # not enough points for fitting
        return center_init, scale_init, 0.0  # we return defaults #
    
    try:
        # fit MP density
        def mp_wrapper(x, center, scale):
            return marchenko_pastur_density(x, gamma_ratio, center, scale)  # we wrap mp #
        
        # bounds: center and scale must be positive
        bounds = ([0, 0.01], [bulk_eigenvalues.max() * 2, 10.0])  # we set bounds #
        
        popt, pcov = curve_fit(mp_wrapper, x_fit, y_fit, 
                              p0=[center_init, scale_init],
                              bounds=bounds,
                              maxfev=5000)  # we fit #
        
        center_fitted, scale_fitted = popt  # we extract fitted params #
        
        # compute R^2
        y_pred = mp_wrapper(x_fit, *popt)  # we predict #
        ss_res = np.sum((y_fit - y_pred)**2)  # we compute residual sum of squares #
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)  # we compute total sum of squares #
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0  # we compute r squared #
        
        return center_fitted, scale_fitted, r_squared  # we return fitted params #
        
    except Exception as e:
        print(f"    Fitting failed: {e}")  # we log error #
        return center_init, scale_init, 0.0  # we return defaults #


def plot_single_mp_config(config_file, meta_file):
    """
    generate mp plot for single configuration
    """
    # load metadata
    with open(meta_file, 'r') as f:
        meta = json.load(f)  # we load metadata #
    
    n = meta["n"]  # we extract n #
    r = meta["r"]  # we extract r #
    d = meta["d"]  # we extract d #
    N = meta.get("N", meta.get("n1", 0))  # we extract width #
    gamma_ratio = meta["gamma_ratio"]  # we extract gamma #
    alpha_ratio = meta.get("alpha_ratio", d/r if r > 0 else 1.0)  # we extract alpha #
    
    # load eigenvalues
    data = np.load(config_file, allow_pickle=True)  # we load data #
    eigenvalues_mean = data["eigenvalues_mean"]  # we extract eigenvalues #
    lambda_spike_mean = float(data["lambda_spike_mean"])  # we extract spike #
    
    # separate spike from bulk
    spike_threshold = 10.0  # we set threshold #
    bulk_mask = eigenvalues_mean < spike_threshold  # we mask bulk #
    bulk_eigenvalues = eigenvalues_mean[bulk_mask]  # we extract bulk #
    
    # theoretical bounds
    a_theory = (1 - np.sqrt(gamma_ratio))**2  # we compute lower edge #
    b_theory = (1 + np.sqrt(gamma_ratio))**2  # we compute upper edge #
    
    # clean bulk
    bulk_clean = bulk_eigenvalues[(bulk_eigenvalues >= max(0, a_theory * 0.5)) & 
                                  (bulk_eigenvalues <= b_theory * 1.5)]  # we filter #
    
    if len(bulk_clean) < 20:
        print(f"  Skipping (too few bulk eigenvalues): {config_file.name}")  # we log #
        return None  # we skip #
    
    # fit MP to bulk
    center_fit, scale_fit, r_squared = fit_mp_to_bulk(bulk_clean, gamma_ratio)  # we fit mp #
    
    # create plot
    fig, ax = plt.subplots(figsize=(10, 7))  # we create figure #
    
    # histogram with Freedman-Diaconis rule
    iqr = np.percentile(bulk_clean, 75) - np.percentile(bulk_clean, 25)  # we compute iqr #
    bin_width = 2.0 * iqr * len(bulk_clean)**(-1.0/3.0)  # we apply diaconis #
    if bin_width > 0:
        n_bins = int(np.ceil((bulk_clean.max() - bulk_clean.min()) / bin_width))  # we compute bins #
        n_bins = min(max(n_bins, 20), 80)  # we clip #
    else:
        n_bins = 40  # we use fallback #
    
    counts, bin_edges = np.histogram(bulk_clean, bins=n_bins, density=True)  # we compute histogram #
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # we compute centers #
    
    # plot histogram
    ax.bar(bin_centers, counts,
           width=(bin_edges[1] - bin_edges[0]),
           alpha=0.4,
           color=colors[0],
           label='Empirical bulk',
           edgecolor='black',
           linewidth=0.5)  # we plot histogram #
    
    # plot fitted MP density
    x_mp = np.linspace(bulk_clean.min() * 0.95, bulk_clean.max() * 1.05, 500)  # we set x range #
    mp_fitted = marchenko_pastur_density(x_mp, gamma_ratio, center_fit, scale_fit)  # we compute fitted mp #
    ax.plot(x_mp, mp_fitted,
            color='red',
            linestyle='-',
            linewidth=3,
            label=f'Fitted MP ($R^2={r_squared:.3f}$)')  # we plot fitted mp #
    
    # plot theoretical MP (for comparison)
    mp_theory = marchenko_pastur_density(x_mp, gamma_ratio, center=0.0, scale=1.0)  # we compute theoretical #
    ax.plot(x_mp, mp_theory,
            color='black',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label='Theoretical MP (standard)')  # we plot theoretical #
    
    # mark theoretical support
    ax.axvline(a_theory, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)  # we mark lower edge #
    ax.axvline(b_theory, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)  # we mark upper edge #
    
    # styling
    ax.set_xlabel(r'Eigenvalue $\lambda$', fontsize=20)  # we set x label #
    ax.set_ylabel(r'Density $\rho(\lambda)$', fontsize=20)  # we set y label #
    ax.set_xlim(bulk_clean.min() * 0.9, bulk_clean.max() * 1.1)  # we set x limits #
    ax.set_ylim(0, None)  # we set y limits #
    ax.grid(True, alpha=0.3)  # we add grid #
    ax.legend(fontsize=14, loc='upper right')  # we add legend #
    
    # title with config details
    title = f'Marchenko-Pastur Fit\n'  # we start title #
    title += f'$N={N}$, $n={n}$, $r={r}$, $d={d}$ '  # we add params #
    title += f'($\\gamma={gamma_ratio:.2f}$, $\\alpha={alpha_ratio:.2f}$, $N/\\max(r,d)={N/max(r,d):.1f}$)'  # we add ratios #
    ax.set_title(title, fontsize=16, pad=15)  # we set title #
    
    # info box
    info_text = f'Spike: $\\lambda_1 = {lambda_spike_mean:.1f}$\n'  # we add spike #
    info_text += f'Bulk: $[{bulk_clean.min():.4f}, {bulk_clean.max():.4f}]$\n'  # we add bulk range #
    info_text += f'Fitted: center={center_fit:.3f}, scale={scale_fit:.3f}\n'  # we add fitted params #
    info_text += f'Theory: $a={a_theory:.3f}$, $b={b_theory:.3f}$'  # we add theory #
    
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))  # we add info box #
    
    plt.tight_layout()  # we adjust layout #
    
    # save
    output_name = config_file.stem + "_mp.png"  # we create filename #
    output_path = OUTPUT_DIR / output_name  # we create path #
    plt.savefig(output_path, bbox_inches='tight', dpi=200)  # we save #
    plt.close()  # we close #
    
    return {
        "file": output_name,
        "n": n,
        "r": r,
        "d": d,
        "N": N,
        "gamma": gamma_ratio,
        "alpha": alpha_ratio,
        "spike": lambda_spike_mean,
        "center_fit": center_fit,
        "scale_fit": scale_fit,
        "r_squared": r_squared
    }  # we return results #


def main():
    print("=" * 80)
    print("INDIVIDUAL MARCHENKO-PASTUR PLOTS GENERATION")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # search for all grid files (both original and large-width)
    grid_files = []  # we initialize list #
    
    # original data
    original_files = sorted(DATA_DIR.glob("grid_n*_N*_r*_d*.npz"))  # we search original #
    original_files = [f for f in original_files if "_ntk_rho" not in f.name]  # we filter #
    grid_files.extend(original_files)  # we add #
    
    # large-width data
    lw_dir = DATA_DIR / "largewidth"  # we set lw dir #
    if lw_dir.exists():
        lw_files = sorted(lw_dir.glob("lw_N*_n*_r*_d*.npz"))  # we search lw #
        lw_files = [f for f in lw_files if "_ntk_rho" not in f.name]  # we filter #
        grid_files.extend(lw_files)  # we add #
    
    print(f"Found {len(grid_files)} total configuration files\n")  # we log count #
    
    results = []  # we initialize results #
    
    for i, grid_file in enumerate(grid_files):
        print(f"[{i+1}/{len(grid_files)}] Processing {grid_file.name}...")  # we log progress #
        
        # find corresponding metadata
        meta_file = grid_file.with_name(grid_file.stem + "_metadata.json")  # we get meta path #
        
        if not meta_file.exists():
            print(f"  Warning: metadata not found. Skipping.")  # we warn #
            continue  # we skip #
        
        try:
            result = plot_single_mp_config(grid_file, meta_file)  # we plot #
            if result is not None:
                results.append(result)  # we append result #
                print(f"  ✓ Saved: {result['file']}")  # we log success #
        except Exception as e:
            print(f"  ✗ Error: {e}")  # we log error #
            continue  # we skip #
    
    print("\n" + "=" * 80)
    print(f"COMPLETED: {len(results)} plots generated")
    print("=" * 80)
    
    # save index
    index_path = OUTPUT_DIR / "mp_individual_index.json"  # we set index path #
    with open(index_path, 'w') as f:
        json.dump({
            "total_plots": len(results),
            "output_directory": str(OUTPUT_DIR),
            "plots": results
        }, f, indent=2)  # we save index #
    
    print(f"\nIndex saved to: {index_path}")  # we log #
    print(f"All plots in: {OUTPUT_DIR}")  # we log #


if __name__ == "__main__":
    main()  # we run main #

