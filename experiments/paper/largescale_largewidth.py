"""
Large-width NTK computation: N >> r, d regime.

This script computes NTK Gram matrices in the PROPER infinite-width regime:
- Widths: N ∈ {2048, 4096, 8192, 16384, 32768, 65536}
- Vary n/r ratios: gamma ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
- Vary r/d ratios: alpha ∈ {0.5, 1.0, 2.0}

Key constraint: N >> max(r, d) to ensure infinite-width limit.

Output: Stores eigenvalues and NTK-rho distributions.
"""

import numpy as np
import scipy.linalg
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import sys
import platform
import hashlib
warnings.filterwarnings('ignore')

# add parent to path to import from largescale.py
sys.path.insert(0, str(Path(__file__).parent))
from largescale import (
    compute_ntk_gram_matrix,
    initialize_rflr_network,
    generate_data,
    compute_theoretical_ntk_limit,
    compute_mp_density_params,
    _compute_ntk_rho_distributions_for_config
)

BASE_SEED = 20250201  # new seed for large-width runs
OUTPUT_DIR = Path("refs/paper/data/largewidth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_largewidth_computation(
    width_powers=range(11, 17),        # N ∈ {2048, 4096, 8192, 16384, 32768, 65536}
    gamma_ratios=[0.25, 0.5, 1.0, 2.0, 4.0],  # n/r ratios
    alpha_ratios=[0.5, 1.0, 2.0],      # r/d ratios
    base_r=128,                        # base rank (will scale)
    n_init=5,                          # initializations per config
    do_ntk_rho=True,                   # compute NTK-vs-rho
    rho_step=0.1,
    samples_per_rho=2000
):
    """
    run large-width grid computation
    
    constraint: N >> r, d for infinite-width regime
    we ensure N >= 4 * max(r, d) as minimum separation
    """
    print("=" * 80)
    print("LARGE-WIDTH NTK COMPUTATION")
    print("=" * 80)
    print(f"Width powers:    {list(width_powers)}")
    print(f"Gamma ratios:    {gamma_ratios}")
    print(f"Alpha ratios:    {alpha_ratios}")
    print(f"Base rank:       {base_r}")
    print(f"n_init:          {n_init}")
    print("=" * 80)
    
    # theoretical kernel values
    K_infty_0 = compute_theoretical_ntk_limit(0.0)  # we compute k at 0 #
    eps = 1e-5  # we set epsilon #
    K_infty_prime_0 = (compute_theoretical_ntk_limit(eps) - K_infty_0) / eps  # we compute derivative #
    K_infty_diag = compute_theoretical_ntk_limit(1.0)  # we compute diagonal #
    
    total_configs = len(list(width_powers)) * len(gamma_ratios) * len(alpha_ratios)  # we count configs #
    cfg_idx = 0  # we initialize counter #
    master_index = []  # we initialize index #
    ntk_rho_index = []  # we initialize ntk rho index #
    
    for N_pow in width_powers:
        N = 2 ** N_pow  # we compute width #
        n1 = n2 = N  # we set layer widths #
        
        for gamma in gamma_ratios:
            for alpha in alpha_ratios:
                cfg_idx += 1  # we increment counter #
                
                # determine r, d, n from ratios and base_r
                # scale r with width power to keep N >> r
                r = base_r * max(1, 2 ** (N_pow - 13))  # we scale r with width #
                r = min(r, N // 8)  # we ensure N >> r (at least 8x) #
                
                # determine d from alpha = r/d
                d = int(r / alpha)  # we compute d from ratio #
                d = max(d, 16)  # we set minimum d #
                
                # determine n from gamma = n/r
                n = int(gamma * r)  # we compute n from ratio #
                n = max(n, 16)  # we set minimum n #
                n = min(n, 2048)  # we set maximum n for memory #
                
                # verify large-width regime
                if N < 4 * max(r, d):
                    print(f"  [WARNING] N={N} not large enough for r={r}, d={d}. Skipping.")  # we warn #
                    continue  # we skip #
                
                print(f"\n[{cfg_idx}/{total_configs}] N={N}, n={n}, r={r}, d={d}")  # we log config #
                print(f"  Ratios: gamma={gamma:.2f}, alpha={alpha:.2f}")  # we log ratios #
                print(f"  Regime check: N/{max(r,d)} = {N/max(r,d):.1f} (need >> 1)")  # we check regime #
                
                # filenames
                config_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}.npz"  # we set filename #
                config_meta_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}_metadata.json"  # we set meta filename #
                
                if config_file.exists() and config_meta_file.exists():
                    print("  Found existing results. Skipping.")  # we log #
                    master_index.append({"file": str(config_file), "meta": str(config_meta_file)})  # we append index #
                    continue  # we skip #
                
                # per-config seed
                s = f"{BASE_SEED}|N={N}|n={n}|r={r}|d={d}"  # we create seed string #
                cfg_seed_hash = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()  # we hash #
                cfg_seed = int(cfg_seed_hash, 16) & 0xFFFFFFFF  # we get seed #
                
                eigenvalues_config = []  # we initialize eigenvalues #
                lambda_spike_config = []  # we initialize spikes #
                seeds_data = []  # we initialize data seeds #
                seeds_init = []  # we initialize init seeds #
                flops_config = 0.0  # we initialize flops #
                
                for init_idx in range(n_init):
                    seed_data = (cfg_seed ^ (0x9E3779B9 + init_idx * 0x85EBCA6B)) & 0xFFFFFFFF  # we derive data seed #
                    seed_init = (cfg_seed ^ (0xC2B2AE35 + init_idx * 0x27D4EB2F)) & 0xFFFFFFFF  # we derive init seed #
                    seeds_data.append(int(seed_data))  # we append #
                    seeds_init.append(int(seed_init))  # we append #
                    
                    rng_data = np.random.default_rng(seed_data)  # we create rng #
                    rng_init = np.random.default_rng(seed_init)  # we create rng #
                    
                    # generate data on unit sphere
                    X = generate_data(n, d, covariance_type="isotropic", rng=rng_data)  # we generate data #
                    
                    # initialize under EOC
                    w1, w2, b1, b2, A1, A2, c = initialize_rflr_network(
                        n1, n2, r, d, rng=rng_init, beta=1.0, sigma_A=np.sqrt(2.0), sigma_c=1.0
                    )  # we initialize network #
                    
                    # compute NTK Gram matrix
                    K, flops_k = compute_ntk_gram_matrix(X, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d)  # we compute ntk #
                    flops_config += float(flops_k)  # we accumulate flops #
                    
                    # eigendecompose
                    ev = np.linalg.eigvalsh(K)  # we compute eigenvalues #
                    ev = np.sort(ev)[::-1]  # we sort descending #
                    eigenvalues_config.append(ev)  # we append #
                    lambda_spike_config.append(float(ev[0]))  # we append spike #
                    
                    if (init_idx + 1) % max(1, n_init // 5) == 0:
                        print(f"    init {init_idx + 1}/{n_init} done")  # we log progress #
                
                # aggregate
                eigenvalues_mean = np.mean(eigenvalues_config, axis=0)  # we compute mean #
                lambda_spike_mean = float(np.mean(lambda_spike_config))  # we compute spike mean #
                lambda_spike_std = float(np.std(lambda_spike_config))  # we compute spike std #
                mp_params = compute_mp_density_params(gamma, K_infty_0, K_infty_prime_0, K_infty_diag)  # we compute mp params #
                
                # save spectra
                np.savez_compressed(
                    config_file,
                    eigenvalues_per_init=np.array(eigenvalues_config, dtype=object),
                    eigenvalues_mean=eigenvalues_mean,
                    lambda_spike_per_init=np.array(lambda_spike_config),
                    lambda_spike_mean=lambda_spike_mean,
                    lambda_spike_std=lambda_spike_std,
                    gamma_ratio=gamma,
                    alpha_ratio=alpha,
                    n=n,
                    r=r,
                    d=d,
                    n1=n1,
                    n2=n2,
                    N=N,
                    seeds_data=np.array(seeds_data, dtype=np.uint32),
                    seeds_init=np.array(seeds_init, dtype=np.uint32),
                    flops_config=float(flops_config)
                )  # we save spectra #
                
                meta_cfg = {
                    "computation_date": datetime.now().isoformat(),
                    "base_seed": BASE_SEED,
                    "config_seed": cfg_seed,
                    "n_init": n_init,
                    "gamma_ratio": gamma,
                    "alpha_ratio": alpha,
                    "n": n,
                    "r": r,
                    "d": d,
                    "n1": n1,
                    "n2": n2,
                    "N": N,
                    "regime_check": f"N/max(r,d) = {N/max(r,d):.2f}",
                    "lambda_spike_mean": lambda_spike_mean,
                    "lambda_spike_std": lambda_spike_std,
                    "mp_params": mp_params,
                    "flops_config": float(flops_config),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "notes": f"Large-width regime: N={N} >> max(r={r}, d={d}); EOC sigma_A=sqrt(2); isotropic data."
                }  # we create metadata #
                
                with open(config_meta_file, "w") as f:
                    json.dump(meta_cfg, f, indent=2)  # we save metadata #
                
                master_index.append({"file": str(config_file), "meta": str(config_meta_file)})  # we append index #
                print(f"  spike={lambda_spike_mean:.2f}±{lambda_spike_std:.2f}, FLOPs={flops_config:.2e}")  # we log results #
                
                # optional: NTK-vs-rho
                if do_ntk_rho:
                    ntk_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}_ntk_rho.npz"  # we set ntk filename #
                    ntk_meta_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}_ntk_rho_metadata.json"  # we set ntk meta #
                    
                    if not (ntk_file.exists() and ntk_meta_file.exists()):
                        rho_vals = np.round(np.arange(-1.0, 1.0 + 1e-9, rho_step), 6)  # we set rho buckets #
                        seed_data_ntk = (cfg_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF  # we derive seed #
                        seed_init_ntk = (cfg_seed ^ 0x5A5A5A5A) & 0xFFFFFFFF  # we derive seed #
                        rng_data_ntk = np.random.default_rng(seed_data_ntk)  # we create rng #
                        rng_init_ntk = np.random.default_rng(seed_init_ntk)  # we create rng #
                        
                        print("  Computing NTK-vs-rho distributions...")  # we log #
                        ntk_res = _compute_ntk_rho_distributions_for_config(
                            n1, n2, r, d, rho_vals, samples_per_rho, rng_init_ntk, rng_data_ntk
                        )  # we compute ntk rho #
                        
                        np.savez_compressed(
                            ntk_file,
                            rho_vals=ntk_res["rho_vals"],
                            ntk_samples=ntk_res["ntk_samples"],
                            ntk_mean=ntk_res["ntk_mean"],
                            ntk_std=ntk_res["ntk_std"],
                            k_infty=ntk_res["k_infty"],
                            n=n,
                            r=r,
                            d=d,
                            n1=n1,
                            n2=n2,
                            N=N,
                            samples_per_rho=samples_per_rho
                        )  # we save ntk data #
                        
                        meta_ntk = {
                            "computation_date": datetime.now().isoformat(),
                            "type": "ntk_rho_distribution",
                            "base_seed": BASE_SEED,
                            "config_seed": cfg_seed,
                            "seeds": {"data": int(seed_data_ntk), "init": int(seed_init_ntk)},
                            "n": n,
                            "r": r,
                            "d": d,
                            "n1": n1,
                            "n2": n2,
                            "N": N,
                            "gamma_ratio": gamma,
                            "alpha_ratio": alpha,
                            "rho_step": rho_step,
                            "samples_per_rho": samples_per_rho,
                            "notes": f"Large-width: N={N} >> max(r={r}, d={d})"
                        }  # we create ntk metadata #
                        
                        with open(ntk_meta_file, "w") as f:
                            json.dump(meta_ntk, f, indent=2)  # we save ntk metadata #
                        
                        ntk_rho_index.append({"file": str(ntk_file), "meta": str(ntk_meta_file)})  # we append #
    
    # save master index
    master_path = OUTPUT_DIR / "largewidth_master_index.json"  # we set master path #
    with open(master_path, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "base_seed": BASE_SEED,
            "width_powers": list(width_powers),
            "gamma_ratios": gamma_ratios,
            "alpha_ratios": alpha_ratios,
            "base_r": base_r,
            "n_init": n_init,
            "total_configs": total_configs,
            "index": master_index,
            "ntk_rho_index": ntk_rho_index,
            "notes": "Large-width regime: N >> max(r,d) for proper infinite-width limit"
        }, f, indent=2)  # we save master index #
    
    print("\nLarge-width computation complete.")  # we log #
    print(f"  Total configurations: {total_configs}")  # we log #
    print(f"  Master index saved to: {master_path}")  # we log #


def run_extensive_grid(
    width_powers=range(11, 18),         # N ∈ {2048, ..., 131072} = 2^11 to 2^17
    n_values=[32, 64, 128, 256, 512, 1024, 2048],  # many n values
    r_values=[16, 32, 64, 128, 256, 512, 1024],     # many r values
    d_values=[16, 32, 64, 128, 256, 512, 1024],     # many d values
    n_init=3,
    do_ntk_rho=True
):
    """
    extensive grid over (N, n, r, d) with proper large-width regime
    
    constraint: only keep configs where N >= 8 * max(r, d)
    """
    print("=" * 80)
    print("EXTENSIVE LARGE-WIDTH GRID COMPUTATION")
    print("=" * 80)
    print(f"Width powers:    {list(width_powers)} → N ∈ [{2**min(width_powers)}, {2**max(width_powers)}]")
    print(f"n values:        {n_values}")
    print(f"r values:        {r_values}")
    print(f"d values:        {d_values}")
    print(f"Constraint:      N >= 8 * max(r, d) (large-width regime)")
    print("=" * 80)
    
    # theoretical kernel values
    K_infty_0 = compute_theoretical_ntk_limit(0.0)  # we compute k at 0 #
    eps = 1e-5  # we set epsilon #
    K_infty_prime_0 = (compute_theoretical_ntk_limit(eps) - K_infty_0) / eps  # we compute derivative #
    K_infty_diag = compute_theoretical_ntk_limit(1.0)  # we compute diagonal #
    
    total_attempted = 0  # we initialize counter #
    total_computed = 0  # we initialize counter #
    total_skipped_regime = 0  # we initialize counter #
    master_index = []  # we initialize index #
    ntk_rho_index = []  # we initialize ntk index #
    
    for N_pow in width_powers:
        N = 2 ** N_pow  # we compute width #
        n1 = n2 = N  # we set layer widths #
        
        for n in n_values:
            for r in r_values:
                for d in d_values:
                    total_attempted += 1  # we increment counter #
                    
                    # check large-width regime
                    if N < 8 * max(r, d):
                        total_skipped_regime += 1  # we increment skipped #
                        continue  # we skip this config #
                    
                    # check memory constraint
                    if n > 2048:
                        continue  # we skip large n #
                    
                    gamma = n / r if r > 0 else 1.0  # we compute gamma #
                    alpha = r / d if d > 0 else 1.0  # we compute alpha #
                    
                    print(f"\n[{total_computed + 1}] N={N}, n={n}, r={r}, d={d}")  # we log config #
                    print(f"  Ratios: gamma={gamma:.2f}, alpha={alpha:.2f}, N/max(r,d)={N/max(r,d):.1f}")  # we log ratios #
                    
                    # filenames
                    config_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}.npz"  # we set filename #
                    config_meta_file = OUTPUT_DIR / f"lw_N{N}_n{n}_r{r}_d{d}_metadata.json"  # we set meta #
                    
                    if config_file.exists() and config_meta_file.exists():
                        print("  Found existing. Skipping.")  # we log #
                        master_index.append({"file": str(config_file), "meta": str(config_meta_file)})  # we append #
                        total_computed += 1  # we increment #
                        continue  # we skip #
                    
                    # per-config seed
                    s = f"{BASE_SEED}|N={N}|n={n}|r={r}|d={d}"  # we create seed string #
                    cfg_seed_hash = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()  # we hash #
                    cfg_seed = int(cfg_seed_hash, 16) & 0xFFFFFFFF  # we get seed #
                    
                    eigenvalues_config = []  # we initialize #
                    lambda_spike_config = []  # we initialize #
                    seeds_data = []  # we initialize #
                    seeds_init = []  # we initialize #
                    flops_config = 0.0  # we initialize #
                    
                    for init_idx in range(n_init):
                        seed_data = (cfg_seed ^ (0x9E3779B9 + init_idx * 0x85EBCA6B)) & 0xFFFFFFFF  # we derive #
                        seed_init = (cfg_seed ^ (0xC2B2AE35 + init_idx * 0x27D4EB2F)) & 0xFFFFFFFF  # we derive #
                        seeds_data.append(int(seed_data))  # we append #
                        seeds_init.append(int(seed_init))  # we append #
                        
                        rng_data = np.random.default_rng(seed_data)  # we create rng #
                        rng_init = np.random.default_rng(seed_init)  # we create rng #
                        
                        X = generate_data(n, d, covariance_type="isotropic", rng=rng_data)  # we generate data #
                        w1, w2, b1, b2, A1, A2, c = initialize_rflr_network(
                            n1, n2, r, d, rng=rng_init, beta=1.0, sigma_A=np.sqrt(2.0), sigma_c=1.0
                        )  # we initialize #
                        
                        K, flops_k = compute_ntk_gram_matrix(X, w1, w2, b1, b2, A1, A2, c, n1, n2, r, d)  # we compute #
                        flops_config += float(flops_k)  # we accumulate #
                        
                        ev = np.linalg.eigvalsh(K)  # we eigendecompose #
                        ev = np.sort(ev)[::-1]  # we sort #
                        eigenvalues_config.append(ev)  # we append #
                        lambda_spike_config.append(float(ev[0]))  # we append spike #
                    
                    eigenvalues_mean = np.mean(eigenvalues_config, axis=0)  # we aggregate #
                    lambda_spike_mean = float(np.mean(lambda_spike_config))  # we aggregate spike #
                    lambda_spike_std = float(np.std(lambda_spike_config))  # we aggregate spike std #
                    mp_params = compute_mp_density_params(gamma, K_infty_0, K_infty_prime_0, K_infty_diag)  # we compute mp #
                    
                    np.savez_compressed(
                        config_file,
                        eigenvalues_per_init=np.array(eigenvalues_config, dtype=object),
                        eigenvalues_mean=eigenvalues_mean,
                        lambda_spike_per_init=np.array(lambda_spike_config),
                        lambda_spike_mean=lambda_spike_mean,
                        lambda_spike_std=lambda_spike_std,
                        gamma_ratio=gamma,
                        alpha_ratio=alpha,
                        n=n, r=r, d=d, n1=n1, n2=n2, N=N,
                        seeds_data=np.array(seeds_data, dtype=np.uint32),
                        seeds_init=np.array(seeds_init, dtype=np.uint32),
                        flops_config=float(flops_config)
                    )  # we save #
                    
                    meta_cfg = {
                        "computation_date": datetime.now().isoformat(),
                        "base_seed": BASE_SEED,
                        "config_seed": cfg_seed,
                        "n_init": n_init,
                        "gamma_ratio": gamma,
                        "alpha_ratio": alpha,
                        "n": n, "r": r, "d": d, "n1": n1, "n2": n2, "N": N,
                        "regime_check": f"N/max(r,d) = {N/max(r,d):.2f}",
                        "lambda_spike_mean": lambda_spike_mean,
                        "lambda_spike_std": lambda_spike_std,
                        "mp_params": mp_params,
                        "flops_config": float(flops_config),
                        "python": sys.version,
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                        "notes": f"Large-width: N={N} >> max(r={r}, d={d})"
                    }  # we create metadata #
                    
                    with open(config_meta_file, "w") as f:
                        json.dump(meta_cfg, f, indent=2)  # we save #
                    
                    master_index.append({"file": str(config_file), "meta": str(config_meta_file)})  # we append #
                    total_computed += 1  # we increment #
                    print(f"  spike={lambda_spike_mean:.2f}, FLOPs={flops_config:.2e}, SAVED")  # we log #
    
    # save master index
    master_path = OUTPUT_DIR / "largewidth_extensive_index.json"  # we set path #
    with open(master_path, "w") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "base_seed": BASE_SEED,
            "width_powers": list(width_powers),
            "n_values": n_values,
            "r_values": r_values,
            "d_values": d_values,
            "n_init": n_init,
            "total_attempted": total_attempted,
            "total_computed": total_computed,
            "total_skipped_regime": total_skipped_regime,
            "index": master_index,
            "notes": "Extensive grid: N >> max(r,d) for all configs"
        }, f, indent=2)  # we save #
    
    print("\n" + "=" * 80)
    print(f"EXTENSIVE GRID COMPLETE")
    print(f"  Total attempted: {total_attempted}")
    print(f"  Total computed: {total_computed}")
    print(f"  Skipped (regime): {total_skipped_regime}")
    print(f"  Master index: {master_path}")
    print("=" * 80)


if __name__ == "__main__":
    # EXTENSIVE configuration
    WIDTH_POWERS = list(range(11, 18))  # N ∈ {2048, 4096, 8192, 16384, 32768, 65536, 131072}
    N_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]  # many n values
    R_VALUES = [16, 32, 64, 128, 256, 512, 1024]  # many r values
    D_VALUES = [16, 32, 64, 128, 256, 512, 1024]  # many d values
    N_INIT = 3  # initializations per config
    
    print("=" * 80)
    print("STARTING EXTENSIVE LARGE-WIDTH COMPUTATION")
    print("=" * 80)
    print(f"Grid size: {len(WIDTH_POWERS)} × {len(N_VALUES)} × {len(R_VALUES)} × {len(D_VALUES)}")
    print(f"Maximum configs: {len(WIDTH_POWERS) * len(N_VALUES) * len(R_VALUES) * len(D_VALUES)}")
    print(f"Constraint: N >= 8 * max(r, d) reduces this significantly")
    print(f"Estimated kept configs: ~500-1000 (after regime filtering)")
    print(f"Estimated time: ~10-50 hours depending on hardware (parallelizable!)")
    print("=" * 80)
    print()
    
    # run extensive grid
    run_extensive_grid(
        width_powers=WIDTH_POWERS,
        n_values=N_VALUES,
        r_values=R_VALUES,
        d_values=D_VALUES,
        n_init=N_INIT,
        do_ntk_rho=False  # disable ntk_rho for speed (only spectra)
    )  # we run computation #
    
    print("\n" + "=" * 80)
    print("EXTENSIVE COMPUTATION COMPLETE!")
    print("=" * 80)

