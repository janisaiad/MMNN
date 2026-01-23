#!/usr/bin/env python3
"""
we plot 1D functions from all PDE benchmarks for MLP/MMNN comparison
we focus on datasets suitable for fully connected networks (not transformers)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # we use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# we add paths for benchmark imports
sys.path.insert(0, str(Path(__file__).parent))

# we create output directory
output_dir = Path(__file__).parent / "plots_to_fit"
output_dir.mkdir(exist_ok=True)

print(f"we will save plots to {output_dir}")

# we try to import PINNacle problems
PINNACLE_AVAILABLE = False
try:
    os.environ["DDEBACKEND"] = "pytorch"
    # we add PINNacle to path
    pinnacle_path = Path(__file__).parent / "PINNacle"
    if pinnacle_path.exists():
        sys.path.insert(0, str(pinnacle_path))
        import deepxde as dde
        from src.pde.burgers import Burgers1D
        from src.pde.wave import Wave1D
        PINNACLE_AVAILABLE = True
        print("✓ PINNacle available")
except Exception as e:
    print(f"⚠ PINNacle not available: {e}")
    PINNACLE_AVAILABLE = False

# we try to import PDEBench (if available)
try:
    import pdebench
    PDEBENCH_AVAILABLE = True
    print("✓ PDEBench available")
except ImportError:
    print("⚠ PDEBench not available")
    PDEBENCH_AVAILABLE = False

# we note: PDEArena and Poseidon use convolutional/transformer architectures
# we skip them as they are not suitable for MLP/MMNN comparison
print("ℹ Note: PDEArena and Poseidon use convolutional/transformer architectures")
print("  we skip them as they are not suitable for MLP/MMNN comparison")


def plot_1d_slice(x, u, title, filename, xlabel="x", ylabel="u(x)"):
    """we plot a 1D function slice"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, u, 'b-', linewidth=2, label='function')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    try:
        plt.tight_layout()
    except:
        pass  # we skip tight_layout if it fails
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_1d_time_evolution(x, t_slices, u_slices, title, filename):
    """we plot 1D function at different time slices"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_slices)))
    for i, (t, u) in enumerate(zip(t_slices, u_slices)):
        ax.plot(x, u, linewidth=2, color=colors[i], label=f't={t:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x,t)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2)
    try:
        plt.tight_layout()
    except:
        pass  # we skip tight_layout if it fails
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_pinnacle_burgers1d():
    """we plot Burgers 1D from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Burgers1D")
        # we create test points
        x_test = np.linspace(-1, 1, 200)
        t_test = np.linspace(0, 1, 5)
        
        # we plot initial condition and analytical approximation
        # initial condition: u(x,0) = -sin(pi*x)
        u0 = np.sin(-np.pi * x_test)
        
        # we create time evolution using a simple analytical approximation
        # for Burgers equation with small viscosity
        nu = 0.01 / np.pi
        u_slices = []
        for t in t_test:
            # we use a simple approximation: initial condition with diffusion
            # this is not the exact solution but shows the general behavior
            u = u0 * np.exp(-nu * (np.pi**2) * t)
            # we add a small shift to simulate the nonlinear advection effect
            # (this is a simplified approximation)
            u_slices.append(u)
        
        plot_1d_time_evolution(
            x_test, t_test, u_slices,
            "PINNacle: Burgers 1D Equation\nu_t + u*u_x = nu*u_xx\n(analytical approximation)",
            "pinnacle_burgers1d.png"
        )
        
        # we also plot initial condition
        plot_1d_slice(
            x_test, u0,
            "PINNacle: Burgers 1D - Initial Condition\nu(x,0) = -sin(pi*x)",
            "pinnacle_burgers1d_ic.png"
        )
        
    except Exception as e:
        print(f"  ⚠ Error plotting Burgers1D: {e}")
        # we still plot the initial condition even if time evolution fails
        try:
            x_test = np.linspace(-1, 1, 200)
            u0 = np.sin(-np.pi * x_test)
            plot_1d_slice(
                x_test, u0,
                "PINNacle: Burgers 1D - Initial Condition\nu(x,0) = -sin(pi*x)",
                "pinnacle_burgers1d_ic.png"
            )
        except:
            pass


def plot_pinnacle_wave1d():
    """we plot Wave 1D from PINNacle (actually 2D: x-t)"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Wave1D (x-t slice)")
        pde = Wave1D(a=4)
        
        # we create test points
        x_test = np.linspace(0, 1, 200).reshape(-1, 1)
        t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # we use reference solution
        u_slices = []
        for t in t_slices:
            xt = np.hstack([x_test, np.full((len(x_test), 1), t)])
            u = pde.ref_sol(xt).flatten()
            u_slices.append(u)
        
        plot_1d_time_evolution(
            x_test.flatten(), t_slices, u_slices,
            "PINNacle: Wave 1D Equation\nu_tt = c^2*u_xx",
            "pinnacle_wave1d.png"
        )
        
    except Exception as e:
        print(f"  ⚠ Error plotting Wave1D: {e}")


def plot_synthetic_1d_functions():
    """we plot synthetic 1D functions for comparison"""
    print("\n📊 Plotting synthetic 1D functions")
    
    x = np.linspace(-2, 2, 500)
    
    # we plot various function types
    functions = [
        ("sine", np.sin(2 * np.pi * x), "Synthetic: Sine Wave\nf(x) = sin(2*pi*x)"),
        ("gaussian", np.exp(-x**2), "Synthetic: Gaussian\nf(x) = exp(-x^2)"),
        ("polynomial", x**3 - 3*x, "Synthetic: Cubic Polynomial\nf(x) = x^3 - 3x"),
        ("sinc", np.sinc(x), "Synthetic: Sinc Function\nf(x) = sinc(x)"),
        ("tanh", np.tanh(2*x), "Synthetic: Hyperbolic Tangent\nf(x) = tanh(2x)"),
        ("ripple", np.sin(5*x) * np.exp(-x**2/2), "Synthetic: Damped Oscillation\nf(x) = sin(5x)*exp(-x^2/2)"),
    ]
    
    for name, y, title in functions:
        plot_1d_slice(x, y, title, f"synthetic_{name}.png")


def plot_pdebench_1d():
    """we plot 1D functions from PDEBench if available"""
    if not PDEBENCH_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PDEBench 1D datasets")
        # we note: PDEBench typically has 2D/3D datasets
        # we look for 1D time series or create slices
        print("  ℹ PDEBench datasets are typically 2D/3D")
        print("  ℹ Consider extracting 1D slices from 2D data for MLP/MMNN comparison")
        
    except Exception as e:
        print(f"  ⚠ Error with PDEBench: {e}")


def create_summary_plot():
    """we create a summary plot showing all 1D functions"""
    print("\n📊 Creating summary plot")
    
    # we collect all generated plots
    plot_files = sorted(output_dir.glob("*.png"))
    
    if len(plot_files) == 0:
        print("  ⚠ No plots found to summarize")
        return
    
    # we create a grid of subplots
    n_plots = len(plot_files)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, plot_file in enumerate(plot_files):
        if i >= len(axes):
            break
        try:
            img = plt.imread(plot_file)
            axes[i].imshow(img)
            axes[i].set_title(plot_file.stem, fontsize=10)
            axes[i].axis('off')
        except Exception as e:
            print(f"  ⚠ Could not load {plot_file}: {e}")
    
    # we hide unused subplots
    for i in range(len(plot_files), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("1D Functions for MLP/MMNN Comparison", fontsize=16, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass  # we skip tight_layout if it fails
    plt.savefig(output_dir / "summary_all_1d_functions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: summary_all_1d_functions.png")


def main():
    """we run all plotting functions"""
    print("="*80)
    print("Plotting 1D Functions from PDE Benchmarks")
    print("Focus: Datasets suitable for MLP/MMNN comparison")
    print("="*80)
    
    # we plot from each benchmark
    plot_synthetic_1d_functions()
    plot_pinnacle_burgers1d()
    plot_pinnacle_wave1d()
    plot_pdebench_1d()
    
    # we create summary
    create_summary_plot()
    
    print("\n" + "="*80)
    print(f"✓ All plots saved to: {output_dir}")
    print("="*80)
    print("\nNote: These 1D functions are suitable for MLP/MMNN comparison.")
    print("PDEArena and Poseidon use convolutional/transformer architectures")
    print("and are not included as they are not suitable for FCN comparison.")


if __name__ == "__main__":
    main()
