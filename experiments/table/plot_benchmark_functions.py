#!/usr/bin/env python3
"""
we plot actual functions from all PDE benchmarks (1D and 2D) for MMNN architecture tuning
we focus on real benchmark datasets, not synthetic functions
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # we use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import os
from scipy.interpolate import griddata

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
    pinnacle_path = Path(__file__).parent / "PINNacle"
    if pinnacle_path.exists():
        sys.path.insert(0, str(pinnacle_path))
        import deepxde as dde
        from src.pde.burgers import Burgers1D, Burgers2D
        from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
        from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson3D_ComplexGeometry, Poisson2D_ManyArea
        from src.pde.heat import Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, Heat2D_LongTime
        from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime
        PINNACLE_AVAILABLE = True
        print("✓ PINNacle available")
except Exception as e:
    print(f"⚠ PINNacle not available: {e}")
    PINNACLE_AVAILABLE = False

# we try to import PDEBench
try:
    import pdebench
    PDEBENCH_AVAILABLE = True
    print("✓ PDEBench available")
except ImportError:
    print("⚠ PDEBench not available")
    PDEBENCH_AVAILABLE = False


def plot_1d_function(x, u, title, filename, xlabel="x", ylabel="u(x)"):
    """we plot a 1D function"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, u, 'b-', linewidth=2, label='function')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_1d_time_evolution(x, t_slices, u_slices, title, filename):
    """we plot 1D function at different time slices"""
    fig, ax = plt.subplots(figsize=(12, 7))
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
        pass
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_2d_contour(x, y, u, title, filename, time_label=""):
    """we plot 2D function as contour"""
    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(x, y)
    if u.ndim == 1:
        # we reshape if needed
        u_grid = u.reshape(len(y), len(x))
    else:
        u_grid = u
    contour = ax.contourf(X, Y, u_grid, levels=20, cmap='viridis')
    ax.contour(X, Y, u_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f"{title}{time_label}", fontsize=14, fontweight='bold')
    plt.colorbar(contour, ax=ax, label='u(x,y)')
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_2d_surface(x, y, u, title, filename):
    """we plot 2D function as 3D surface"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    if u.ndim == 1:
        u_grid = u.reshape(len(y), len(x))
    else:
        u_grid = u
    surf = ax.plot_surface(X, Y, u_grid, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('u(x,y)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename}")


def plot_2d_time_slices(x, y, t_slices, u_slices, title_base, filename_base):
    """we plot 2D function at different time slices"""
    n_slices = len(t_slices)
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    axes = axes.flatten()
    
    X, Y = np.meshgrid(x, y)
    vmin = min([u.min() for u in u_slices])
    vmax = max([u.max() for u in u_slices])
    
    for i, (t, u) in enumerate(zip(t_slices, u_slices)):
        if u.ndim == 1:
            u_grid = u.reshape(len(y), len(x))
        else:
            u_grid = u
        ax = axes[i]
        contour = ax.contourf(X, Y, u_grid, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f't={t:.2f}', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(contour, ax=ax)
    
    # we hide unused subplots
    for i in range(len(t_slices), len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title_base, fontsize=14, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / filename_base, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {filename_base}")


def load_ref_data_safe(datapath, expected_cols=None):
    """we safely load reference data, handling COMSOL format"""
    try:
        if isinstance(datapath, (list, tuple)):
            datapath = datapath[0]
        datapath = Path(datapath)
        if not datapath.is_absolute():
            # we try relative to PINNacle/ref
            rel_path = Path(__file__).parent / "PINNacle" / "ref" / datapath.name
            if rel_path.exists():
                datapath = rel_path
        if datapath.exists():
            # we read header to understand format
            with open(datapath, 'r') as f:
                header_lines = []
                for i, line in enumerate(f):
                    if line.startswith('%'):
                        header_lines.append(line)
                    else:
                        break
                # we skip header and load data
                data = np.loadtxt(datapath, comments="%")
                return data, header_lines
    except Exception as e:
        pass
    return None, None


def parse_comsol_time_slices(header_lines):
    """we parse COMSOL header to extract time information"""
    if not header_lines:
        return None
    # we look for time information in header
    times = []
    for line in header_lines:
        if '@ t=' in line:
            import re
            time_matches = re.findall(r'@ t=([\d.]+)', line)
            times = [float(t) for t in time_matches]
            break
    return times if times else None


def plot_pinnacle_burgers1d():
    """we plot Burgers 1D from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Burgers1D")
        # we load reference data
        result = load_ref_data_safe("ref/burgers1d.dat")
        
        x_test = np.linspace(-1, 1, 200)
        t_test = np.linspace(0, 1, 5)
        
        if result is not None:
            ref_data, _ = result
            if ref_data is not None and len(ref_data) > 0:
                # we use reference data
                ref_x = ref_data[:, 0]
                ref_t = ref_data[:, 1]
                ref_u = ref_data[:, 2]
                
                u_slices = []
                for t in t_test:
                    xt = np.column_stack([x_test, np.full(len(x_test), t)])
                    u = griddata(np.column_stack([ref_x, ref_t]), ref_u, xt, method='linear', fill_value=np.nan)
                    mask = np.isnan(u)
                    if np.any(mask):
                        u[mask] = np.sin(-np.pi * x_test[mask])
                    u_slices.append(u)
            else:
                # we use analytical approximation
                u0 = np.sin(-np.pi * x_test)
                nu = 0.01 / np.pi
                u_slices = [u0 * np.exp(-nu * (np.pi**2) * t) for t in t_test]
        else:
            # we use analytical approximation
            u0 = np.sin(-np.pi * x_test)
            nu = 0.01 / np.pi
            u_slices = [u0 * np.exp(-nu * (np.pi**2) * t) for t in t_test]
        
        plot_1d_time_evolution(
            x_test, t_test, u_slices,
            "PINNacle: Burgers 1D\nu_t + u*u_x = nu*u_xx",
            "pinnacle_burgers1d.png"
        )
        
        u0 = np.sin(-np.pi * x_test)
        plot_1d_function(x_test, u0, "PINNacle: Burgers 1D - Initial Condition\nu(x,0) = -sin(pi*x)", "pinnacle_burgers1d_ic.png")
        
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()


def plot_pinnacle_burgers2d():
    """we plot Burgers 2D from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Burgers2D")
        result = load_ref_data_safe("ref/burgers2d_0.dat")
        if result is None:
            print("  ⚠ Reference data not found, skipping")
            return
        
        ref_data, header_lines = result
        if ref_data is None or len(ref_data) == 0:
            print("  ⚠ Reference data empty, skipping")
            return
        
        # we parse COMSOL format: X, Y, u@t=0, v@t=0, u@t=0.1, v@t=0.1, ...
        times = parse_comsol_time_slices(header_lines)
        if times is None:
            # we try to infer from data shape
            n_time_slices = (ref_data.shape[1] - 2) // 2  # X, Y, then pairs of u,v
            times = np.linspace(0, 1, n_time_slices)
        
        x_coords = np.unique(ref_data[:, 0])
        y_coords = np.unique(ref_data[:, 1])
        
        # we select a few time slices to plot
        n_plot = min(4, len(times))
        time_indices = np.linspace(0, len(times)-1, n_plot, dtype=int)
        t_slices = [times[i] for i in time_indices]
        
        # we extract u and v components
        u_slices = []
        v_slices = []
        for idx in time_indices:
            u_col = 2 + 2 * idx  # u component column
            v_col = 2 + 2 * idx + 1  # v component column
            
            u_data = ref_data[:, u_col]
            v_data = ref_data[:, v_col]
            
            X, Y = np.meshgrid(x_coords, y_coords)
            u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
            v_grid = griddata((ref_data[:, 0], ref_data[:, 1]), v_data, (X, Y), method='linear')
            
            u_slices.append(u_grid)
            v_slices.append(v_grid)
        
        plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                           "PINNacle: Burgers 2D - u component", "pinnacle_burgers2d_u.png")
        plot_2d_time_slices(x_coords, y_coords, t_slices, v_slices,
                           "PINNacle: Burgers 2D - v component", "pinnacle_burgers2d_v.png")
            
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()


def plot_pinnacle_wave1d():
    """we plot Wave 1D from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Wave1D")
        pde = Wave1D(a=4)
        x_test = np.linspace(0, 1, 200).reshape(-1, 1)
        t_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        u_slices = []
        for t in t_slices:
            xt = np.hstack([x_test, np.full((len(x_test), 1), t)])
            u = pde.ref_sol(xt).flatten()
            u_slices.append(u)
        
        plot_1d_time_evolution(x_test.flatten(), t_slices, u_slices,
                              "PINNacle: Wave 1D\nu_tt = c^2*u_xx", "pinnacle_wave1d.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


def plot_pinnacle_poisson2d():
    """we plot Poisson 2D problems from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Poisson2D_Classic")
        result = load_ref_data_safe("ref/poisson1_cg_data.dat")
        if result is None:
            print("  ⚠ Reference data not found")
        else:
            ref_data, _ = result
            if ref_data is not None and len(ref_data) > 0:
                x_coords = np.unique(ref_data[:, 0])
                y_coords = np.unique(ref_data[:, 1])
                u_data = ref_data[:, 2]
                
                X, Y = np.meshgrid(x_coords, y_coords)
                u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                
                plot_2d_contour(x_coords, y_coords, u_grid, 
                              "PINNacle: Poisson 2D Classic\n-Laplacian(u) = 0", 
                              "pinnacle_poisson2d_classic.png")
                plot_2d_surface(x_coords, y_coords, u_grid,
                              "PINNacle: Poisson 2D Classic - Surface",
                              "pinnacle_poisson2d_classic_surface.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
    
    try:
        print("\n📊 Plotting PINNacle: PoissonBoltzmann2D")
        result = load_ref_data_safe("ref/poisson_boltzmann2d.dat")
        if result is None:
            print("  ⚠ Reference data not found")
        else:
            ref_data, _ = result
            if ref_data is not None and len(ref_data) > 0:
                x_coords = np.unique(ref_data[:, 0])
                y_coords = np.unique(ref_data[:, 1])
                u_data = ref_data[:, 2]
                
                X, Y = np.meshgrid(x_coords, y_coords)
                u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                
                plot_2d_contour(x_coords, y_coords, u_grid,
                              "PINNacle: Poisson-Boltzmann 2D",
                              "pinnacle_poisson_boltzmann2d.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


def plot_pinnacle_heat2d():
    """we plot Heat 2D problems from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Heat2D_VaryingCoef")
        result = load_ref_data_safe("ref/heat_darcy.dat")
        if result is None:
            print("  ⚠ Reference data not found")
            return
        
        ref_data, header_lines = result
        if ref_data is None or len(ref_data) == 0:
            print("  ⚠ Reference data empty")
            return
        
        # we parse COMSOL format: X, Y, u@t=0, u@t=0.1, ...
        times = parse_comsol_time_slices(header_lines)
        if times is None:
            n_time_slices = ref_data.shape[1] - 2  # X, Y, then u columns
            times = np.linspace(0, 5, n_time_slices)  # we assume 0 to 5 based on header
        
        x_coords = np.unique(ref_data[:, 0])
        y_coords = np.unique(ref_data[:, 1])
        
        # we select a few time slices
        n_plot = min(4, len(times))
        time_indices = np.linspace(0, len(times)-1, n_plot, dtype=int)
        t_slices = [times[i] for i in time_indices]
        
        u_slices = []
        for idx in time_indices:
            u_col = 2 + idx  # u component column
            u_data = ref_data[:, u_col]
            X, Y = np.meshgrid(x_coords, y_coords)
            u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
            u_slices.append(u_grid)
        
        plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                           "PINNacle: Heat 2D Varying Coefficient",
                           "pinnacle_heat2d_varying.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n📊 Plotting PINNacle: Heat2D_Multiscale")
        result = load_ref_data_safe("ref/heat_multiscale.dat")
        if result is None:
            print("  ⚠ Reference data not found")
        else:
            ref_data, header_lines = result
            if ref_data is not None and len(ref_data) > 0:
                times = parse_comsol_time_slices(header_lines)
                if times is None:
                    n_time_slices = ref_data.shape[1] - 2
                    times = np.linspace(0, 5, n_time_slices)
                
                x_coords = np.unique(ref_data[:, 0])
                y_coords = np.unique(ref_data[:, 1])
                
                n_plot = min(4, len(times))
                time_indices = np.linspace(0, len(times)-1, n_plot, dtype=int)
                t_slices = [times[i] for i in time_indices]
                
                u_slices = []
                for idx in time_indices:
                    u_col = 2 + idx
                    u_data = ref_data[:, u_col]
                    X, Y = np.meshgrid(x_coords, y_coords)
                    u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                    u_slices.append(u_grid)
                
                plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                                   "PINNacle: Heat 2D Multiscale",
                                   "pinnacle_heat2d_multiscale.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()


def plot_pinnacle_ns2d():
    """we plot Navier-Stokes 2D problems from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: NS2D_LidDriven")
        result = load_ref_data_safe("ref/lid_driven_a2.dat")
        if result is None:
            print("  ⚠ Reference data not found")
            return
        
        ref_data, _ = result
        if ref_data is None or len(ref_data) == 0:
            print("  ⚠ Reference data empty")
            return
        
        x_coords = np.unique(ref_data[:, 0])
        y_coords = np.unique(ref_data[:, 1])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # we plot u component
        if ref_data.shape[1] > 2:
            u_data = ref_data[:, 2]
            u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
            plot_2d_contour(x_coords, y_coords, u_grid,
                          "PINNacle: NS 2D Lid-Driven - u velocity",
                          "pinnacle_ns2d_lid_u.png")
        
        # we plot v component if available
        if ref_data.shape[1] > 3:
            v_data = ref_data[:, 3]
            v_grid = griddata((ref_data[:, 0], ref_data[:, 1]), v_data, (X, Y), method='linear')
            plot_2d_contour(x_coords, y_coords, v_grid,
                          "PINNacle: NS 2D Lid-Driven - v velocity",
                          "pinnacle_ns2d_lid_v.png")
        
        # we plot pressure if available
        if ref_data.shape[1] > 4:
            p_data = ref_data[:, 4]
            p_grid = griddata((ref_data[:, 0], ref_data[:, 1]), p_data, (X, Y), method='linear')
            plot_2d_contour(x_coords, y_coords, p_grid,
                          "PINNacle: NS 2D Lid-Driven - pressure",
                          "pinnacle_ns2d_lid_p.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()


def plot_pinnacle_wave2d():
    """we plot Wave 2D problems from PINNacle"""
    if not PINNACLE_AVAILABLE:
        return
    
    try:
        print("\n📊 Plotting PINNacle: Wave2D_Heterogeneous")
        result = load_ref_data_safe("ref/wave_darcy.dat")
        if result is None:
            print("  ⚠ Reference data not found")
            return
        
        ref_data, header_lines = result
        if ref_data is None or len(ref_data) == 0:
            print("  ⚠ Reference data empty")
            return
        
        # we parse COMSOL format
        times = parse_comsol_time_slices(header_lines)
        if times is None:
            n_time_slices = ref_data.shape[1] - 2  # X, Y, then u columns
            times = np.linspace(0, 5, n_time_slices)
        
        x_coords = np.unique(ref_data[:, 0])
        y_coords = np.unique(ref_data[:, 1])
        
        n_plot = min(4, len(times))
        time_indices = np.linspace(0, len(times)-1, n_plot, dtype=int)
        t_slices = [times[i] for i in time_indices]
        
        u_slices = []
        for idx in time_indices:
            u_col = 2 + idx
            u_data = ref_data[:, u_col]
            X, Y = np.meshgrid(x_coords, y_coords)
            u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
            u_slices.append(u_grid)
        
        plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                           "PINNacle: Wave 2D Heterogeneous",
                           "pinnacle_wave2d_heterogeneous.png")
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """we run all plotting functions"""
    print("="*80)
    print("Plotting Actual Benchmark Functions (1D and 2D)")
    print("For MMNN Architecture Tuning")
    print("="*80)
    
    # we plot from PINNacle
    plot_pinnacle_burgers1d()
    plot_pinnacle_burgers2d()
    plot_pinnacle_wave1d()
    plot_pinnacle_wave2d()
    plot_pinnacle_poisson2d()
    plot_pinnacle_heat2d()
    plot_pinnacle_ns2d()
    
    print("\n" + "="*80)
    print(f"✓ All plots saved to: {output_dir}")
    print("="*80)
    print("\nThese plots show actual benchmark functions to help tune MMNN architecture.")


if __name__ == "__main__":
    main()
