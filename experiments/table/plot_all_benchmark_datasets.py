#!/usr/bin/env python3
"""
we plot ALL datasets from ALL benchmarks systematically
this covers every available problem across all benchmarks for MMNN architecture tuning
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
import os
from scipy.interpolate import griddata
import re
from collections import defaultdict

# we add paths
sys.path.insert(0, str(Path(__file__).parent))

# we create output directory
output_dir = Path(__file__).parent / "plots_to_fit"
output_dir.mkdir(exist_ok=True)

# we create subdirectories for organization
(output_dir / "1d").mkdir(exist_ok=True)
(output_dir / "2d").mkdir(exist_ok=True)
(output_dir / "3d").mkdir(exist_ok=True)
(output_dir / "nd").mkdir(exist_ok=True)
(output_dir / "time_series").mkdir(exist_ok=True)

print(f"we will save plots to {output_dir}")

# we try to import all PINNacle problems
PINNACLE_AVAILABLE = False
PINNACLE_PROBLEMS = {}
try:
    os.environ["DDEBACKEND"] = "pytorch"
    pinnacle_path = Path(__file__).parent / "PINNacle"
    if pinnacle_path.exists():
        sys.path.insert(0, str(pinnacle_path))
        import deepxde as dde
        
        # we import all problem classes
        from src.pde.burgers import Burgers1D, Burgers2D
        from src.pde.wave import Wave1D, Wave2D_Heterogeneous, Wave2D_LongTime
        from src.pde.poisson import (Poisson1D, Poisson2D_Classic, PoissonBoltzmann2D, 
                                    Poisson3D_ComplexGeometry, Poisson2D_ManyArea, PoissonND)
        from src.pde.heat import (Heat2D_VaryingCoef, Heat2D_Multiscale, Heat2D_ComplexGeometry, 
                                 Heat2D_LongTime, HeatND)
        from src.pde.ns import NS2D_LidDriven, NS2D_BackStep, NS2D_LongTime, NS2D_Classic
        from src.pde.chaotic import GrayScottEquation, KuramotoSivashinskyEquation
        from src.pde.inverse import PoissonInv, HeatInv
        from src.pde.helmholtz import Helmholtz2D
        
        # we organize all problems
        PINNACLE_PROBLEMS = {
            'burgers': {
                'Burgers1D': Burgers1D,
                'Burgers2D': Burgers2D,
            },
            'wave': {
                'Wave1D': Wave1D,
                'Wave2D_Heterogeneous': Wave2D_Heterogeneous,
                'Wave2D_LongTime': Wave2D_LongTime,
            },
            'poisson': {
                'Poisson1D': Poisson1D,
                'Poisson2D_Classic': Poisson2D_Classic,
                'PoissonBoltzmann2D': PoissonBoltzmann2D,
                'Poisson3D_ComplexGeometry': Poisson3D_ComplexGeometry,
                'Poisson2D_ManyArea': Poisson2D_ManyArea,
                'PoissonND': PoissonND,
            },
            'heat': {
                'Heat2D_VaryingCoef': Heat2D_VaryingCoef,
                'Heat2D_Multiscale': Heat2D_Multiscale,
                'Heat2D_ComplexGeometry': Heat2D_ComplexGeometry,
                'Heat2D_LongTime': Heat2D_LongTime,
                'HeatND': HeatND,
            },
            'ns': {
                'NS2D_LidDriven': NS2D_LidDriven,
                'NS2D_BackStep': NS2D_BackStep,
                'NS2D_LongTime': NS2D_LongTime,
                'NS2D_Classic': NS2D_Classic,
            },
            'chaotic': {
                'GrayScottEquation': GrayScottEquation,
                'KuramotoSivashinskyEquation': KuramotoSivashinskyEquation,
            },
            'inverse': {
                'PoissonInv': PoissonInv,
                'HeatInv': HeatInv,
            },
            'other': {
                'Helmholtz2D': Helmholtz2D,
            }
        }
        
        PINNACLE_AVAILABLE = True
        print("✓ PINNacle available")
        print(f"  Found {sum(len(v) for v in PINNACLE_PROBLEMS.values())} problem classes")
except Exception as e:
    print(f"⚠ PINNacle not available: {e}")
    PINNACLE_AVAILABLE = False

# we also check for additional data files
def find_all_ref_files():
    """we find all reference data files"""
    ref_dir = Path(__file__).parent / "PINNacle" / "ref"
    if not ref_dir.exists():
        return []
    return list(ref_dir.glob("*.dat"))


def load_ref_data_safe(datapath, expected_cols=None):
    """we safely load reference data, handling COMSOL format"""
    try:
        if isinstance(datapath, (list, tuple)):
            datapath = datapath[0]
        datapath = Path(datapath)
        if not datapath.is_absolute():
            rel_path = Path(__file__).parent / "PINNacle" / "ref" / datapath.name
            if rel_path.exists():
                datapath = rel_path
        if datapath.exists():
            with open(datapath, 'r') as f:
                header_lines = []
                for i, line in enumerate(f):
                    if line.startswith('%'):
                        header_lines.append(line)
                    else:
                        break
                data = np.loadtxt(datapath, comments="%")
                return data, header_lines
    except Exception as e:
        pass
    return None, None


def parse_comsol_time_slices(header_lines):
    """we parse COMSOL header to extract time information"""
    if not header_lines:
        return None
    times = []
    for line in header_lines:
        if '@ t=' in line:
            time_matches = re.findall(r'@ t=([\d.]+)', line)
            times = [float(t) for t in time_matches]
            break
    return times if times else None


def plot_1d_function(x, u, title, filename, subdir="1d"):
    """we plot a 1D function"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, u, 'b-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    try:
        plt.tight_layout()
    except:
        pass
    filepath = output_dir / subdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved: {subdir}/{filename}")


def plot_1d_time_evolution(x, t_slices, u_slices, title, filename, subdir="1d"):
    """we plot 1D function at different time slices"""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_slices)))
    for i, (t, u) in enumerate(zip(t_slices, u_slices)):
        ax.plot(x, u, linewidth=2, color=colors[i], label=f't={t:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x,t)', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, fontsize=9)
    try:
        plt.tight_layout()
    except:
        pass
    filepath = output_dir / subdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved: {subdir}/{filename}")


def plot_2d_contour(x, y, u, title, filename, subdir="2d", time_label=""):
    """we plot 2D function as contour"""
    fig, ax = plt.subplots(figsize=(10, 8))
    X, Y = np.meshgrid(x, y)
    if u.ndim == 1:
        u_grid = u.reshape(len(y), len(x))
    else:
        u_grid = u
    contour = ax.contourf(X, Y, u_grid, levels=20, cmap='viridis')
    ax.contour(X, Y, u_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f"{title}{time_label}", fontsize=12, fontweight='bold')
    plt.colorbar(contour, ax=ax, label='u(x,y)')
    try:
        plt.tight_layout()
    except:
        pass
    filepath = output_dir / subdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved: {subdir}/{filename}")


def plot_2d_time_slices(x, y, t_slices, u_slices, title, filename, subdir="2d"):
    """we plot 2D function at different time slices"""
    n_slices = len(t_slices)
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    axes = axes.flatten()
    
    X, Y = np.meshgrid(x, y)
    vmin = min([u.min() for u in u_slices if u is not None and not np.isnan(u).all()])
    vmax = max([u.max() for u in u_slices if u is not None and not np.isnan(u).all()])
    
    for i, (t, u) in enumerate(zip(t_slices, u_slices)):
        if u is None or np.isnan(u).all():
            continue
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
    
    for i in range(len(t_slices), len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    try:
        plt.tight_layout()
    except:
        pass
    filepath = output_dir / subdir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    saved: {subdir}/{filename}")


def plot_problem(problem_name, problem_class, problem_kwargs=None, category="unknown"):
    """we plot a single problem"""
    if problem_kwargs is None:
        problem_kwargs = {}
    
    # we first try to find and load data files directly
    ref_data = None
    header_lines = None
    
    # we try to find matching data files
    ref_dir = Path(__file__).parent / "PINNacle" / "ref"
    if ref_dir.exists():
        # we look for files matching the problem name
        pattern = problem_name.lower().replace('_', '').replace('2d', '2d').replace('1d', '1d')
        matching_files = [f for f in ref_dir.glob("*.dat") if pattern in f.name.lower() or 
                         problem_name.lower().replace('_', '') in f.name.lower()]
        
        # we also check for common patterns
        if not matching_files:
            # we try common naming patterns
            if 'burgers1d' in problem_name.lower():
                matching_files = [ref_dir / "burgers1d.dat"]
            elif 'burgers2d' in problem_name.lower():
                if '_v' in problem_name or 'v' in problem_name:
                    # we extract variant number
                    match = re.search(r'v?(\d+)', problem_name)
                    if match:
                        variant = match.group(1)
                        matching_files = [ref_dir / f"burgers2d_{variant}.dat"]
                    else:
                        matching_files = [ref_dir / "burgers2d_0.dat"]
                else:
                    matching_files = [ref_dir / "burgers2d_0.dat"]
            elif 'lid' in problem_name.lower() or 'ns2d' in problem_name.lower():
                match = re.search(r'a(\d+)', problem_name)
                if match:
                    a_val = match.group(1)
                    matching_files = [ref_dir / f"lid_driven_a{a_val}.dat"]
                else:
                    matching_files = [ref_dir / "lid_driven_a2.dat"]
            elif 'poisson' in problem_name.lower():
                if 'classic' in problem_name.lower() or '2d_classic' in problem_name.lower():
                    matching_files = [ref_dir / "poisson1_cg_data.dat", ref_dir / "poisson_classic.dat"]
                elif 'boltzmann' in problem_name.lower():
                    matching_files = [ref_dir / "poisson_boltzmann2d.dat"]
                elif 'many' in problem_name.lower():
                    matching_files = [ref_dir / "poisson_manyarea.dat"]
                elif '3d' in problem_name.lower():
                    matching_files = [ref_dir / "poisson_3d.dat"]
            elif 'heat' in problem_name.lower():
                if 'varying' in problem_name.lower() or 'darcy' in problem_name.lower():
                    matching_files = [ref_dir / "heat_darcy.dat"]
                elif 'multiscale' in problem_name.lower():
                    matching_files = [ref_dir / "heat_multiscale.dat"]
                elif 'complex' in problem_name.lower():
                    matching_files = [ref_dir / "heat_complex.dat"]
                elif 'long' in problem_name.lower():
                    matching_files = [ref_dir / "heat_longtime.dat"]
            elif 'wave' in problem_name.lower():
                if 'heterogeneous' in problem_name.lower() or 'darcy' in problem_name.lower():
                    matching_files = [ref_dir / "wave_darcy.dat"]
            elif 'grayscott' in problem_name.lower():
                matching_files = [ref_dir / "grayscott.dat"]
            elif 'kuramoto' in problem_name.lower() or 'sivashinsky' in problem_name.lower():
                matching_files = [ref_dir / "Kuramoto_Sivashinsky.dat"]
            elif 'ns' in problem_name.lower() and 'back' in problem_name.lower():
                matching_files = [ref_dir / "ns_0_obstacle.dat", ref_dir / "ns_4_obstacle.dat"]
            elif 'ns' in problem_name.lower() and 'long' in problem_name.lower():
                matching_files = [ref_dir / "ns_long.dat"]
        
        # we try to load the first matching file
        for data_file in matching_files:
            if data_file.exists():
                result = load_ref_data_safe(data_file)
                if result[0] is not None:
                    ref_data, header_lines = result
                    break
    
    # we now try to instantiate the problem (with error handling)
    pde = None
    try:
        print(f"\n  📊 {problem_name}")
        # we modify kwargs to skip data loading if we already have data
        safe_kwargs = problem_kwargs.copy()
        # we try to instantiate
        pde = problem_class(**safe_kwargs)
        # we use pde's ref_data if available and we don't have our own
        if ref_data is None and hasattr(pde, 'ref_data') and pde.ref_data is not None:
            ref_data = pde.ref_data
    except FileNotFoundError as e:
        # we continue without the problem instance if data file is missing
        print(f"    ℹ Data file not found, using direct data loading: {e}")
    except Exception as e:
        print(f"    ⚠ Could not instantiate problem: {e}")
        # we continue with just the data if available
    
    try:
        
        # we determine dimension
        if pde is not None:
            input_dim = pde.input_dim if hasattr(pde, 'input_dim') else getattr(pde, 'input_dim', 2)
            output_dim = pde.output_dim if hasattr(pde, 'output_dim') else 1
            is_time_dependent = hasattr(pde, 'geomtime') or 'Time' in str(type(pde))
            bbox = pde.bbox if hasattr(pde, 'bbox') else None
        else:
            # we infer from data or problem name
            if ref_data is not None:
                input_dim = ref_data.shape[1] - 1  # we assume last column is output
                output_dim = 1
                is_time_dependent = input_dim >= 3  # x, y, t or more
            else:
                # we infer from problem name
                if '1d' in problem_name.lower():
                    input_dim = 1
                elif '2d' in problem_name.lower():
                    input_dim = 2
                elif '3d' in problem_name.lower():
                    input_dim = 3
                else:
                    input_dim = 2  # we default to 2D
                output_dim = 1
                is_time_dependent = 'Time' in problem_name or 'LongTime' in problem_name
            bbox = None
        
        # we plot based on dimension
        if input_dim == 1 and not is_time_dependent:
            # we plot 1D static
            if ref_data is not None and len(ref_data) > 0:
                x = ref_data[:, 0]
                u = ref_data[:, 1] if ref_data.shape[1] > 1 else ref_data[:, 0]
                plot_1d_function(x, u, f"{problem_name}", f"{problem_name.lower()}.png", "1d")
            elif pde is not None and hasattr(pde, 'ref_sol') and pde.ref_sol is not None and bbox is not None:
                x = np.linspace(bbox[0], bbox[1], 200)
                u = pde.ref_sol(x.reshape(-1, 1)).flatten()
                plot_1d_function(x, u, f"{problem_name}", f"{problem_name.lower()}.png", "1d")
        
        elif input_dim == 2 and not is_time_dependent:
            # we plot 2D static
            if ref_data is not None and len(ref_data) > 0:
                x_coords = np.unique(ref_data[:, 0])
                y_coords = np.unique(ref_data[:, 1])
                u_data = ref_data[:, 2] if ref_data.shape[1] > 2 else ref_data[:, 0]
                X, Y = np.meshgrid(x_coords, y_coords)
                try:
                    u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                    if u_grid is not None and not np.isnan(u_grid).all():
                        plot_2d_contour(x_coords, y_coords, u_grid, f"{problem_name}", 
                                      f"{problem_name.lower()}.png", "2d")
                except Exception as e:
                    print(f"      ⚠ Could not interpolate 2D data: {e}")
        
        elif input_dim == 1 and is_time_dependent:
            # we plot 1D time-dependent
            if ref_data is not None and len(ref_data) > 0:
                x_coords = np.unique(ref_data[:, 0])
                t_coords = np.unique(ref_data[:, 1])
                u_data = ref_data[:, 2] if ref_data.shape[1] > 2 else ref_data[:, 0]
                t_slices = t_coords[::max(1, len(t_coords)//4)][:4]
                u_slices = []
                for t in t_slices:
                    mask = np.isclose(ref_data[:, 1], t)
                    if np.any(mask):
                        u_t = u_data[mask]
                        x_t = ref_data[mask, 0]
                        u_interp = griddata(x_t, u_t, x_coords, method='linear', fill_value=0)
                        u_slices.append(u_interp)
                if u_slices:
                    plot_1d_time_evolution(x_coords, t_slices, u_slices, f"{problem_name}",
                                         f"{problem_name.lower()}.png", "time_series")
            elif pde is not None and hasattr(pde, 'ref_sol') and pde.ref_sol is not None and bbox is not None and len(bbox) >= 4:
                x = np.linspace(bbox[0], bbox[1], 200)
                t_slices = np.linspace(bbox[2], bbox[3], 5)
                u_slices = []
                for t in t_slices:
                    xt = np.column_stack([x, np.full(len(x), t)])
                    u = pde.ref_sol(xt).flatten()
                    u_slices.append(u)
                plot_1d_time_evolution(x, t_slices, u_slices, f"{problem_name}",
                                     f"{problem_name.lower()}.png", "time_series")
        
        elif input_dim == 2 and is_time_dependent:
            # we plot 2D time-dependent
            if ref_data is not None and len(ref_data) > 0:
                # we check COMSOL format
                times = parse_comsol_time_slices(header_lines) if header_lines else None
                if times is not None:
                    # we handle COMSOL format
                    x_coords = np.unique(ref_data[:, 0])
                    y_coords = np.unique(ref_data[:, 1])
                    n_plot = min(4, len(times))
                    time_indices = np.linspace(0, len(times)-1, n_plot, dtype=int)
                    t_slices = [times[i] for i in time_indices]
                    u_slices = []
                    for idx in time_indices:
                        u_col = 2 + idx
                        if u_col < ref_data.shape[1]:
                            u_data = ref_data[:, u_col]
                            X, Y = np.meshgrid(x_coords, y_coords)
                    try:
                        u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                        if u_grid is not None and not np.isnan(u_grid).all():
                            u_slices.append(u_grid)
                    except Exception:
                        pass
                if u_slices:
                    plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                                      f"{problem_name}", f"{problem_name.lower()}.png", "2d")
                else:
                    # we handle standard format
                    x_coords = np.unique(ref_data[:, 0])
                    y_coords = np.unique(ref_data[:, 1])
                    t_coords = np.unique(ref_data[:, 2])
                    u_data = ref_data[:, 3] if ref_data.shape[1] > 3 else ref_data[:, 0]
                    t_slices = t_coords[::max(1, len(t_coords)//4)][:4]
                    u_slices = []
                    for t in t_slices:
                        mask = np.isclose(ref_data[:, 2], t)
                        u_t = u_data[mask]
                        x_t = ref_data[mask, 0]
                        y_t = ref_data[mask, 1]
                        X, Y = np.meshgrid(x_coords, y_coords)
                        try:
                            u_grid = griddata((x_t, y_t), u_t, (X, Y), method='linear')
                            if u_grid is not None and not np.isnan(u_grid).all():
                                u_slices.append(u_grid)
                        except Exception:
                            pass
                    if u_slices:
                        plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                                          f"{problem_name}", f"{problem_name.lower()}.png", "2d")
        
        elif input_dim == 3:
            # we plot 3D (we show 2D slices)
            if ref_data is not None and len(ref_data) > 0:
                # we take a slice at z=0 or middle z
                z_coords = np.unique(ref_data[:, 2])
                z_slice = z_coords[len(z_coords)//2]
                mask = np.isclose(ref_data[:, 2], z_slice)
                slice_data = ref_data[mask]
                if len(slice_data) > 0:
                    x_coords = np.unique(slice_data[:, 0])
                    y_coords = np.unique(slice_data[:, 1])
                    u_data = slice_data[:, 3] if slice_data.shape[1] > 3 else slice_data[:, 0]
                    X, Y = np.meshgrid(x_coords, y_coords)
                    try:
                        u_grid = griddata((slice_data[:, 0], slice_data[:, 1]), u_data, (X, Y), method='linear')
                        if u_grid is not None and not np.isnan(u_grid).all():
                            plot_2d_contour(x_coords, y_coords, u_grid, f"{problem_name} (z={z_slice:.2f})",
                                          f"{problem_name.lower()}_slice.png", "3d")
                    except Exception as e:
                        print(f"      ⚠ Could not plot 3D slice: {e}")
        
        elif input_dim > 3:
            # we plot ND (we show 2D projection)
            print(f"    ⚠ High-dimensional ({input_dim}D), skipping detailed plot")
            plot_1d_function(np.array([0, 1]), np.array([0, 1]), 
                           f"{problem_name} ({input_dim}D problem)", 
                           f"{problem_name.lower()}_info.png", "nd")
        
        # we also handle multi-component outputs (e.g., NS with u, v, p)
        if output_dim > 1 and ref_data is not None:
            if input_dim == 2:
                x_coords = np.unique(ref_data[:, 0])
                y_coords = np.unique(ref_data[:, 1])
                X, Y = np.meshgrid(x_coords, y_coords)
                for comp_idx in range(min(output_dim, 3)):  # we plot up to 3 components
                    u_col = 2 + comp_idx
                    if u_col < ref_data.shape[1]:
                        u_data = ref_data[:, u_col]
                        try:
                            u_grid = griddata((ref_data[:, 0], ref_data[:, 1]), u_data, (X, Y), method='linear')
                            if u_grid is not None and not np.isnan(u_grid).all():
                                comp_name = ['u', 'v', 'p', 'w'][comp_idx] if comp_idx < 4 else f'comp{comp_idx}'
                                plot_2d_contour(x_coords, y_coords, u_grid,
                                              f"{problem_name} - {comp_name}",
                                              f"{problem_name.lower()}_{comp_name}.png", "2d")
                        except Exception:
                            pass
        
    except Exception as e:
        print(f"    ⚠ Error plotting {problem_name}: {e}")
        import traceback
        traceback.print_exc()


def plot_all_pinnacle_problems():
    """we plot all PINNacle problems"""
    if not PINNACLE_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print("Plotting ALL PINNacle Problems")
    print("="*80)
    
    total_problems = 0
    plotted_problems = 0
    
    # we plot all problems by category
    for category, problems in PINNACLE_PROBLEMS.items():
        print(f"\n📁 Category: {category.upper()}")
        for problem_name, problem_class in problems.items():
            total_problems += 1
            try:
                plot_problem(problem_name, problem_class, {}, category)
                plotted_problems += 1
            except Exception as e:
                print(f"    ⚠ Failed to plot {problem_name}: {e}")
        
        # we also try variants with different parameters
        if category == 'burgers' and 'Burgers2D' in problems:
            # we try different Burgers2D variants
            for i in range(1, 5):
                try:
                    result = load_ref_data_safe(f"ref/burgers2d_{i}.dat")
                    if result[0] is not None:
                        plot_problem(f"Burgers2D_v{i}", problems['Burgers2D'], 
                                   {"datapath": f"ref/burgers2d_{i}.dat"}, category)
                        plotted_problems += 1
                except:
                    pass
        
        if category == 'ns' and 'NS2D_LidDriven' in problems:
            # we try different NS2D_LidDriven variants
            for a in [2, 4, 6, 8, 10, 16, 32]:
                try:
                    result = load_ref_data_safe(f"ref/lid_driven_a{a}.dat")
                    if result[0] is not None:
                        plot_problem(f"NS2D_LidDriven_a{a}", problems['NS2D_LidDriven'],
                                   {"datapath": f"ref/lid_driven_a{a}.dat", "a": a}, category)
                        plotted_problems += 1
                except:
                    pass
        
        if category == 'wave' and 'Wave1D' in problems:
            # we try different Wave1D variants
            for a in [2, 6, 8, 10]:
                try:
                    plot_problem(f"Wave1D_a{a}", problems['Wave1D'], {"a": a}, category)
                    plotted_problems += 1
                except:
                    pass
        
        if category == 'heat' and 'HeatND' in problems:
            # we try different HeatND variants
            for dim in [4, 6, 8, 10]:
                try:
                    plot_problem(f"HeatND_dim{dim}", problems['HeatND'], {"dim": dim}, category)
                    plotted_problems += 1
                except:
                    pass
    
    print(f"\n✓ Plotted {plotted_problems}/{total_problems} problems")
    return plotted_problems, total_problems


def create_summary():
    """we create a summary of all plotted datasets"""
    print("\n" + "="*80)
    print("Creating Summary")
    print("="*80)
    
    # we count plots in each directory
    counts = {}
    for subdir in ["1d", "2d", "3d", "nd", "time_series"]:
        count = len(list((output_dir / subdir).glob("*.png")))
        counts[subdir] = count
        print(f"  {subdir}: {count} plots")
    
    total = sum(counts.values())
    print(f"\n  Total: {total} plots")
    
    # we create a summary file
    summary_file = output_dir / "DATASETS_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("PINNacle Benchmark Datasets Summary\n")
        f.write("="*80 + "\n\n")
        for category, problems in PINNACLE_PROBLEMS.items():
            f.write(f"{category.upper()}:\n")
            for name in problems.keys():
                f.write(f"  - {name}\n")
            f.write("\n")
        f.write(f"\nTotal problems: {sum(len(v) for v in PINNACLE_PROBLEMS.values())}\n")
        f.write(f"Total plots generated: {total}\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")


def main():
    """we run comprehensive plotting"""
    print("="*80)
    print("Comprehensive Benchmark Dataset Plotting")
    print("Plotting ALL datasets from ALL benchmarks")
    print("="*80)
    
    plotted = 0
    total = 0
    
    if PINNACLE_AVAILABLE:
        plotted, total = plot_all_pinnacle_problems()
    
    create_summary()
    
    print("\n" + "="*80)
    print(f"✓ Complete! Generated plots for {plotted} problems")
    print(f"  All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
