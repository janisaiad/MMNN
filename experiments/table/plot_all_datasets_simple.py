#!/usr/bin/env python3
"""
we plot ALL datasets from ALL benchmarks by directly processing data files
this is a simpler, more robust approach that doesn't require instantiating problem classes
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from scipy.interpolate import griddata
import re

# we create output directory
output_dir = Path(__file__).parent / "plots_to_fit"
output_dir.mkdir(exist_ok=True)

# we create subdirectories
for subdir in ["1d", "2d", "3d", "nd", "time_series"]:
    (output_dir / subdir).mkdir(exist_ok=True)

print(f"we will save plots to {output_dir}")


def load_data_file(filepath):
    """we load a data file, handling COMSOL format"""
    try:
        with open(filepath, 'r') as f:
            header_lines = []
            for line in f:
                if line.startswith('%'):
                    header_lines.append(line)
                else:
                    break
            data = np.loadtxt(filepath, comments="%")
            return data, header_lines
    except Exception as e:
        return None, None


def parse_comsol_times(header_lines):
    """we parse COMSOL header for time information"""
    if not header_lines:
        return None
    for line in header_lines:
        if '@ t=' in line:
            times = re.findall(r'@ t=([\d.]+)', line)
            return [float(t) for t in times]
    return None


def plot_1d(x, u, title, filename, subdir="1d"):
    """we plot 1D function"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, u, 'b-', linewidth=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / subdir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_1d_time(x, t_slices, u_slices, title, filename, subdir="time_series"):
    """we plot 1D time evolution"""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_slices)))
    for i, (t, u) in enumerate(zip(t_slices, u_slices)):
        if u is not None and len(u) > 0:
            ax.plot(x, u, linewidth=2, color=colors[i], label=f't={t:.2f}')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('u(x,t)', fontsize=12)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if len(t_slices) <= 8:
        ax.legend(loc='best', ncol=2, fontsize=9)
    try:
        plt.tight_layout()
    except:
        pass
    plt.savefig(output_dir / subdir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_2d_contour(x, y, u, title, filename, subdir="2d"):
    """we plot 2D contour"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        X, Y = np.meshgrid(x, y)
        if u.ndim == 1:
            u_grid = u.reshape(len(y), len(x))
        else:
            u_grid = u
        if np.isnan(u_grid).all():
            return False
        contour = ax.contourf(X, Y, u_grid, levels=20, cmap='viridis')
        ax.contour(X, Y, u_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(contour, ax=ax, label='u(x,y)')
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(output_dir / subdir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        return False


def plot_2d_time_slices(x, y, t_slices, u_slices, title, filename, subdir="2d"):
    """we plot 2D time slices"""
    try:
        valid_slices = [(t, u) for t, u in zip(t_slices, u_slices) if u is not None and not np.isnan(u).all()]
        if not valid_slices:
            return False
        
        t_slices, u_slices = zip(*valid_slices)
        n_slices = len(t_slices)
        n_cols = min(4, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        axes = axes.flatten()
        
        X, Y = np.meshgrid(x, y)
        vmin = min([u.min() for u in u_slices if not np.isnan(u).all()])
        vmax = max([u.max() for u in u_slices if not np.isnan(u).all()])
        
        for i, (t, u) in enumerate(zip(t_slices, u_slices)):
            if u.ndim == 1:
                u_grid = u.reshape(len(y), len(x))
            else:
                u_grid = u
            ax = axes[i]
            contour = ax.contourf(X, Y, u_grid, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f't={t:.2f}', fontsize=10)
            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)
            plt.colorbar(contour, ax=ax)
        
        for i in range(len(t_slices), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=11, fontweight='bold', y=0.995)
        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(output_dir / subdir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        return False


def process_data_file(filepath):
    """we process a single data file and create plots"""
    filename = filepath.name
    print(f"\n  📊 Processing: {filename}")
    
    data, header_lines = load_data_file(filepath)
    if data is None or len(data) == 0:
        print(f"    ⚠ Could not load data")
        return False
    
    # we determine format
    n_cols = data.shape[1]
    times = parse_comsol_times(header_lines)
    
    plotted = False
    
    try:
        if n_cols == 2:
            # we assume 1D: x, u
            x = data[:, 0]
            u = data[:, 1]
            plot_1d(x, u, filename.replace('.dat', ''), filename.replace('.dat', '.png'), "1d")
            plotted = True
        
        elif n_cols == 3:
            # we could be 1D time-dependent (x, t, u) or 2D static (x, y, u)
            x_coords = np.unique(data[:, 0])
            second_coords = np.unique(data[:, 1])
            
            if len(second_coords) < 10:  # we assume time dimension
                # we treat as 1D time-dependent
                t_slices = second_coords[::max(1, len(second_coords)//4)][:4]
                u_slices = []
                for t in t_slices:
                    mask = np.isclose(data[:, 1], t)
                    if np.any(mask):
                        u_t = data[mask, 2]
                        x_t = data[mask, 0]
                        u_interp = griddata(x_t, u_t, x_coords, method='linear', fill_value=0)
                        u_slices.append(u_interp)
                if u_slices:
                    plot_1d_time(x_coords, t_slices, u_slices, 
                               filename.replace('.dat', ''), 
                               filename.replace('.dat', '.png'), "time_series")
                    plotted = True
            else:
                # we treat as 2D static
                x_coords = np.unique(data[:, 0])
                y_coords = np.unique(data[:, 1])
                u_data = data[:, 2]
                X, Y = np.meshgrid(x_coords, y_coords)
                try:
                    u_grid = griddata((data[:, 0], data[:, 1]), u_data, (X, Y), method='linear')
                    if u_grid is not None and not np.isnan(u_grid).all():
                        plot_2d_contour(x_coords, y_coords, u_grid,
                                      filename.replace('.dat', ''),
                                      filename.replace('.dat', '.png'), "2d")
                        plotted = True
                except:
                    pass
        
        elif n_cols > 3:
            # we check if it's COMSOL format with time slices
            if times is not None:
                # we handle COMSOL format
                x_coords = np.unique(data[:, 0])
                y_coords = np.unique(data[:, 1]) if len(np.unique(data[:, 1])) > 1 else [0]
                
                if len(y_coords) == 1:
                    # we treat as 1D time-dependent
                    n_time = len(times)
                    n_plot = min(4, n_time)
                    time_indices = np.linspace(0, n_time-1, n_plot, dtype=int)
                    t_slices = [times[i] for i in time_indices]
                    u_slices = []
                    for idx in time_indices:
                        u_col = 2 + idx
                        if u_col < n_cols:
                            u_data = data[:, u_col]
                            u_slices.append(u_data)
                    if u_slices:
                        plot_1d_time(x_coords, t_slices, u_slices,
                                   filename.replace('.dat', ''),
                                   filename.replace('.dat', '.png'), "time_series")
                        plotted = True
                else:
                    # we treat as 2D time-dependent
                    n_time = len(times)
                    n_plot = min(4, n_time)
                    time_indices = np.linspace(0, n_time-1, n_plot, dtype=int)
                    t_slices = [times[i] for i in time_indices]
                    u_slices = []
                    for idx in time_indices:
                        u_col = 2 + idx
                        if u_col < n_cols:
                            u_data = data[:, u_col]
                            X, Y = np.meshgrid(x_coords, y_coords)
                            try:
                                u_grid = griddata((data[:, 0], data[:, 1]), u_data, (X, Y), method='linear')
                                if u_grid is not None and not np.isnan(u_grid).all():
                                    u_slices.append(u_grid)
                            except:
                                pass
                    if u_slices:
                        plot_2d_time_slices(x_coords, y_coords, t_slices, u_slices,
                                           filename.replace('.dat', ''),
                                           filename.replace('.dat', '.png'), "2d")
                        plotted = True
            else:
                # we check if it's multi-component (e.g., u, v, p)
                x_coords = np.unique(data[:, 0])
                y_coords = np.unique(data[:, 1])
                
                if len(y_coords) > 1:
                    # we plot each component
                    for comp_idx in range(2, min(n_cols, 6)):  # we plot up to 4 components
                        u_data = data[:, comp_idx]
                        X, Y = np.meshgrid(x_coords, y_coords)
                        try:
                            u_grid = griddata((data[:, 0], data[:, 1]), u_data, (X, Y), method='linear')
                            if u_grid is not None and not np.isnan(u_grid).all():
                                comp_name = ['u', 'v', 'p', 'w'][comp_idx-2] if comp_idx-2 < 4 else f'comp{comp_idx-2}'
                                plot_2d_contour(x_coords, y_coords, u_grid,
                                              f"{filename.replace('.dat', '')} - {comp_name}",
                                              f"{filename.replace('.dat', '')}_{comp_name}.png", "2d")
                                plotted = True
                        except:
                            pass
    except Exception as e:
        print(f"    ⚠ Error processing: {e}")
    
    if plotted:
        print(f"    ✓ Plotted successfully")
    else:
        print(f"    ⚠ Could not determine format or plot")
    
    return plotted


def main():
    """we process all data files"""
    print("="*80)
    print("Plotting ALL Datasets from ALL Benchmarks")
    print("Processing data files directly")
    print("="*80)
    
    # we find all data files
    ref_dir = Path(__file__).parent / "PINNacle" / "ref"
    if not ref_dir.exists():
        print(f"⚠ Reference directory not found: {ref_dir}")
        return
    
    data_files = sorted(ref_dir.glob("*.dat"))
    print(f"\nFound {len(data_files)} data files")
    
    plotted_count = 0
    for data_file in data_files:
        if process_data_file(data_file):
            plotted_count += 1
    
    # we create summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for subdir in ["1d", "2d", "3d", "nd", "time_series"]:
        count = len(list((output_dir / subdir).glob("*.png")))
        print(f"  {subdir}: {count} plots")
    
    total = sum(len(list((output_dir / subdir).glob("*.png"))) for subdir in ["1d", "2d", "3d", "nd", "time_series"])
    print(f"\n  Total: {total} plots generated")
    print(f"  Processed: {plotted_count}/{len(data_files)} files")
    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
