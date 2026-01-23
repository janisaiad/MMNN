#!/usr/bin/env python3
"""we generate preview plot of the function to fit"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# we configure matplotlib for LaTeX formatting
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

def target_function(x):
    """we define the multi-frequency function with phase shifts"""
    return np.cos(12 * np.pi * x) + np.cos(24 * np.pi * x + 0.5) + np.cos(36 * np.pi * x) + np.cos(72 * np.pi * x + 0.5)

# we generate plot
x = np.linspace(-1, 1, 5000)
y = target_function(x)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, 'b-', linewidth=1.5, label='$f(x) = \\cos(12\\pi x) + \\cos(24\\pi x + 0.5) + \\cos(36\\pi x) + \\cos(72\\pi x + 0.5)$')
ax.set_xlabel('$x$', fontsize=22)
ax.set_ylabel('$f(x)$', fontsize=22)
ax.set_title('Multi-Frequency Function to Fit\n(with phase shifts: 24π and 72π shifted by 0.5)', fontsize=20)
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=18, loc='best')
ax.tick_params(labelsize=18)
ax.set_xlim(-1, 1)
plt.tight_layout()

output_path = Path(__file__).parent / "function_to_fit_preview.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Preview plot saved to: {output_path}")
print(f"Function: cos(12πx) + cos(24πx + 0.5) + cos(36πx) + cos(72πx + 0.5)")
print(f"Range: x ∈ [-1, 1]")
