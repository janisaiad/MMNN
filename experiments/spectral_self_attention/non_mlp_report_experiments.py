"""Reproducible figures and focused checks for the non-MLP report.

The script does not search for a new attractor.  It independently replays the
attention-only cycle already isolated by the component ablation, measures its
return time, checks attraction toward the closed orbit, and compares finite
normalized residual layers with the limiting ODE.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "spectral_self_attention"
REPORT = ROOT / "refs" / "spectral_self_attention" / "non_mlp_cycles_paper"
FIGURES = REPORT / "figures"


def wrap(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def softmax_attention(
    angles: np.ndarray, score: np.ndarray, value: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    scores = np.einsum("...id,de,...je->...ij", tokens, score, tokens, optimize=True)
    logits = beta * scores
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    transported = np.einsum("de,...je->...jd", value, tokens, optimize=True)
    output = np.einsum("...ij,...jd->...id", weights, transported, optimize=True)
    return output, weights


def angular_field(
    angles: np.ndarray,
    score: np.ndarray,
    value: np.ndarray,
    beta: float,
    depth_scale: float,
) -> np.ndarray:
    output, _ = softmax_attention(angles, score, value, beta)
    tangent = np.stack((-np.sin(angles), np.cos(angles)), axis=-1)
    return depth_scale * np.einsum("...id,...id->...i", tangent, output)


def rk4_step(
    angles: np.ndarray,
    dt: float,
    score: np.ndarray,
    value: np.ndarray,
    beta: float,
    depth_scale: float,
) -> np.ndarray:
    field = lambda z: angular_field(z, score, value, beta, depth_scale)
    k1 = field(angles)
    k2 = field(wrap(angles + 0.5 * dt * k1))
    k3 = field(wrap(angles + 0.5 * dt * k2))
    k4 = field(wrap(angles + dt * k3))
    return wrap(angles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def layer_step(
    angles: np.ndarray,
    ratio: float,
    score: np.ndarray,
    value: np.ndarray,
    beta: float,
    depth_scale: float,
) -> np.ndarray:
    output, _ = softmax_attention(angles, score, value, beta)
    tokens = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    updated = tokens + ratio * depth_scale * output
    updated /= np.linalg.norm(updated, axis=-1, keepdims=True)
    return np.arctan2(updated[..., 1], updated[..., 0])


def simulate_ode(
    initial: np.ndarray,
    duration: float,
    dt: float,
    sample_spacing: float,
    score: np.ndarray,
    value: np.ndarray,
    beta: float,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(initial, dtype=float).copy()
    steps = int(np.ceil(duration / dt))
    stride = max(1, int(round(sample_spacing / dt)))
    history = []
    times = []
    for step in range(steps):
        angles = rk4_step(angles, dt, score, value, beta, depth_scale)
        if (step + 1) % stride == 0:
            history.append(angles.copy())
            times.append((step + 1) * dt)
    return np.asarray(times), np.asarray(history)


def recurrence_period(history: np.ndarray, spacing: float) -> tuple[float, float, int]:
    minimum = max(3, int(round(4.0 / spacing)))
    maximum = min(len(history) // 2, int(round(14.0 / spacing)))
    errors = []
    for lag in range(minimum, maximum + 1):
        error = np.quantile(np.abs(wrap(history[lag:] - history[:-lag])), 0.9)
        errors.append((float(error), lag))
    error, lag = min(errors)
    return lag * spacing, error, lag


def distance_to_orbit(states: np.ndarray, orbit: np.ndarray) -> np.ndarray:
    differences = wrap(states[:, None, :] - orbit[None, :, :])
    distances = np.sqrt(np.mean(differences**2, axis=-1))
    return np.min(distances, axis=1)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    payload = json.loads(
        (DATA / "continuous_ode_f4_p3_cycle_attention_only.json").read_text()
    )
    record = payload["records"][0]
    model = record["model"]
    score = np.asarray(model["score"], dtype=float)
    value = np.asarray(model["value"], dtype=float)
    beta = float(model["beta"])
    depth_scale = float(model["step_size"])
    initial = np.asarray(record["final_angle"], dtype=float)

    # Settle once more with a smaller RK4 step, then save a long clean trace.
    _, settled = simulate_ode(
        initial, 120.0, 0.005, 0.05, score, value, beta, depth_scale
    )
    times, history = simulate_ode(
        settled[-1], 55.0, 0.005, 0.01, score, value, beta, depth_scale
    )
    period, recurrence_error, period_lag = recurrence_period(history, 0.01)
    orbit = history[:period_lag]

    tokens = np.stack((np.cos(history), np.sin(history)), axis=-1)
    gram = np.einsum("tid,tjd->tij", tokens, tokens, optimize=True)
    _, weights = softmax_attention(history, score, value, beta)
    relative = wrap(history[:, 1:] - history[:, :1])

    set_plot_style()
    colors = ("#1565c0", "#d84315", "#2e7d32")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    cycle_points = history[:period_lag]
    cycle_relative = relative[:period_lag]
    phase_color = np.linspace(0.0, 1.0, len(cycle_relative))
    scatter = axes[0].scatter(
        cycle_relative[:, 0],
        cycle_relative[:, 1],
        c=phase_color,
        s=5,
        cmap="viridis",
        linewidths=0,
    )
    axes[0].set_xlabel(r"angle du jeton 2 moins angle du jeton 1")
    axes[0].set_ylabel(r"angle du jeton 3 moins angle du jeton 1")
    axes[0].set_title("La forme interne décrit une boucle fermée")
    x_margin = max(0.08, 0.12 * np.ptp(cycle_relative[:, 0]))
    y_margin = max(0.08, 0.12 * np.ptp(cycle_relative[:, 1]))
    axes[0].set_xlim(
        np.min(cycle_relative[:, 0]) - x_margin,
        np.max(cycle_relative[:, 0]) + x_margin,
    )
    axes[0].set_ylim(
        np.min(cycle_relative[:, 1]) - y_margin,
        np.max(cycle_relative[:, 1]) + y_margin,
    )
    colorbar = fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("avancement dans un tour")

    circle = plt.Circle((0, 0), 1.0, fill=False, color="#777777", linewidth=0.8)
    axes[1].add_patch(circle)
    phase_indices = np.linspace(0, period_lag - 1, 7, dtype=int)
    for phase_number, index in enumerate(phase_indices):
        alpha = 0.24 + 0.11 * phase_number
        for token_index, color in enumerate(colors):
            point = np.array(
                [np.cos(cycle_points[index, token_index]), np.sin(cycle_points[index, token_index])]
            )
            axes[1].plot(
                [0.0, point[0]],
                [0.0, point[1]],
                color=color,
                alpha=alpha,
                linewidth=0.8,
            )
            axes[1].scatter(*point, s=14, color=color, alpha=alpha)
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.12, 1.12)
    axes[1].set_ylim(-1.12, 1.12)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Positions des trois jetons pendant un tour")
    for token_index, color in enumerate(colors):
        axes[1].plot([], [], color=color, marker="o", linestyle="", label=f"jeton {token_index + 1}")
    axes[1].legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURES / "attention_only_cycle_geometry.pdf")
    fig.savefig(FIGURES / "attention_only_cycle_geometry.png")
    plt.close(fig)

    one_turn = times <= period
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.25), sharex=True)
    pairs = ((0, 1), (0, 2), (1, 2))
    for color, (left, right) in zip(colors, pairs, strict=True):
        axes[0].plot(
            times[one_turn],
            gram[one_turn, left, right],
            color=color,
            label=f"jetons {left + 1}–{right + 1}",
        )
    axes[0].set_ylabel("proximité (1 = mêmes directions)")
    axes[0].set_title("Les distances entre jetons changent réellement")
    axes[0].legend(ncol=3, frameon=False)
    for index, color in enumerate(colors):
        axes[1].plot(
            times[one_turn],
            weights[one_turn, 0, index],
            color=color,
            label=f"poids du jeton 1 vers {index + 1}",
        )
    axes[1].set_xlabel("profondeur normalisée")
    axes[1].set_ylabel("poids d’attention")
    axes[1].set_title("Les choix d’attention oscillent avec la géométrie")
    axes[1].legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "attention_only_cycle_observables.pdf")
    fig.savefig(FIGURES / "attention_only_cycle_observables.png")
    plt.close(fig)

    # Transverse attraction: phase shifts along the cycle are removed by measuring
    # distance to the entire closed orbit rather than to one synchronized replay.
    rng = np.random.default_rng(260816101)
    perturbed = wrap(orbit[0] + 0.08 * rng.normal(size=(128, len(initial))))
    contraction_times = np.arange(0.0, 36.0 + 1e-9, 0.25)
    contraction = []
    contraction.append(distance_to_orbit(perturbed, orbit))
    current_time = 0.0
    for target in contraction_times[1:]:
        for _ in range(int(round((target - current_time) / 0.01))):
            perturbed = rk4_step(
                perturbed, 0.01, score, value, beta, depth_scale
            )
        current_time = target
        contraction.append(distance_to_orbit(perturbed, orbit))
    contraction = np.asarray(contraction)

    ratios = np.asarray(
        [1 / 16, 1 / 31, 1 / 64, 1 / 127, 1 / 256, 1 / 509, 1 / 1024],
        dtype=float,
    )
    layer_counts = []
    physical_returns = []
    return_errors = []
    map_state = orbit[0].copy()
    for ratio in ratios:
        state = map_state.copy()
        for _ in range(int(np.ceil(80.0 / ratio))):
            state = layer_step(state, ratio, score, value, beta, depth_scale)
        reference = state.copy()
        low = max(1, int(np.floor(0.72 * period / ratio)))
        high = int(np.ceil(1.28 * period / ratio))
        best_error = np.inf
        best_lag = 0
        for lag in range(1, high + 1):
            state = layer_step(state, ratio, score, value, beta, depth_scale)
            if lag >= low:
                error = float(np.max(np.abs(wrap(state - reference))))
                if error < best_error:
                    best_error = error
                    best_lag = lag
        layer_counts.append(best_lag)
        physical_returns.append(best_lag * ratio)
        return_errors.append(best_error)
    layer_counts = np.asarray(layer_counts)
    physical_returns = np.asarray(physical_returns)
    return_errors = np.asarray(return_errors)
    slope, intercept = np.polyfit(np.log(1.0 / ratios), np.log(layer_counts), 1)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
    median = np.median(contraction, axis=1)
    low = np.quantile(contraction, 0.1, axis=1)
    high = np.quantile(contraction, 0.9, axis=1)
    axes[0].fill_between(contraction_times, low, high, color="#90caf9", alpha=0.45)
    axes[0].semilogy(contraction_times, median, color="#1565c0", linewidth=1.5)
    axes[0].set_xlabel("profondeur normalisée")
    axes[0].set_ylabel("distance à la boucle")
    axes[0].set_title("128 perturbations reviennent vers le cycle")
    inverse = 1.0 / ratios
    axes[1].loglog(inverse, layer_counts, "o", color="#d84315", label="mesure")
    fitted = np.exp(intercept) * inverse**slope
    axes[1].loglog(inverse, fitted, "--", color="#555555", label=f"pente {slope:.3f}")
    axes[1].set_xlabel("affaiblissement d’une couche (1 / ratio)")
    axes[1].set_ylabel("nombre de couches par tour")
    axes[1].set_title("Un tour demande proportionnellement plus de couches")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "attention_only_cycle_stability_depth.pdf")
    fig.savefig(FIGURES / "attention_only_cycle_stability_depth.png")
    plt.close(fig)

    # Small-system equilibrium catalogue from the independently saturated root search.
    all_rows = []
    with (DATA / "equilibria" / "planar_small_systems.csv").open() as handle:
        for row in csv.DictReader(handle):
            all_rows.append(row)
    case_order = [
        "indefinite_n2",
        "indefinite_n3",
        "negative_n2",
        "negative_n3",
        "positive_n2",
        "positive_n3",
    ]
    first_seed = {
        case: min(int(row["seed"]) for row in all_rows if row["case"] == case)
        for case in case_order
    }
    rows = [
        row
        for row in all_rows
        if int(row["seed"]) == first_seed[row["case"]]
    ]
    totals = []
    stable = []
    for case in case_order:
        selected_rows = [row for row in rows if row["case"] == case]
        totals.append(len(selected_rows))
        stable.append(sum(row["stable"] == "True" for row in selected_rows))
    labels = ["indéfini\n2", "indéfini\n3", "négatif\n2", "négatif\n3", "positif\n2", "positif\n3"]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.bar(positions, totals, width=0.68, color="#bbdefb", label="équilibres distincts")
    ax.bar(positions, stable, width=0.42, color="#1565c0", label="stables")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("nombre trouvé")
    ax.set_xlabel("type des deux valeurs propres, puis nombre de jetons")
    ax.set_title("Même sans MLP, plusieurs destinations stables peuvent coexister")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "small_equilibrium_catalogue.pdf")
    fig.savefig(FIGURES / "small_equilibrium_catalogue.png")
    plt.close(fig)

    spectrum_payload = json.loads(
        (
            DATA
            / "continuous_lyapunov_spectrum_f4_p3_cycle_attention_only_relaxed2.json"
        ).read_text()
    )
    result = {
        "source_identity": record["identity"],
        "model": {
            "score": score.tolist(),
            "value": value.tolist(),
            "beta": beta,
            "depth_scale": depth_scale,
        },
        "ode": {
            "rk4_step": 0.005,
            "period_normalized": period,
            "period_natural_ode_time": period * depth_scale,
            "recurrence_error_90_percentile_radians": recurrence_error,
            "gram_range": {
                f"{left + 1}-{right + 1}": [
                    float(np.min(gram[:period_lag, left, right])),
                    float(np.max(gram[:period_lag, left, right])),
                ]
                for left, right in pairs
            },
            "attention_weight_range_from_token_1": [
                [float(np.min(weights[:period_lag, 0, index])), float(np.max(weights[:period_lag, 0, index]))]
                for index in range(3)
            ],
            "lyapunov_spectrum_per_normalized_time": spectrum_payload["spectrum"],
            "perturbation_distance": {
                "initial_median": float(median[0]),
                "final_median": float(median[-1]),
                "initial_90_percentile": float(high[0]),
                "final_90_percentile": float(high[-1]),
            },
        },
        "finite_layers": {
            "ratios": ratios.tolist(),
            "layer_counts": layer_counts.tolist(),
            "normalized_return_times": physical_returns.tolist(),
            "return_errors_radians": return_errors.tolist(),
            "log_log_slope": float(slope),
        },
        "small_equilibrium_catalogue": {
            case: {"total": total, "stable": stable_count}
            for case, total, stable_count in zip(case_order, totals, stable, strict=True)
        },
    }
    (DATA / "non_mlp_report_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
