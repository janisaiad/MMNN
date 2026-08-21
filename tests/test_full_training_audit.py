from __future__ import annotations

import math
import runpy
from pathlib import Path

AUDIT = runpy.run_path(
    Path(__file__).parents[1]
    / "experiments"
    / "mup_dmft_frequency"
    / "audit_full_training_results.py"
)
all_json_numbers_finite = AUDIT["all_json_numbers_finite"]
expected_calibration_tags = AUDIT["expected_calibration_tags"]
expected_campaign_tags = AUDIT["expected_campaign_tags"]
expected_discretization_tags = AUDIT["expected_discretization_tags"]
expected_masked_spectrum_tags = AUDIT["expected_masked_spectrum_tags"]
expected_masked_tags = AUDIT["expected_masked_tags"]
first_numeric_crossing = AUDIT["first_numeric_crossing"]
matched_discretization_log_ratios = AUDIT["matched_discretization_log_ratios"]
median_event = AUDIT["median_event"]


def test_prespecified_manifests_have_exact_sizes() -> None:
    assert len(expected_masked_tags()) == 87
    assert len(expected_masked_spectrum_tags()) == 6
    assert len(expected_calibration_tags()) == 72
    assert len(expected_campaign_tags()) == 184
    assert len(expected_discretization_tags()) == 60


def test_right_censored_median_is_not_imputed() -> None:
    four_events = [
        {"t50_8": value}
        for value in ("25", "50", "75", "100", "", "", "", "")
    ]
    three_events = [
        {"t50_8": value}
        for value in ("25", "50", "75", "", "", "", "", "")
    ]
    assert median_event(four_events, 8) == 100.0
    assert median_event(three_events, 8) is None


def test_discretization_ratios_are_matched_within_seed() -> None:
    rows: list[dict[str, str]] = []
    for architecture in ("fc", "mmnn"):
        for optimizer in ("mup_gd", "muon_p1_3"):
            for seed in range(30, 33):
                base = float(seed)
                rows.extend(
                    (
                        {
                            "tag": (
                                f"dt_{architecture}_{optimizer}_base_s{seed}"
                            ),
                            "final_loss": str(base),
                        },
                        {
                            "tag": (
                                f"dt_{architecture}_{optimizer}_half_s{seed}"
                            ),
                            "final_loss": str(2.0 * base),
                        },
                        {
                            "tag": (
                                f"dt_{architecture}_{optimizer}_quarter_s{seed}"
                            ),
                            "final_loss": str(4.0 * base),
                        },
                        {
                            "tag": (
                                f"dt_{architecture}_{optimizer}_eighth_s{seed}"
                            ),
                            "final_loss": str(8.0 * base),
                        },
                        {
                            "tag": (
                                f"grid_{architecture}_{optimizer}_g256_s{seed}"
                            ),
                            "final_loss": str(0.5 * base),
                        },
                    )
                )
    base_half, half_quarter, quarter_eighth, grid = (
        matched_discretization_log_ratios(rows)
    )
    assert (
        len(base_half)
        == len(half_quarter)
        == len(quarter_eighth)
        == len(grid)
        == 12
    )
    assert all(math.isclose(value, math.log(2.0)) for value in base_half)
    assert all(math.isclose(value, math.log(2.0)) for value in half_quarter)
    assert all(math.isclose(value, math.log(2.0)) for value in quarter_eighth)
    assert all(math.isclose(value, math.log(2.0)) for value in grid)


def test_json_finiteness_walk_rejects_nested_nan() -> None:
    assert all_json_numbers_finite({"a": [1.0, {"b": 2.0}]})
    assert not all_json_numbers_finite({"a": [float("nan")]})


def test_threshold_recomputation_preserves_blank_cells() -> None:
    rows = [
        {"step": "0", "relative_error_8": ""},
        {"step": "10", "relative_error_8": "0.8"},
        {"step": "20", "relative_error_8": "None"},
        {"step": "30", "relative_error_8": "0.4"},
    ]
    assert first_numeric_crossing(
        rows,
        time_key="step",
        value_key="relative_error_8",
        threshold=0.5,
        direction="below",
    ) == 30.0
    assert (
        first_numeric_crossing(
            rows,
            time_key="step",
            value_key="missing_frequency",
            threshold=0.5,
            direction="below",
        )
        is None
    )
