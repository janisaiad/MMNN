import numpy as np

from experiments.spectral_self_attention.mean_field_extensions import (
    mixed_equation,
    no_sphere_summary,
    polygon_kernel_rows,
    roots_for_kernel,
)


def test_mixed_roots_solve_each_kernel_equation() -> None:
    for kernel in ["exponential", "sigmoid", "softplus", "polynomial4"]:
        roots = roots_for_kernel(kernel)
        assert roots
        np.testing.assert_allclose(
            [mixed_equation(kernel, root) for root in roots], 0.0, atol=1e-11
        )


def test_kernel_shape_changes_polygon_stability_but_not_normalization_sign() -> None:
    rows = polygon_kernel_rows()
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["kernel"]), int(row["root_index"])), []).append(row)
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0]["stable"] == pair[1]["stable"]

    assert any(row["stable"] for row in rows if row["kernel"] == "exponential")
    assert any(row["stable"] for row in rows if row["kernel"] == "polynomial4")
    assert not any(row["stable"] for row in rows if row["kernel"] == "sigmoid")
    assert not any(row["stable"] for row in rows if row["kernel"] == "softplus")


def test_removing_sphere_can_create_finite_time_blowup() -> None:
    summary = no_sphere_summary()
    assert summary["projected_norm"] == 1.0
    assert (
        summary["unprojected_unnormalized_finite_blowup_time"]
        < summary["unprojected_row_normalized_time_to_target"]
    )
