import numpy as np

from experiments.spectral_self_attention.multihead_phase import (
    cluster_endpoints,
    normalize,
    probe_field,
)


def test_probe_field_is_tangent_and_heads_add() -> None:
    probes = normalize(np.array([[1.0, 0.2], [-0.3, 1.0]]))
    anchors = normalize(np.array([[0.7, 0.4], [-0.6, 0.9], [-0.8, -0.3]]))
    matrix = np.diag([2.0, -3.0])
    one_head = probe_field(probes, anchors, [matrix], beta=4.0)
    two_identical_heads = probe_field(probes, anchors, [matrix, matrix], beta=4.0)

    np.testing.assert_allclose(np.sum(one_head * probes, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(two_identical_heads, 2.0 * one_head, atol=1e-12)


def test_endpoint_clustering_is_deterministic_on_circle() -> None:
    angles = np.array([0.01, -0.01, np.pi / 2 - 0.01, np.pi / 2 + 0.01])
    endpoints = np.column_stack([np.cos(angles), np.sin(angles)])
    labels, centres, counts = cluster_endpoints(endpoints, angular_tolerance=0.05)

    assert labels.tolist() == [0, 0, 1, 1]
    assert counts.tolist() == [2, 2]
    np.testing.assert_allclose(centres, np.array([[1.0, 0.0], [0.0, 1.0]]), atol=1e-12)
