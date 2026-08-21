#!/usr/bin/env python3
"""Exact pathwise NTK recursion for frozen-feature MMNNs.

The normalized architecture is

    z_l(x) = W_l h_{l-1}(x) / sqrt(r_{l-1}) + b_l,
    h_l(x) = A_l phi(z_l(x)) / sqrt(n_l) + c_l,

with frozen ``(W_l,b_l)`` and trainable ``(A_l,c_l)``.  This module does not
take an infinite-width, infinite-rank, Gaussian-process, or independence
limit.  For every realization it evaluates

    Theta_l(x,y) = q_l(x,y) I + J_l(x) Theta_{l-1}(x,y) J_l(y)^T,

where q_l=phi(z_l(x)).phi(z_l(y))/n_l+1 and
J_l=A_l D_l W_l/sqrt(n_l r_{l-1}).

The identity follows directly from the chain rule and is the correct base
object for MMNN Feynman diagrams.  The rank-r Gram line appears after
concatenating two blocks: the tangent metric of M=W A/sqrt(r) with respect to
A is (W W^T/r) tensor the identity on the right index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MMNNLayer:
    """One frozen-feature/trainable-readout MMNN block."""

    W: Array
    b: Array
    A: Array
    c: Array

    def validate(self, input_rank: int) -> None:
        width, actual_input_rank = self.W.shape
        output_rank, actual_width = self.A.shape
        if actual_input_rank != input_rank:
            raise ValueError((actual_input_rank, input_rank))
        if actual_width != width or self.b.shape != (width,):
            raise ValueError("incompatible width indices in W, b, and A")
        if self.c.shape != (output_rank,):
            raise ValueError("incompatible output indices in A and c")


def relu(x: Array) -> Array:
    return np.maximum(x, 0.0)


def relu_derivative(x: Array) -> Array:
    return (x > 0.0).astype(x.dtype)


def forward_with_cache(
    inputs: Array,
    layers: Sequence[MMNNLayer],
    activation: Callable[[Array], Array] = relu,
) -> tuple[Array, list[tuple[Array, Array, Array]]]:
    """Return outputs and per-layer (input, preactivation, feature) caches."""
    h = np.asarray(inputs, dtype=float)
    if h.ndim != 2:
        raise ValueError("inputs must have shape (number of samples, input rank)")
    caches = []
    for layer in layers:
        layer.validate(h.shape[1])
        z = h @ layer.W.T / np.sqrt(h.shape[1]) + layer.b
        feature = activation(z)
        caches.append((h, z, feature))
        h = feature @ layer.A.T / np.sqrt(feature.shape[1]) + layer.c
    return h, caches


def exact_pathwise_ntk(
    inputs: Array,
    layers: Sequence[MMNNLayer],
    activation: Callable[[Array], Array] = relu,
    activation_derivative: Callable[[Array], Array] = relu_derivative,
) -> Array:
    """Exact matrix-valued empirical NTK for trainable A_l and c_l.

    The result has indices ``[sample_x, sample_y, output_x, output_y]``.
    """
    _, caches = forward_with_cache(inputs, layers, activation)
    sample_count = len(inputs)
    input_rank = np.asarray(inputs).shape[1]
    theta = np.zeros((sample_count, sample_count, input_rank, input_rank))
    for layer, (_, z, feature) in zip(layers, caches):
        width = feature.shape[1]
        output_rank = layer.A.shape[0]
        current = np.empty(
            (sample_count, sample_count, output_rank, output_rank), dtype=float
        )
        jacobians = np.einsum(
            "aw,xw,wi->xai",
            layer.A,
            activation_derivative(z),
            layer.W,
            optimize=True,
        ) / np.sqrt(width * input_rank)
        for x in range(sample_count):
            for y in range(sample_count):
                direct = float(feature[x] @ feature[y]) / width + 1.0
                transported = jacobians[x] @ theta[x, y] @ jacobians[y].T
                current[x, y] = transported + direct * np.eye(output_rank)
        theta = current
        input_rank = output_rank
    return theta


def concatenated_factor_tangent_metric(W: Array) -> Array:
    """Metric inserted by M=W A/sqrt(r) when only A is trainable."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a matrix")
    rank = W.shape[1]
    return W @ W.T / rank


def explicit_parameter_jacobian_ntk(
    inputs: Array,
    layers: Sequence[MMNNLayer],
) -> Array:
    """Slow finite-difference-free reference using explicit forward tangents.

    This independent implementation propagates one tangent for every scalar
    trainable parameter.  It is intended for tests and small audits.
    """
    outputs, caches = forward_with_cache(inputs, layers)
    sample_count, final_rank = outputs.shape
    parameter_count = sum(layer.A.size + layer.c.size for layer in layers)
    tangent = np.zeros((sample_count, np.asarray(inputs).shape[1], parameter_count))
    offset = 0
    for layer, (_, z, feature) in zip(layers, caches):
        width = feature.shape[1]
        output_rank = layer.A.shape[0]
        jacobians = np.einsum(
            "aw,xw,wi->xai",
            layer.A,
            relu_derivative(z),
            layer.W,
            optimize=True,
        ) / np.sqrt(width * tangent.shape[1])
        tangent = np.einsum("xai,xip->xap", jacobians, tangent, optimize=True)

        for a in range(output_rank):
            for mu in range(width):
                tangent[:, a, offset + a * width + mu] = feature[:, mu] / np.sqrt(
                    width
                )
        offset += layer.A.size
        for a in range(output_rank):
            tangent[:, a, offset + a] = 1.0
        offset += layer.c.size
    if tangent.shape != (sample_count, final_rank, parameter_count):
        raise AssertionError("tangent propagation produced an inconsistent shape")
    return np.einsum("xap,ybp->xyab", tangent, tangent, optimize=True)

