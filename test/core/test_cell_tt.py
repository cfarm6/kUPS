# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""TT-only checks for cell setup."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.cell import to_lower_triangular


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_to_lower_triangular_uses_host_qr_on_tt():
    vecs = jnp.array([[1.0, 2.0, 0.3], [0.2, 3.0, 0.1], [0.5, 0.4, 4.0]])
    lower, mapper = to_lower_triangular(vecs)
    rotation = jnp.stack([mapper(row) for row in jnp.eye(3)])

    npt.assert_allclose(lower @ rotation.T, vecs, atol=5e-3)
