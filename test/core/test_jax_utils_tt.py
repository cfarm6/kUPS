# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""TT-only checks for the isin fallback (wayfinder #72)."""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from kups.core.data.index import Index
from kups.core.utils.jax import isin


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_isin_membership_on_tt():
    # tt scatter on bool tables silently no-ops for single-element index sets
    # (wayfinder #72); the fallback must return true membership.
    npt.assert_array_equal(
        jax.jit(isin, static_argnums=2)(
            jnp.array([0, 0, 0], jnp.int32), jnp.array([0], jnp.int32), 3
        ),
        jnp.array([True, True, True]),
    )


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_isin_sentinel_semantics_on_tt():
    # OOB sentinel (== max_item) is never a member, even when b holds it.
    npt.assert_array_equal(
        jax.jit(isin, static_argnums=2)(
            jnp.array([0, 3], jnp.int32), jnp.array([0, 3], jnp.int32), 3
        ),
        jnp.array([True, False]),
    )


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_where_flat_per_species_on_tt():
    # Multi-species placement selects each species separately (the #72 failure
    # shape: single-element scatter into a two-species table).
    idx = Index((0, 1), jnp.array([0, 1, 0, 1, 0, 1], jnp.int32))
    tgt = Index((0, 1), jnp.array([1], jnp.int32))
    npt.assert_array_equal(idx.where_flat(tgt), jnp.array([1, 3, 5]))
