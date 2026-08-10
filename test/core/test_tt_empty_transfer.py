# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""TT-only regression for empty host-to-device transfers (wayfinder #66).

The tt backend crashes with ``RuntimeError: Buffer pointers must not be null``
when an empty one-dimensional 64-bit numpy array is transferred host to
device. kUPS funnels such arrays through ``tt_safe_asarray`` and
``Index.zeros``, which create empty arrays on the device instead.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kups.core.data import Index
from kups.core.typing import SystemId
from kups.core.utils.jax import tt_safe_asarray


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_empty_1d_64bit_transfer_does_not_crash():
    for dtype in (np.int64, np.float64):
        out = tt_safe_asarray(np.zeros(0, dtype=dtype))
        assert out.shape == (0,)
        ref = jnp.asarray(np.zeros(1, dtype=dtype))
        assert out.dtype == ref.dtype


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires the TT backend")
def test_index_zeros_empty_does_not_crash():
    idx = Index.zeros(0)
    assert idx.indices.shape == (0,)
    assert idx.keys == (0,)
    labeled = Index.zeros(0, label=SystemId)
    assert labeled.indices.shape == (0,)
