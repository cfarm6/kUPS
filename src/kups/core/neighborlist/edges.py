# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Edge representations for molecular systems.

An [`Edges`][kups.core.neighborlist.edges.Edges] value encodes the
connectivity produced by a neighbor list (or built explicitly for bonded
terms). It is generic in its ``Degree`` so the same dataclass represents
pairs (``Degree=2``), angles (``Degree=3``), dihedrals (``Degree=4``), etc.
"""

from __future__ import annotations

from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.cell import AnyPeriodicity, OrthogonalFrame
from kups.core.data import Index, Sliceable, Table
from kups.core.typing import (
    HasCell,
    HasPositionsAndSystemIndex,
    ParticleId,
    SystemId,
)
from kups.core.utils.jax import dataclass


@jax.custom_vjp
def _edge_dvec(pos: Array, idx: Array, abs_shifts: Array) -> Array:
    """Per-edge difference vectors from raw positions.

    ``dvec_e = pos[idx_e,1] - pos[idx_e,0] + abs_shifts_e`` with shape
    ``(n_edges, 1, 3)`` (the ``Degree-1`` axis, matching the plain path).
    ``idx`` may contain the OOB padding sentinel (gather clamps it, exactly
    like the plain Table gather).
    """

    return (pos[idx[:, 1]] - pos[idx[:, 0]])[:, None, :] + abs_shifts


def _edge_dvec_fwd(pos: Array, idx: Array, abs_shifts: Array) -> tuple[Array, tuple]:
    out = (pos[idx[:, 1]] - pos[idx[:, 0]])[:, None, :] + abs_shifts
    n_atoms = pos.shape[0]
    valid = (idx[:, 0] < n_atoms) & (idx[:, 1] < n_atoms)
    return out, (idx, valid, n_atoms)


def _edge_dvec_bwd(
    res: tuple, ct: Array
) -> tuple[Array, None, Array]:
    idx, valid, n_atoms = res
    ct2 = ct[:, 0]  # (n_edges, 3)
    # Mask only the position accumulation: OOB padding rows have no position
    # incidence, and a NaN cotangent there must not poison the matmul. The
    # shifts cotangent is the raw ct (dropped rows carry zero from the LJ
    # sum_over), and the outer absolute_shifts chain owns the shift/cell
    # semantics.
    ct_pos = jnp.where(valid[:, None], ct2, 0.0)
    signed = (
        (idx[:, 1, None] == jnp.arange(n_atoms)).astype(jnp.float32)
        - (idx[:, 0, None] == jnp.arange(n_atoms)).astype(jnp.float32)
    )
    pos_grad = signed.T @ ct_pos  # (n_atoms, 3)
    return pos_grad, None, ct


_edge_dvec.defvjp(_edge_dvec_fwd, _edge_dvec_bwd)


@dataclass
class Edges[Degree: int](Sliceable):
    """Represents edges (connections) between particles in a molecular system.

    The degree is purely for type checking and does not affect runtime behavior:
    ``Degree=2`` stores pair edges, ``Degree=3`` angle triples, etc. The shifts
    array has shape ``(n_edges, Degree-1, 3)`` and holds *fractional* image
    shifts; [`absolute_shifts`][kups.core.neighborlist.edges.Edges.absolute_shifts]
    converts them to Cartesian vectors via the cell frame.
    """

    # The degree is purely for type checking and does not affect runtime behavior
    indices: Index[ParticleId]  # (n_edges, Degree)
    shifts: Array  # (n_edges, Degree - 1, 3)

    def __post_init__(self) -> None:
        # Resolve the underlying array for validation
        raw = self.indices.indices if isinstance(self.indices, Index) else self.indices
        if raw.ndim != 2 or raw.shape[1] != self.shifts.shape[1] + 1:
            raise ValueError(
                "Edges indices and shifts must be degree-consistent: "
                f"got indices {raw.shape} and shifts {self.shifts.shape}."
            )

    def difference_vectors(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
    ) -> Array:
        """Compute difference vectors between connected particles.

        For each edge, computes the vector from the first particle to each
        subsequent particle, accounting for periodic boundary conditions.

        Args:
            particles: Particle positions with system index information.
            systems: System data with cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing difference vectors.
        """
        shifts = self.absolute_shifts(particles, systems)
        if (
            jax.default_backend() == "tt"
            and self.degree == 2
            and isinstance(systems.data.cell.frame, OrthogonalFrame)
        ):
            # tt port (wayfinder #102): the plain VJP of the position gather
            # lowers to a Tensix scatter kernel whose float accumulation is
            # f16-Dst (fp32_dest_acc_en=false) — per-atom forces come out ~26%
            # low (measured). _edge_dvec keeps the exact forward (energy
            # unchanged) and accumulates the position cotangent with a signed
            # one-hot incidence matmul (TF32-class, ~0.1% error). Other
            # backends keep the plain autograd; non-orthogonal tt frames keep
            # the plain path (already TF32-limited there).
            return _edge_dvec(particles.data.positions, self.indices.indices, shifts)
        pos = particles[self.indices].positions
        return pos[:, 1:] - pos[:, :1] + shifts

    def absolute_shifts(
        self,
        particles: Table[ParticleId, HasPositionsAndSystemIndex],
        systems: Table[SystemId, HasCell[AnyPeriodicity]],
    ) -> Array:
        """Compute absolute shift vectors for all particles in each edge.

        Converts relative shifts to absolute Cartesian shift vectors.

        Args:
            particles: Particle data with system index information.
            systems: System data with cell for periodic boundary conditions.

        Returns:
            Array of shape `(n_edges, Degree-1, 3)` containing absolute shift vectors.
        """
        lattice = systems.map_data(lambda x: x.cell.materialize())
        cells = lattice[particles[self.indices[:, 0]].system][:, None]
        return cells.frame.to_real(self.shifts)

    @property
    def degree(self) -> int:
        return self.indices.shape[-1]

    @override
    def __len__(self) -> int:
        return self.indices.shape[0]
