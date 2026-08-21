# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Edge representations for molecular systems.

An [`Edges`][kups.core.neighborlist.edges.Edges] value encodes the
connectivity produced by a neighbor list (or built explicitly for bonded
terms). It is generic in its ``Degree`` so the same dataclass represents
pairs (``Degree=2``), angles (``Degree=3``), dihedrals (``Degree=4``), etc.
"""

from __future__ import annotations

import dataclasses
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
from kups.core.utils.jax import dataclass, pairwise_sum
from kups.core.utils.ops import take_col, use_mask_selection


@jax.custom_vjp
def _edge_dvec(pos: Array, idx: Array, abs_shifts: Array) -> Array:
    """Per-edge difference vectors from raw positions.

    ``dvec_e = pos[idx_e,1] - pos[idx_e,0] + abs_shifts_e`` with shape
    ``(n_edges, 1, 3)`` (the ``Degree-1`` axis, matching the plain path).
    ``idx`` may contain the OOB padding sentinel (gather clamps it, exactly
    like the plain Table gather).
    """

    return (pos[take_col(idx, 1)] - pos[take_col(idx, 0)])[:, None, :] + abs_shifts


def _edge_dvec_fwd(pos: Array, idx: Array, abs_shifts: Array) -> tuple[Array, tuple]:
    out = (pos[take_col(idx, 1)] - pos[take_col(idx, 0)])[:, None, :] + abs_shifts
    n_atoms = pos.shape[0]
    valid = (take_col(idx, 0) < n_atoms) & (take_col(idx, 1) < n_atoms)
    return out, (idx, valid, n_atoms)


def _edge_dvec_bwd(
    res: tuple, ct: Array
) -> tuple[Array, None, Array]:
    idx, valid, n_atoms = res
    ct2 = take_col(ct, 0, axis=-2)  # (n_edges, 3)
    # Mask only the position accumulation: OOB padding rows have no position
    # incidence, and a NaN cotangent there must not poison the matmul. The
    # shifts cotangent is the raw ct (dropped rows carry zero from the LJ
    # sum_over), and the outer absolute_shifts chain owns the shift/cell
    # semantics.
    ct_pos = jnp.where(valid[:, None], ct2, 0.0)
    signed = (
        (take_col(idx, 1)[:, None] == jnp.arange(n_atoms)).astype(jnp.float32)
        - (take_col(idx, 0)[:, None] == jnp.arange(n_atoms)).astype(jnp.float32)
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
        # Degree consistency: pairs/angles carry Degree = shifts + 1; the
        # degree-0 point-cloud path (EmptyNeighborList, init probes) carries
        # (n, 0) indices with (n, 0, 3) shifts — both empty until populated.
        if raw.ndim != 2 or not (
            raw.shape[1] == self.shifts.shape[1] + 1
            or (raw.shape[1] == 0 and self.shifts.shape[1] == 0)
        ):
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
        if jax.default_backend() == "tt" and self.degree == 2:
            # tt port (wayfinder #102/#152): the plain VJP of the position gather
            # lowers to a Tensix scatter kernel whose float accumulation is
            # f16-Dst (fp32_dest_acc_en=false) — per-atom forces come out ~26%
            # low (measured). _edge_dvec keeps the exact forward (energy
            # unchanged) and accumulates the position cotangent with a signed
            # one-hot incidence matmul (TF32-class, ~0.1% error). Extended to
            # all pair edges on tt (#152): MC NVT uses TriclinicFrame even for
            # diagonal cells; the OrthogonalFrame-only gate left guest-stress
            # off-diagonals on the corrupt gather/scatter path.
            return _edge_dvec(particles.data.positions, self.indices.indices, shifts)
        pos = particles[self.indices].positions
        if use_mask_selection():
            # tt port (wayfinder #107): keep-dim static slices of the
            # cap-height edge tables lower to full-tensor ttnn.slice CBs;
            # take lowers to ttir.gather (DRAM-pipelined).
            return jnp.take(pos, jnp.array([1]), axis=-2) - jnp.take(
                pos, jnp.array([0]), axis=-2
            ) + shifts
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
        sys_idx = particles[self.indices._take_col(0)].system
        if jax.default_backend() == "tt":
            # tt port (wayfinder #103): every gather on tt (Table or raw-array,
            # ttir.embedding or ttir.gather) bf16-casts its f32 operand, so
            # gathering the cell (21.04 -> 21.0) freezes the per-edge shifts'
            # length dependence and the virial cell term h^T·∂U/∂h vanishes
            # (canonical pressure 66% low). Select the per-edge frame params
            # with a masked jnp.where (exact on tt) over the small system
            # axis, reduced with the pairwise-add tree (no reduce kernel).
            frame = lattice.data.frame
            leaves = {}
            for f in dataclasses.fields(frame):
                v = getattr(frame, f.name)
                if isinstance(v, Array) and v.ndim > 0:
                    n_sys = v.shape[0]
                    sel = sys_idx.indices[None, :, None] == jnp.arange(n_sys)[:, None, None]
                    rows = jnp.where(sel, v[None, :, :], 0.0)  # (n_sys, n_edges, C)
                    leaves[f.name] = pairwise_sum(rows)[:, None]  # (n_edges, 1, C)
                else:
                    leaves[f.name] = v
            cells_frame = type(frame)(**leaves)
            return cells_frame.to_real(self.shifts)
        cells = lattice[sys_idx][:, None]
        return cells.frame.to_real(self.shifts)

    @property
    def degree(self) -> int:
        return self.indices.shape[-1]

    @override
    def __len__(self) -> int:
        return self.indices.shape[0]
