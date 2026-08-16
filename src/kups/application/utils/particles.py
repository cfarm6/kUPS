# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared particle data structures and ASE loading utilities."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Callable

import ase
import ase.io
import jax
import numpy as np
import jax.numpy as jnp
from jax import Array

from kups.core.cell import (
    AnyPeriodicity,
    Cell,
    OrthogonalFrame,
    TriclinicFrame,
    to_lower_triangular,
)
from kups.core.data import Index, Table
from kups.core.typing import ExclusionId, InclusionId, Label, ParticleId, SystemId
from kups.core.utils.jax import dataclass, tt_safe_asarray


@dataclass
class Particles:
    """Particle state shared across simulation types.

    Attributes:
        positions: Cartesian coordinates in the lower-triangular frame, shape (n_atoms, 3).
        masses: Atomic masses (amu), shape (n_atoms,).
        atomic_numbers: Atomic numbers, shape (n_atoms,).
        charges: Partial charges, shape (n_atoms,).
        labels: Per-atom string labels.
        system: Index mapping each particle to a system.
    """

    positions: Array
    masses: Array
    atomic_numbers: Array
    charges: Array
    labels: Index[Label]
    system: Index[SystemId]

    @property
    def inclusion(self) -> Index[InclusionId]:
        """System index re-labeled as InclusionId."""
        return Index(tuple(map(InclusionId, self.system.keys)), self.system.indices)


def default_exclusion(n: int) -> Index[ExclusionId]:
    """Build a default per-particle exclusion index (each atom excludes itself).

    Args:
        n: Number of particles.

    Returns:
        Index mapping each particle to a unique ExclusionId.
    """
    return Index.integer(jnp.arange(n), n=n, label=ExclusionId)


def particles_from_ase(
    atoms: ase.Atoms | str | Path,
) -> tuple[
    Table[ParticleId, Particles], Cell[AnyPeriodicity], Callable[[Array], Array]
]:
    """Build particle data and cell from an ASE Atoms object or file path.

    Results are cached when ``atoms`` is a file path.

    Args:
        atoms: ASE Atoms object, or a file path (str/Path) readable by
            ``ase.io.read``.

    Returns:
        Tuple of (particles, cell, uc_transform) where uc_transform
        rotates Cartesian positions into the lower-triangular frame.
    """
    if isinstance(atoms, (str, Path)):
        return _particles_from_path(atoms)
    return _particles_from_atoms(atoms)


@cache
def _particles_from_path(
    path: str | Path,
) -> tuple[
    Table[ParticleId, Particles], Cell[AnyPeriodicity], Callable[[Array], Array]
]:
    """Read an ASE-readable file and build cached particle data and cell."""
    try:
        atoms = next(ase.io.iread(path, index=-1, store_tags=True))
    except TypeError:
        # Input-robustness fallback (wayfinder #105): ASE xyz/traj readers
        # reject `store_tags` (CIF-only keyword); retry without it so
        # non-CIF formats — required for non-periodic cells, which CIF cannot
        # carry (`pbc=False`) — load through the same path.
        atoms = next(ase.io.iread(path, index=-1))
    return _particles_from_atoms(atoms)


def _particles_from_atoms(
    atoms: ase.Atoms,
) -> tuple[
    Table[ParticleId, Particles], Cell[AnyPeriodicity], Callable[[Array], Array]
]:
    """Build particle data and cell from an ASE Atoms object."""
    L_np = np.asarray(atoms.cell.array)
    L, uc_transform = to_lower_triangular(L_np)  # host einsum: f64-exact
    pbc = (bool(atoms.pbc[0]), bool(atoms.pbc[1]), bool(atoms.pbc[2]))
    frame_cls = TriclinicFrame
    if jax.default_backend() == "tt":
        # tt port (wayfinder #102): on diagonal cells the TriclinicFrame
        # wrap/to_real/to_fractional lower to Tensix FPU matmuls (TF32-class
        # precision — ~1e-3 fractional error) whose error flips the periodic
        # fold boundary (position jumps of one box length every step) and
        # corrupts the trajectory. OrthogonalFrame keeps the same ops on the
        # exact SFPU elementwise path (r/L, r*L). Physics-identical for a
        # diagonal cell; other backends keep TriclinicFrame untouched.
        _L = np.asarray(L)
        if np.allclose(_L, np.diag(np.diag(_L))):
            frame_cls = OrthogonalFrame
    if jax.default_backend() == "tt":
        # f32 host arrays: the tt device materializes f64 device tensors as bf16
        # (21.04 -> 21.0 — #85 family) and jnp.diagonal on the device returns the
        # diagonal bf16-quantized (wayfinder #102); casting after creation cannot
        # recover the lost mantissa. Construct the frame from host f32 numpy so
        # the transfer is exact.
        L32 = np.asarray(L, dtype=np.float32)
        if frame_cls is OrthogonalFrame:
            cell = Cell.from_pbc(OrthogonalFrame(jnp.asarray(np.diag(L32))), pbc)
        else:
            cell = Cell.from_pbc(frame_cls.from_matrix(L32), pbc)
    else:
        # CPU/other backends keep the f64 host cell so the reference
        # observables stay exact (volume 21.04^3 = 9314.020864, SEM ~0) —
        # wayfinder #47: the unconditional f32 cast (introduced with #102)
        # leaked the tt boundary downcast into the CPU reference (volume f32
        # 9314.02246, f32-reduction SEM 3.27e-5) and failed the 10-SEM band
        # on a 1.7e-7-relative representation difference.
        cell = Cell.from_pbc(frame_cls.from_matrix(L), pbc)
    positions = tt_safe_asarray(uc_transform(np.asarray(atoms.positions)))
    masses = tt_safe_asarray(atoms.get_masses())
    atomic_numbers = tt_safe_asarray(atoms.get_atomic_numbers())
    n_atoms = len(masses)
    charges = tt_safe_asarray(
        atoms.info.get(
            "_atom_type_partial_charge",
            atoms.info.get("_atom_site_charge", jnp.zeros((len(positions),))),
        )
    )
    labels = list(
        map(Label, atoms.info.get("_atom_site_label", atoms.get_chemical_symbols()))
    )
    particles = Table.arange(
        Particles(
            positions=positions,
            masses=masses,
            atomic_numbers=atomic_numbers,
            charges=charges,
            labels=Index.new(labels),
            system=Index.integer(jnp.zeros(n_atoms, dtype=int), label=SystemId),
        ),
        label=ParticleId,
    )
    return particles, cell, uc_transform
