# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Unified molecular-dynamics entry point.

The force field is selected by the ``potential`` config (LJ / tojax-MLFF /
torch-MLFF). The potential is built directly with its parameters, so a single
force-field-agnostic ``MdState`` and driver serve every backend.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # for torch backends

import jax
import jax.numpy as jnp
import rich
import rich.logging
from nanoargs import NanoArgs
from pydantic import BaseModel

from kups.application.md.analysis import analyze_md_file
from kups.application.md.data import (
    MdParameters,
    MDParticles,
    MdRunConfig,
    MdState,
    MDSystems,
    md_state_from_ase,
)
from kups.application.md.simulation import IsMdState, make_md_propagator, run_md
from kups.application.potential.filter import POSITIONS_AND_CELL
from kups.application.simulations.potentials import (
    MaceConfig,
    PotentialConfig,
    TojaxPotentialConfig,
    UmaConfig,
)
from kups.core.data import Table
from kups.core.lens import identity_lens
from kups.core.neighborlist import UniversalNeighborlistParameters
from kups.core.typing import ParticleId, SystemId
from kups.core.utils.jax import key_chain

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler()],
)


class Config(BaseModel):
    """Top-level configuration for a molecular-dynamics run."""

    run: MdRunConfig
    md: MdParameters
    potential: PotentialConfig
    inp_files: tuple[str | Path, ...]


def run(config: Config) -> None:
    """Run a molecular-dynamics simulation for the configured force field."""
    seed = config.run.seed or time.time_ns()
    chain = key_chain(jax.random.key(seed))
    state_lens = identity_lens(MdState)
    potential, cutoff = config.potential.build(state_lens, POSITIONS_AND_CELL)
    propagator = make_md_propagator(state_lens, config.md.integrator, potential)

    mb_key = next(chain) if config.md.initialize_momenta else None
    all_particles: list[Table[ParticleId, MDParticles]] = []
    all_systems: list[Table[SystemId, MDSystems]] = []
    for inp_file in config.inp_files:
        logging.info(f"Loading structure from {inp_file}")
        particles_i, systems_i = md_state_from_ase(inp_file, config.md, key=mb_key)
        all_particles.append(particles_i)
        all_systems.append(systems_i)
    particles, systems = Table.union(all_particles, all_systems)

    base = 1 if isinstance(config.potential, TojaxPotentialConfig) else 2
    multiplier = 2.0 if isinstance(config.potential, MaceConfig | UmaConfig) else 1.0
    neighborlist_params = UniversalNeighborlistParameters.estimate(
        particles.data.system.counts, systems, cutoff, base=base, multiplier=multiplier
    )
    state = MdState(
        particles=particles,
        systems=systems,
        neighborlist_params=neighborlist_params,
        step=jnp.array([0]),
    )
    if jax.default_backend() == "tt":
        state = _f32_geometry(state)
    run_md(next(chain), propagator, state, config.run)


def _f32_geometry[State: IsMdState](state: State) -> State:
    """tt port: f32-normalize geometry at the device boundary (wayfinder #85).

    The tt device materializes f64 device tensors as bf16 (8-bit mantissa),
    which rounds the cell (21.04 -> 21.0, -0.57% density) and positions
    (0.125 A ulp at 21 A) and shifts the LJ NVE PE mean ~+1.4% — out of the
    kUPS-CI 10-sigma band. Casting the geometry (and all f64 state floats) to
    f32 at the boundary keeps the device arithmetic f32 (24-bit), matching the
    CPU reference within the band. Host-side, @no_jax_tracing init data
    (two-pattern rule); CPU/other backends keep f64 untouched.
    """
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if getattr(x, "dtype", None) == jnp.float64 else x,
        state,
    )


def main() -> None:
    """CLI entry point for molecular-dynamics simulations."""
    cli = NanoArgs(Config)
    config = cli.parse()
    rich.print(config)
    run(config)
    rich.print(analyze_md_file(config.run.out_file))


if __name__ == "__main__":
    main()
