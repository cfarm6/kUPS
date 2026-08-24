# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from kups.application.md.data import (
    MDParticles,
    MdRunConfig,
    MDSystems,
)
from kups.application.md.integrators import make_md_step_from_state
from kups.application.md.logging import MDLoggedData
from kups.application.utils.propagate import (
    make_cycle_function,
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.cell import AnyPeriodicity, Cell
from kups.core.data import Table
from kups.core.lens import Lens, lens
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.potential import (
    EMPTY,
    CachedPotential,
    EmptyType,
    MappedPotential,
    MappedPotentialInput,
    Potential,
    PotentialAsPropagator,
    PotentialOut,
)
from kups.core.propagator import (
    LoopPropagator,
    Propagator,
    ResetOnErrorPropagator,
    SequentialPropagator,
    step_counter_propagator,
)
from kups.core.storage import HDF5StorageWriter
from kups.core.typing import IsState, ParticleId, SystemId
from kups.core.utils.jax import key_chain
from kups.md.integrators import Integrator


class IsMdGradients(Protocol):
    """Protocol for MD gradient outputs.

    Attributes:
        positions: Position gradients as Table[ParticleId, Array].
        cell: Cell gradients as Table[SystemId, Cell].
    """

    @property
    def positions(self) -> Table[ParticleId, Array]: ...
    @property
    def cell(self) -> Table[SystemId, Cell[AnyPeriodicity]]: ...


class IsMdState(IsState[MDParticles, MDSystems], Protocol):
    """Protocol for the full MD simulation state.

    Attributes:
        particles: Per-particle data (positions, momenta, forces, etc.).
        systems: Per-system data (cell, thermostat parameters, etc.).
        step: Current simulation step counter.
    """

    @property
    def step(self) -> Array: ...


def potential_map(
    input: MappedPotentialInput[IsMdState, IsMdGradients, Any],
) -> PotentialOut[tuple[Array, Cell[AnyPeriodicity]], EmptyType]:
    """Map the full potential output to the specific gradients needed for MD."""
    return PotentialOut(
        input.potential_out.total_energies,
        (
            input.potential_out.gradients.positions.data,
            input.potential_out.gradients.cell.data,
        ),
        EMPTY,
    )


def make_md_propagator[State: IsMdState, Grad: IsMdGradients](
    state_lens: Lens[State, State],
    integrator: Integrator,
    potential: Potential[State, Grad, EmptyType, Any],
) -> Propagator[State]:
    """Build a single MD propagator step with error recovery and step counting.

    Args:
        state_lens: Lens focusing on the MD sub-state within the full state.
        integrator: Integration algorithm for equations of motion.
        potential: Potential energy function providing forces and gradients.

    Returns:
        Propagator that advances the state by one MD step.
    """
    mapped_potential = MappedPotential(potential, potential_map)
    derivative_computation = PotentialAsPropagator(
        CachedPotential(
            mapped_potential,
            lens(
                lambda x: PotentialOut(
                    x.systems.map_data(lambda x: x.potential_energy),
                    (
                        x.particles.data.position_gradients,
                        x.systems.data.cell_gradients,
                    ),
                    EMPTY,
                )
            ),
            # pyrefly: ignore [bad-argument-type]
            lambda x: PotentialOut(
                x.systems.index,  # type: ignore
                (x.particles.data.system, x.systems.index),
                EMPTY,
            ),
        )
    )
    md_propagator = make_md_step_from_state(
        state_lens, derivative_computation, integrator
    )
    step_count_propagator = step_counter_propagator(state_lens.focus(lambda x: x.step))
    propagator = ResetOnErrorPropagator(
        SequentialPropagator((md_propagator, step_count_propagator))
    )
    return propagator


def inject_host_chi2_draws[State: IsMdState](
    state: State, config: MdRunConfig, key: Array
) -> State:
    """tt port: host-side chi2 substitution (wayfinder #44).

    Precompute the per-step chisquare draws (df = 3N-4 per system) on the host
    and carry them in the state — the donated cycle input — so csvr/csvr_npt
    trace only the counted step-loop while (no nested gamma whiles). Called on
    the tt backend only; ``CSVRStep`` reads ``state.chi2_draws[step]`` when
    present and otherwise falls back to the traced draw.
    """
    counts = state.particles.data.system.counts.data
    # Clamp so the host draw stays defined for degenerate systems (df <= 0);
    # CSVRStep's where guard masks those to 0.0 anyway.
    dof_minus_one = jnp.maximum(counts * 3 - 4, 0.0)
    total_steps = config.num_warmup_steps + config.num_steps
    seed = int(jax.random.key_data(key)[0])
    _cpu = jax.devices("cpu")[0]
    with jax.default_device(_cpu):
        draws = jax.random.chisquare(
            jax.random.key(seed), dof_minus_one, shape=(total_steps, len(counts))
        )
    return dataclasses.replace(state, chi2_draws=jax.device_put(draws))


def run_md[State: IsMdState](
    key: Array, propagator: Propagator[State], state: State, config: MdRunConfig
) -> State:
    """Run a full MD simulation with warmup and production phases.

    Args:
        key: JAX PRNG key.
        propagator: MD propagator produced by `make_md_propagator`.
        state: Initial simulation state.
        config: Run configuration (steps, output file, seed).

    Returns:
        Final simulation state after production run.
    """
    chain = key_chain(key)
    if jax.default_backend() == "tt":
        state = inject_host_chi2_draws(state, config, key)
    # Check assertions after each step during warmup cycles
    cycle_fn = make_cycle_function(propagator)
    logging.info("Warmup")
    state = run_warmup_cycles(next(chain), cycle_fn, state, config.num_warmup_steps)

    logging.info("Starting MD simulation")
    # Each cycle fuses block_size steps into one dispatch and saves its final frame, so the
    # trajectory keeps num_steps // block_size frames (block_size=1 = one step per cycle).
    cycle_fn = make_cycle_function(LoopPropagator(propagator, config.block_size))
    num_cycles = config.num_steps // config.block_size
    writer = HDF5StorageWriter(config.out_file, MDLoggedData(), state, num_cycles)
    logger = CompositeLogger(
        TqdmLogger(num_cycles),
        writer,
    )
    state = run_simulation_cycles(
        next(chain),
        cycle_fn,
        state,
        num_cycles,
        logger,
        batch_size=64 if jax.default_backend() == "tt" else 1,
        batch_capture=writer.capture_batch,
    )
    return state
