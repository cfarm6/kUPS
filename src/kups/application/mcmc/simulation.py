# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generic simulation loop for rigid-body MCMC simulations."""

from __future__ import annotations

import logging

import jax
from jax import Array

from kups.application.mcmc.data import RunConfig
from kups.application.mcmc.logging import IsMCMCState, MCMCLoggedData
from kups.application.utils.propagate import (
    make_cycle_function,
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.propagator import Propagator
from kups.core.storage import HDF5StorageWriter
from kups.core.utils.jax import jit, key_chain
from kups.core.utils.ops import use_host_rng_draws
from kups.mcmc.rng import inject_host_rng_draws


def run_mcmc[State: IsMCMCState](
    key: Array,
    propagator: Propagator[State],
    state: State,
    config: RunConfig,
    logged_data: MCMCLoggedData[State],
) -> State:
    """Run a µVT MCMC simulation with warmup and production phases.

    Args:
        key: JAX PRNG key.
        propagator: Propagator, e.g. from :func:`~kups.application.simulations.mcmc_rigid.make_propagator`.
        state: Initial simulation state.
        config: Run configuration.
        logged_data: Logging configuration with host/adsorbate split.

    Returns:
        Final simulation state after production run.
    """

    @jit
    def _postfix_jit(state: State):
        return state.groups.num_occupied

    def postfix(state: State):
        return {"Loading": f"{_postfix_jit(state)}"}

    chain = key_chain(key)
    cycle_fn = make_cycle_function(propagator)
    if use_host_rng_draws():
        # tt port (wayfinder #92): host-side per-cycle RNG substitution. The
        # MC cycle loop is a python loop over jitted per-cycle calls, so each
        # cycle's proposal draws are precomputed on the host (CPU, jax x64
        # semantics) and carried in the state -- the donated cycle input --
        # mirroring MD's chi2 substitution (wayfinder #44, md/simulation.py).
        # Only states carrying `rng_draws` are injected; others (e.g. Widom)
        # keep their traced device draws.
        draw_seed = config.seed if config.seed is not None else int(
            jax.random.key_data(key)[0]
        )
        _draw_chain = key_chain(jax.random.key(draw_seed))
        _cycle_fn = cycle_fn

        def cycle_fn(key: Array, state: State) -> Result[State, State]:
            return _cycle_fn(key, inject_host_rng_draws(state, config, next(_draw_chain)))
    logging.info("Warming up (%d cycles)...", config.num_warmup_cycles)
    state = run_warmup_cycles(next(chain), cycle_fn, state, config.num_warmup_cycles)
    logging.info("Production run (%d cycles)...", config.num_cycles)
    logger = CompositeLogger[State](
        HDF5StorageWriter(config.out_file, logged_data, state, config.num_cycles),
        TqdmLogger(config.num_cycles, postfix=postfix),
    )
    state = run_simulation_cycles(
        next(chain), cycle_fn, state, config.num_cycles, logger
    )
    logging.info("Done.")
    return state
