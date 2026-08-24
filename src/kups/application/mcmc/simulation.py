# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Generic simulation loop for rigid-body MCMC simulations."""

from __future__ import annotations

import dataclasses
import logging
import jax
import jax.numpy as jnp

from kups.application.mcmc.data import RunConfig
from kups.application.utils.propagate import (
    make_cycle_function,
    propagate_and_fix,
    run_simulation_cycles,
    run_warmup_cycles,
)
from kups.core.logging import CompositeLogger, TqdmLogger
from kups.core.propagator import Propagator
from kups.core.storage import HDF5StorageWriter
from kups.core.utils.jax import jit, key_chain
from kups.core.utils.ops import use_host_rng_draws
from kups.mcmc.rng import draw_rng_bundle, inject_host_rng_draws


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
    batch_size = 64 if jax.default_backend() == "tt" else 1
    batch_runner = None
    if use_host_rng_draws():
        # tt port (wayfinder #92): host-side per-cycle RNG substitution.
        draw_seed = config.seed if config.seed is not None else int(
            jax.random.key_data(key)[0]
        )
        _draw_chain = key_chain(jax.random.key(draw_seed))
        _cycle_fn = cycle_fn

        def cycle_fn(key: Array, state: State) -> Result[State, State]:
            return _cycle_fn(key, inject_host_rng_draws(state, config, next(_draw_chain)))

        def batch_runner(keys, current, capture):
            max_steps = current.groups.data.system.max_count
            if max_steps is None:
                max_steps = int(
                    jnp.maximum(
                        current.groups.data.system.counts.data.max(),
                        config.min_cycle_length,
                    )
                )
            max_steps = max(max_steps, config.min_cycle_length)
            bundles = [
                draw_rng_bundle(next(_draw_chain), current, max_steps)
                for _ in range(keys.shape[0])
            ]
            rng_batch = jax.tree.map(
                lambda *leaves: jnp.stack(leaves), *bundles
            )

            def step(carry, inputs):
                cycle_key, rng_draws = inputs
                injected = dataclasses.replace(
                    carry,
                    rng_step=jnp.zeros((1,), dtype=jnp.int32),
                    rng_draws=rng_draws,
                )
                result = _cycle_fn(cycle_key, injected)
                next_state = result.value
                return next_state, (capture(next_state), result.all_assertions_pass)

            next_state, (data, passes) = jax.lax.scan(
                step, current, (keys, jax.device_put(rng_batch))
            )
            return next_state, data, passes

    logging.info("Warming up (%d cycles)...", config.num_warmup_cycles)
    state = run_warmup_cycles(next(chain), cycle_fn, state, config.num_warmup_cycles)
    logging.info("Production run (%d cycles)...", config.num_cycles)
    writer = HDF5StorageWriter(config.out_file, logged_data, state, config.num_cycles)
    logger = CompositeLogger[State](
        writer,
        TqdmLogger(config.num_cycles, postfix=postfix),
    )
    state = run_simulation_cycles(
        next(chain),
        cycle_fn,
        state,
        config.num_cycles,
        logger,
        batch_size=batch_size,
        batch_capture=writer.capture_batch,
        batch_runner=batch_runner,
    )
    logging.info("Done.")
    return state
