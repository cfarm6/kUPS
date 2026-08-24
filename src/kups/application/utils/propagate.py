# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Shared propagation utilities for simulation loops.

Provides warmup, sampling, and data-parallelism helpers used across
MD, MCMC, and relaxation application modules.
"""

import logging
from typing import Any, Callable, Protocol

import jax
import tqdm
from jax import Array

from kups.core.logging import Logger
from kups.core.propagator import (
    Propagator,
    propagate_and_fix,
    propagator_with_assertions,
)
from kups.core.result import Result, as_result_function
from kups.core.utils.jax import jit, key_chain

__all__ = [
    "propagate_and_fix",
    "propagator_with_assertions",
    "make_cycle_function",
]


class CycleFunction[State](Protocol):
    def __call__(self, key: Array, state: State, /) -> Result[State, State]: ...

type BatchRunner[State] = Callable[
    [Array, State, Callable[[State], Any]], tuple[State, Any, Array]
]


def make_cycle_function[State](propagator: Propagator[State]) -> CycleFunction[State]:
    """JIT a propagator into a reusable per-cycle function with state donation.

    Pass the result as ``cycle_fn`` to both :func:`run_warmup_cycles` and
    :func:`run_simulation_cycles` so a single traced-and-compiled program is shared
    across the warmup and sampling phases. For blocked stepping, compose the propagator
    with :class:`~kups.core.propagator.LoopPropagator` before passing it in.

    Args:
        propagator: Step propagator to compile.

    Returns:
        A jitted ``(key, state) -> Result`` cycle function.
    """
    return jit(as_result_function(propagator), donate_argnums=(1,))


def run_warmup_cycles[State](
    key: Array, cycle_fn: CycleFunction[State], state: State, num_cycles: int
) -> State:
    """Run warmup propagation cycles without logging.

    Args:
        key: JAX PRNG key.
        cycle_fn: Compiled per-cycle function from :func:`make_cycle_function`.
        state: Initial simulation state.
        num_cycles: Number of warmup steps.

    Returns:
        State after warmup.
    """
    chain = key_chain(key)
    for _ in tqdm.trange(num_cycles):
        state = propagate_and_fix(cycle_fn, next(chain), state)
    return state

def run_simulation_cycles[State](
    key: Array,
    cycle_fn: CycleFunction[State],
    state: State,
    num_cycles: int,
    logger: Logger[State],
    *,
    convergence_fn: Callable[[State], bool] | None = None,
    batch_size: int = 1,
    batch_capture: Callable[[State], Any] | None = None,
    batch_runner: BatchRunner[State] | None = None,
) -> State:
    """Run simulation steps with logging and optional early stopping."""
    chain = key_chain(key)
    with logger:
        if batch_size <= 1 or convergence_fn is not None or batch_capture is None:
            for i in range(num_cycles):
                state = propagate_and_fix(cycle_fn, next(chain), state)
                logger.log(state, i)
                if convergence_fn is not None and convergence_fn(state):
                    logging.info("Converged at step %d", i + 1)
                    break
            return state

        capture = batch_capture
        assert capture is not None

        def default_batch_runner(
            keys: Array, current: State, capture: Callable[[State], Any]
        ) -> tuple[State, Any, Array]:
            def step(carry: State, cycle_key: Array) -> tuple[State, tuple[Any, Array]]:
                result = cycle_fn(cycle_key, carry)
                next_state = result.value
                return next_state, (capture(next_state), result.all_assertions_pass)

            next_state, (data, passes) = jax.lax.scan(step, current, keys)
            return next_state, data, passes

        runner = batch_runner or default_batch_runner
        if num_cycles == 0:
            return state
        state = propagate_and_fix(cycle_fn, next(chain), state)
        logger.log(state, 0)
        for start in range(1, num_cycles, batch_size):
            count = min(batch_size, num_cycles - start)
            keys = jax.numpy.stack([next(chain) for _ in range(count)])
            state, data, passes = runner(keys, state, capture)
            if not bool(jax.numpy.all(passes)):
                raise RuntimeError("Runtime assertion failed in batched simulation")
            logger.log_batch(data, start)
        return state


