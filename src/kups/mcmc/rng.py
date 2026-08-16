# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""tt port: host-side MC proposal RNG substitution (wayfinder #92).

The tt backend's f64 RNG pipeline is broken under ``jax_enable_x64``
(``jax.random.uniform`` -> all-inf, ``normal`` -> all-nan, eager and jitted;
measured 2026-08-16 on the current stack, see #91/#92). The MCMC cycle's
jitted proposal draws consume device f64 RNG, so every proposal is garbage
on tt. This module precomputes the per-cycle draws on the host (jax CPU,
x64 semantics, deterministic from the run seed) and carries them in the
state -- the donated cycle input -- mirroring the MD chi2 substitution
(``md/simulation.py::inject_host_chi2_draws``, wayfinder #44).

Per-cycle flow (application/mcmc/simulation.py::run_mcmc, tt only):

    draw_chain = key_chain(jax.random.key(draw_seed))
    def cycle_fn(key, state):            # wrapper around the donated jit
        state = inject_host_rng_draws(state, config, next(draw_chain))
        return _inner(key, state)

Inside the jit, every proposal RNG site reads ``state.rng_draws.<leaf>[i]``
where ``i = state.rng_step[0]`` (incremented once per LoopPropagator
iteration by :class:`RngStepCounter`). Distribution families match the
device draws exactly (identical laws; independent stream, so the TT
trajectory differs from CPU while remaining statistically equivalent --
sufficient for the kUPS-CI 10-SEM band). ``jax.random.bits`` (pure u32,
works on tt) is left as-is.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jax import Array

from kups.core.utils.jax import dataclass, key_chain
from kups.core.utils.ops import use_host_rng_draws


@dataclass
class MCMCRngDraws:
    """Per-cycle host-drawn proposal RNG values (tt backend only).

    Every leaf carries the per-step dimension first, ``(n_steps, ...)``,
    where ``n_steps`` is the cycle's LoopPropagator iteration count
    (``max(counts.max(), min_cycle_length)``), so the jit body can slice
    ``leaf[state.rng_step[0]]`` per iteration. Shapes follow the proposal
    sites in ``mcmc/moves.py`` / ``core/propagator.py``.

    Attributes:
        which_uniform: Move-type selection, shared across systems.
        exchange_uniform: Insert-vs-delete selection, shared across systems.
        translation: Normal offsets for group translation.
        rotation_u: Shoemake SO(3) uniforms for rotation.
        reinsertion_u: Shoemake SO(3) uniforms for reinsertion.
        reinsertion_offset: Fractional offsets for reinsertion.
        insert_motif_uniform: Motif selection for insertion.
        insert_offset: Fractional offsets for insertion.
        insert_u: Shoemake SO(3) uniforms for insertion.
        delete_motif_uniform: Motif selection for deletion.
        accept: Metropolis acceptance uniforms (per system).
    """

    which_uniform: Array  # (n_steps,)
    exchange_uniform: Array  # (n_steps,)
    translation: Array  # (n_steps, n_sys, 3)
    rotation_u: Array  # (n_steps, n_sys, 3)
    reinsertion_u: Array  # (n_steps, n_sys, 3)
    reinsertion_offset: Array  # (n_steps, n_sys, 3)
    insert_motif_uniform: Array  # (n_steps, n_sys)
    insert_offset: Array  # (n_steps, n_sys, 3)
    insert_u: Array  # (n_steps, n_sys, 3)
    delete_motif_uniform: Array  # (n_steps, n_sys)
    accept: Array  # (n_steps, n_sys)


def draw_rng_bundle(key: Array, state: object, n_steps: int) -> MCMCRngDraws:
    """Draw one cycle's proposal RNG bundle on the host (jax CPU, x64).

    ``key`` advances per cycle from the run seed chain, so the TT run is
    deterministic. The per-cycle jit body must not run any of these draws on
    the tt backend -- every call here executes under ``default_device(cpu)``.

    Args:
        key: Per-cycle PRNG key (CPU array from the draw chain).
        state: Simulation state carrying the MCMC arrays.
        n_steps: LoopPropagator iteration count for this cycle.

    Returns:
        The bundle with ``(n_steps, ...)`` leaves, on CPU.
    """
    n_sys = state.groups.data.system.counts.shape[0]
    chain = key_chain(key)

    def u(*shape) -> Array:
        return jax.random.uniform(next(chain), shape)

    def n(*shape) -> Array:
        return jax.random.normal(next(chain), shape)

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        return MCMCRngDraws(
            which_uniform=u(n_steps),
            exchange_uniform=u(n_steps),
            translation=n(n_steps, n_sys, 3),
            rotation_u=u(n_steps, n_sys, 3),
            reinsertion_u=u(n_steps, n_sys, 3),
            reinsertion_offset=u(n_steps, n_sys, 3),
            insert_motif_uniform=u(n_steps, n_sys),
            insert_offset=u(n_steps, n_sys, 3),
            insert_u=u(n_steps, n_sys, 3),
            delete_motif_uniform=u(n_steps, n_sys),
            accept=u(n_steps, n_sys),
        )


@dataclass
class RngStepCounter:
    """Increment ``state.rng_step`` once per MCMC step (tt backend only).

    Composed *after* the MCMC propagator so each step's draws are read at
    index ``state.rng_step[0]`` (0-based) before the counter advances.
    """

    def __call__(self, key: Array, state: object) -> object:
        return dataclasses.replace(state, rng_step=state.rng_step + 1)


def inject_host_rng_draws(state: object, config: object, key: Array) -> object:
    """tt port (wayfinder #92): carry one cycle's host-drawn RNG in the state.

    tt backend only; the CPU path never injects (``rng_draws`` stays
    ``None`` and every site falls back to the traced device draw, keeping
    CPU/reference runs byte-identical to before).

    Args:
        state: MCMC state (with ``groups``/``rng_step``/``rng_draws``).
        config: Run configuration (``min_cycle_length``).
        key: Per-cycle draw key from the run seed chain.

    Returns:
        The state with a fresh bundle and a zeroed step counter.
    """
    if not use_host_rng_draws() or not hasattr(state, "rng_draws"):
        return state
    counts = state.groups.data.system.counts.data
    n_steps = int(jnp.maximum(counts.max(), config.min_cycle_length))
    bundle = draw_rng_bundle(key, state, n_steps)
    return dataclasses.replace(
        state,
        rng_step=jnp.zeros((1,), dtype=jnp.int32),
        rng_draws=jax.device_put(bundle),
    )