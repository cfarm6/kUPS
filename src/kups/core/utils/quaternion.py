# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Quaternion representation for 3D rotations.

This module provides a quaternion class for representing and manipulating 3D
rotations efficiently. Quaternions avoid gimbal lock and provide smooth
interpolation compared to Euler angles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, core
from jax.interpreters import mlir

from kups.core.data import Sliceable

from .jax import dataclass, vectorize

import jax.extend.core as jax_core


_quaternion_rotate_p = jax_core.Primitive("quaternion_rotate")
_quaternion_rotate_p.multiple_results = False
_tt_lowering_registered = False


def _rotate_vectors_impl(q: Array, vector: Array) -> Array:
    """Rotate broadcast-compatible vectors with component arithmetic."""
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    vx, vy, vz = jnp.moveaxis(vector, -1, 0)
    return jnp.stack(
        [
            (w * w + x * x - y * y - z * z) * vx
            + 2 * (x * y - z * w) * vy
            + 2 * (x * z + y * w) * vz,
            2 * (x * y + z * w) * vx
            + (-x * x + y * y - z * z + w * w) * vy
            + 2 * (y * z - x * w) * vz,
            2 * (x * z - y * w) * vx
            + 2 * (y * z + x * w) * vy
            + (-x * x - y * y + z * z + w * w) * vz,
        ],
        axis=-1,
    )


_quaternion_rotate_p.def_impl(_rotate_vectors_impl)
_quaternion_rotate_p.def_abstract_eval(
    lambda q, vector: core.ShapedArray(vector.shape, vector.dtype)
)
mlir.register_lowering(
    _quaternion_rotate_p,
    mlir.lower_fun(_rotate_vectors_impl, multiple_results=False),
)


def _tt_quaternion_rotate_lowering(ctx: object, q: object, vector: object) -> list[object]:
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])  # type: ignore[attr-defined]
    layouts = [
        list(range(len(aval.shape) - 1, -1, -1))
        for aval in (*ctx.avals_in, *ctx.avals_out)  # type: ignore[attr-defined]
    ]
    return [
        mlir.custom_call(
            "tenstorrent.quaternion_rotate",
            result_types=[result_type],
            operands=[q, vector],
            operand_layouts=layouts[:2],
            result_layouts=layouts[2:],
        ).result
    ]


def _ensure_tt_quaternion_rotate_lowering() -> None:
    global _tt_lowering_registered
    if _tt_lowering_registered or jax.default_backend() != "tt":
        return
    mlir.register_lowering(
        _quaternion_rotate_p,
        _tt_quaternion_rotate_lowering,
        platform="tt",
    )
    _tt_lowering_registered = True


@dataclass
class Quaternion(Sliceable):
    """Unit quaternion representing a 3D rotation.

    Quaternions are represented as `(w, x, y, z)` where `w` is the scalar (real)
    part and `(x, y, z)` is the vector (imaginary) part. For a rotation by angle
    θ around axis **u**:

    $$
    q = \\cos(\\theta/2) + \\sin(\\theta/2) \\cdot \\mathbf{u}
    $$

    Attributes:
        components: Array of shape `(..., 4)` containing quaternion components as
            `[w, x, y, z]`.

    Example:
        ```python
        # Create identity quaternion (no rotation)
        q = Quaternion.identity()

        # Generate random rotation
        key = jax.random.PRNGKey(0)
        q_random = Quaternion.random(key, shape=(10,))

        # Rotate a point
        point = jnp.array([1.0, 0.0, 0.0])
        rotated = point @ q

        # Compose rotations
        q_combined = q1 * q2

        # Invert rotation
        q_inv = q.inv()
        ```
    """

    components: Array

    def __post_init__(self) -> None:
        # Skip checks for tracers/wrappers
        if not isinstance(self.components, jax.Array):
            return
        if self.components.shape[-1] != 4:
            raise ValueError(
                f"Quaternion components must have shape (..., 4), got {self.components.shape}"
            )

    @classmethod
    def from_uniform(cls, u: Array) -> Quaternion:
        """Build uniform SO(3) quaternions from pre-drawn ``u ~ U[0,1)``.

        The draw half of :meth:`random` (Shoemake 1992), factored out so the
        tt backend can substitute host-drawn uniforms (wayfinder #92); the
        ``u1,u2,u3`` -> quaternion math still runs on the device.

        Args:
            u: Array of shape ``(*batch, 3)`` of uniform draws.

        Returns:
            Random unit quaternion(s) with shape ``(*batch, 4)``.
        """
        u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
        q = jnp.stack(
            [
                jnp.sqrt(1 - u1) * jnp.sin(2 * jnp.pi * u2),
                jnp.sqrt(1 - u1) * jnp.cos(2 * jnp.pi * u2),
                jnp.sqrt(u1) * jnp.sin(2 * jnp.pi * u3),
                jnp.sqrt(u1) * jnp.cos(2 * jnp.pi * u3),
            ],
            axis=-1,
        )
        return cls(q)

    @classmethod
    def random(cls, key: Array, shape: tuple[int, ...] = ()) -> Quaternion:
        """Generate uniformly distributed random rotation quaternions.

        Uses the method from Shoemake (1992) to sample uniformly from the
        rotation group SO(3).

        Args:
            key: JAX PRNG key for random number generation.
            shape: Shape of the batch dimensions. Default is `()` for a single quaternion.

        Returns:
            Random unit quaternion(s) with shape `(*shape, 4)`.

        Reference:
            K. Shoemake, "Uniform random rotations", Graphics Gems III, 1992.
        """
        return cls.from_uniform(jax.random.uniform(key, shape=(*shape, 3)))

    @classmethod
    def identity(cls) -> Quaternion:
        """Create an identity quaternion representing no rotation.

        Returns:
            Identity quaternion `[1, 0, 0, 0]`.
        """
        return cls(jnp.array([1.0, 0.0, 0.0, 0.0]))

    def inv(self) -> Quaternion:
        """Compute the inverse (conjugate) quaternion.

        For unit quaternions, the inverse equals the conjugate, which negates
        the vector part while keeping the scalar part unchanged.

        Returns:
            Inverse quaternion that reverses this rotation.
        """
        return Quaternion(self.components * jnp.array([1, -1, -1, -1]))

    def __rmatmul__(self, other: object) -> Array:
        """Rotate a 3D vector or batch of vectors by this quaternion.

        Args:
            other: Array of shape ``(..., 3)`` representing 3D point(s).

        Returns:
            Rotated point(s) with the same shape as input.

        Raises:
            TypeError: If ``other`` is not a JAX array.
            ValueError: If the last dimension of ``other`` is not 3.
        """
        if not isinstance(other, jax.Array):
            raise TypeError(
                f"Unsupported type for right multiplication: {type(other)}. "
                "Expected Array."
            )
        if other.shape[-1] != 3:
            raise ValueError(
                f"Expected last dimension of other to be 3, got {other.shape[-1]}"
            )
        return _rotate_vectors(self.components, other)

    def __mul__(self, other: object) -> Quaternion:
        """Compose two rotations via quaternion multiplication.

        Quaternion multiplication is non-commutative. The result ``q1 * q2``
        applies rotation ``q2`` first, then ``q1``.

        Args:
            other: Another quaternion to compose with.

        Returns:
            Composed quaternion representing the combined rotation.

        Raises:
            TypeError: If ``other`` is not a ``Quaternion``.

        Example:
            ```python
            q1 = Quaternion.random(jax.random.PRNGKey(0))
            q2 = Quaternion.random(jax.random.PRNGKey(1))
            q_combined = q1 * q2  # Apply q2, then q1
            ```
        """
        if not isinstance(other, Quaternion):
            raise TypeError(
                f"Unsupported type for multiplication: {type(other)}. "
                "Expected Quaternion."
            )
        return Quaternion(_multiply_quaternions(self.components, other.components))

    def __pow__(self, other: object) -> Quaternion:
        """Raise quaternion to a power for scaled rotations.

        For a quaternion representing rotation by angle theta, raising it to
        power p produces a rotation by angle ``p * theta`` around the same axis.

        Args:
            other: Scalar or array exponent.

        Returns:
            Quaternion representing the scaled rotation.

        Raises:
            TypeError: If ``other`` is not a scalar, int, or JAX array.

        Example:
            ```python
            q = Quaternion.random(jax.random.PRNGKey(0))
            q_half = q ** 0.5  # Half the rotation angle
            ```
        """
        if not isinstance(other, (jax.Array, float, int)):
            raise TypeError(
                f"Unsupported type for exponentiation: {type(other)}. "
                "Expected Array or float."
            )
        return Quaternion(_pow_quaternion(self.components, other))

    def as_matrix(self) -> Array:
        """Convert quaternion to a 3×3 rotation matrix.

        Returns:
            Rotation matrix of shape `(..., 3, 3)` equivalent to this quaternion.

        Example:
            ```python
            q = Quaternion.identity()
            R = q.as_matrix()  # Returns 3×3 identity matrix
            ```
        """
        return _quat_to_mat(self.components)


@vectorize(signature="(4),(4)->(4)")
def _multiply_quaternions(q1: Array, q2: Array) -> Array:
    """Hamilton product of two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


@vectorize(signature="(4),()->(4)")
def _pow_quaternion(q: Array, exponent: Array | float) -> Array:
    """Raise a quaternion to a power.

    Handles the identity quaternion (zero rotation) gracefully by
    returning identity when ``sin_half_angle`` is near zero.
    """
    exponent = jnp.asarray(exponent)
    quat = jnp.where(q[0] < 0, -q, q)
    v = quat[1:]
    w = quat[0]

    angle = 2 * jnp.arccos(jnp.clip(w, -1.0, 1.0))
    sin_half_angle: Array = jnp.linalg.norm(v)  # type: ignore[assignment]

    # Safe division: use [1,0,0] axis when sin_half_angle ~ 0 (identity)
    safe_sin = jnp.where(sin_half_angle > 1e-10, sin_half_angle, 1.0)
    axis = v / safe_sin

    new_angle = angle * exponent
    new_v = axis * jnp.sin(new_angle / 2)
    new_w = jnp.cos(new_angle / 2)
    return jnp.concatenate([new_w[None], new_v])



def _rotate_vectors(q: Array, vector: Array) -> Array:
    """Rotate vectors with the TT semantic operation when available."""
    leading_shape = jnp.broadcast_shapes(q.shape[:-1], vector.shape[:-1])
    q_broadcast = jnp.broadcast_to(q, (*leading_shape, 4))
    vector_broadcast = jnp.broadcast_to(vector, (*leading_shape, 3))
    _ensure_tt_quaternion_rotate_lowering()
    return _quaternion_rotate_p.bind(q_broadcast, vector_broadcast)


@vectorize(signature="(4)->(3,3)")
def _quat_to_mat(q: Array) -> Array:
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w
    return jnp.array(
        [
            [+x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
        ]
    )
