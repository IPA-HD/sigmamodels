import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def metric_to_filter(diffusion_tensor):
    """
    Calcualte the local filter matrix from a given discrete diffusion tensor/conformally invariant metric.
    """
    dt00 = diffusion_tensor
    dt01 = jnp.roll(dt00, 1, axis=1)
    dt10 = jnp.roll(dt00, 1, axis=0)
    dt11 = jnp.roll(dt01, 1, axis=0)
    dt_10 = jnp.roll(dt00, -1, axis=0)
    dt_11 = jnp.roll(dt_10, 1, axis=1)
    a00, c00, b00 = jnp.split(dt00, 3, axis=-1)
    a01, c01, b01 = jnp.split(dt01, 3, axis=-1)
    a10, c10, b10 = jnp.split(dt10, 3, axis=-1)
    a11, c11, b11 = jnp.split(dt11, 3, axis=-1)
    a_11, c_11, b_11 = jnp.split(dt_11, 3, axis=-1)
    A0 = (jnp.abs(b_11) - b_11 + jnp.abs(b00) - b00) / 4.0
    A1 = (c01 + c00 - jnp.abs(b01) - jnp.abs(b00)) / 2.0
    A2 = (jnp.abs(b11) + b11 + jnp.abs(b00) + b00) / 4.0
    A5 = (a10 + a00 - jnp.abs(b10) - jnp.abs(b00)) / 2.0
    down = jnp.roll(A1, -1, axis=1)
    left = jnp.roll(A5, -1, axis=0)
    down_left = jnp.roll(A2, (-1, -1), axis=(0, 1))
    down_right = jnp.roll(A0, (1, -1), axis=(0, 1))
    A7 = down
    A6 = down_left
    A3 = left
    A8 = down_right
    A4 = -(A0 + A1 + A2 + A3 + A5 + A6 + A7 + A8)
    A = jnp.stack([A0, A1, A2, A3, A4, A5, A6, A7, A8], axis=-1)
    return A


def Laplace_Beltrami(diffusion_tensor, x):
    """
    Compute the unnormalized Laplace-Beltrami operator with a given diffusion tensor and signal X.
    """
    A = metric_to_filter(diffusion_tensor)
    return jax.lax.conv_general_dilated_local(
        x[jnp.newaxis, ...],
        A,
        filter_shape=(3, 3),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("CHWN", "HWOI", "CHWN"),
    )[0]


# @partial(jax.vmap, in_axes=[None, -1], out_axes=-1)
def norm_cotangent(inv_metric, omega):
    """
    Compute the norm induced by a metric and a contangent vector omega.
    """
    a, c, b = jnp.split(inv_metric, 3, axis=-1)
    omegax, omegay = jnp.split(omega, 2)
    return (a * (omegax**2) + 2 * b * (omegax * omegay) + c * (omegay**2))[0]
