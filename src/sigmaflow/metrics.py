import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def metric_generator_cells(
    x: Float[Array, "3"],
) -> tuple[Float[Array, "3"], Float[Array, "1"]]:
    """Metric for cells"""
    v, scale, alpha = x
    scale = jax.nn.sigmoid(scale) * 0.5 + 0.5
    l1 = jax.nn.softplus(v) + 0.5
    cos2 = jax.nn.sigmoid(alpha)
    cos = jnp.sqrt(cos2)
    a = jnp.arccos(cos)
    sincos = jnp.tan(a) * cos2
    Delta = (1 - l1**2) / l1
    l2 = l1 + Delta
    a = l2 - cos2 * Delta
    c = l1 + cos2 * Delta
    b = Delta * sincos
    diff_tens = jnp.array([a, c, b])
    scale = scale[..., jnp.newaxis]
    return diff_tens, scale


def metric_generator_baboon(
    x: Float[Array, "3"],
) -> tuple[Float[Array, "3"], Float[Array, "1"]]:
    """Metric for baboon"""
    v, scale, alpha = x
    scale = jax.nn.sigmoid(scale) * 1 + 0.1
    l1 = jax.nn.softplus(v) + 1
    cos2 = jax.nn.sigmoid(alpha)
    cos = jnp.sqrt(cos2)
    a = jnp.arccos(cos)
    sincos = jnp.tan(a) * cos2
    Delta = (1 - l1**2) / l1
    l2 = l1 + Delta
    a = l2 - cos2 * Delta
    c = l1 + cos2 * Delta
    b = Delta * sincos
    diff_tens = jnp.array([a, c, b])
    scale = scale[..., jnp.newaxis]
    return diff_tens, scale
