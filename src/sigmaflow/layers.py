import equinox as eqx
import jax
import jax.numpy as jnp
from .flow import Laplace_Beltrami
from einops import rearrange
import numpy as np
from jaxtyping import PRNGKeyArray, Float, Array
from collections.abc import Callable
from typing import Generic, TypeVar


class Aggregator(eqx.Module):
    cnv: eqx.nn.Conv2d

    def __init__(self, dim1: int, dim2: int, ks: int, key: PRNGKeyArray):
        self.cnv = eqx.nn.Conv2d(dim1, dim2, kernel_size=ks, padding="same", key=key)

    def __call__(self, x: Float[Array, "w h c"]) -> Float[Array, "w h d"]:
        x = rearrange(x, "w h c -> c w h")
        x = self.cnv(x)
        return rearrange(x, "c w h -> w h c")


def state_to_metric(
    x: Float[Array, "3"],
) -> tuple[Float[Array, "3"], Float[Array, "1"]]:
    v, scale, alpha = x
    scale = 1 - jax.nn.tanh(jnp.abs(scale)) * 0.9
    l1 = 1 - jax.nn.tanh(jnp.abs(v)) * 0.9
    Delta = (1 - l1**2) / l1
    alpha = jax.nn.tanh(alpha) * jnp.pi * 0.5
    cos = jnp.cos(alpha)
    cos2 = cos**2
    sincos = jnp.sin(alpha) * cos
    a = l1 + (1 - cos2) * Delta
    c = l1 + cos2 * Delta
    b = Delta * sincos
    diff_tens = jnp.array([a, c, b])
    root_det_h = scale[..., jnp.newaxis]
    return diff_tens, root_det_h


class SFLayer(eqx.Module):
    cnv: Aggregator
    mass: jax.Array
    mlp: eqx.nn.Sequential

    def __init__(
        self, key: PRNGKeyArray, dim1: int, dim2: int, ks: int, mass: float | int
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.cnv = Aggregator(dim1, dim2, ks, k2)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim2, dim2, key=k2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Linear(dim2, 3, key=k3),
            ]
        )
        self.mass = jnp.array(mass)

    def __call__(self, x: Float[Array, "w h c"], key=None) -> Float[Array, "w h c"]:
        x = self.cnv
        y = self.cnv(x)
        jjm = jax.vmap(jax.vmap((self.mlp)))
        y = jjm(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        return (1 + self.mass) * x + 0.5 * Laplace_Beltrami(a, x) / b

    def metric(
        self, x: Float[Array, "w h c"], key=None
    ) -> tuple[Float[Array, "w h 3"], Float[Array, "w h 1"]]:
        y = self.agg(x)
        y = jax.vmap(jax.vmap((self.mlp)))(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        return a, b


class AFLayer(eqx.Module):
    cnv: Aggregator
    mass: Array
    mlp: eqx.nn.Sequential
    noise: float
    key: int

    def __init__(
        self,
        key: PRNGKeyArray,
        dim1: int,
        dim2: int,
        ks: int,
        mass: int | float,
        noise: float = 0.0,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.cnv = Aggregator(dim1, dim2, ks, k2)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim2, dim2, key=k2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Linear(dim2, 3, key=k3),
            ]
        )
        self.mass = jnp.array(mass)
        self.key = np.random.randint(0, 13891298)
        self.noise = noise

    def __call__(self, x: Float[Array, "w h c"], key=None) -> Float[Array, "w h c"]:
        y = self.cnv(x)
        jjm = jax.vmap(jax.vmap((self.mlp)))
        y = jjm(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        q = x + jax.random.normal(jax.random.key(self.key), x.shape) * self.noise
        p = jax.nn.softmax(x, axis=-1)
        return (1 + self.mass) * x + 0.5 * Laplace_Beltrami(a, p) / b

    def metric(
        self, x, key=None
    ) -> tuple[Float[Array, "w h 3"], Float[Array, "w h 1"]]:
        y = self.agg(x)
        y = jax.vmap(jax.vmap((self.mlp)))(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        return a, b


def sigmalayers(nl, dim1, dim2, ks, mass, scale, seed=13812378, **kwargs):
    key = jax.random.key(seed)
    m = eqx.nn.Sequential(
        [SFLayer(k, dim1, dim2, ks, mass=mass) for k in jax.random.split(key, nl)]
    )
    weights = lambda mp: jax.tree_util.tree_leaves(
        mp, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    tr = weights(m)
    wgts = jax.tree.map(lambda x: scale * x if isinstance(x, jax.Array) else x, tr)
    return eqx.tree_at(weights, m, wgts)


def aflayers(nl, dim1, dim2, ks, mass, scale, af_noise, seed=13812378, **kwargs):
    key = jax.random.key(seed)
    m = eqx.nn.Sequential(
        [
            AFLayer(k, dim1, dim2, ks, mass=mass, noise=af_noise / (i + 1))
            for i, k in enumerate(jax.random.split(key, nl))
        ]
    )
    weights = lambda mp: jax.tree_util.tree_leaves(
        mp, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    tr = weights(m)
    wgts = jax.tree.map(lambda x: scale * x if isinstance(x, jax.Array) else x, tr)
    return eqx.tree_at(weights, m, wgts)


class SigmaFlow(eqx.Module):
    cnv: Aggregator
    mlp: eqx.nn.Sequential
    mass: jax.Array
    nl: int

    def __init__(self, key, nl, dim1, dim2, ks, mass):
        k1, k2, k3 = jax.random.split(key, 3)
        self.cnv = Aggregator(dim1 + 1, dim2, ks, k2)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim2, dim2, key=k2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Linear(dim2, 3, key=k3),
            ]
        )
        self.mass = jnp.array(mass)
        self.nl = nl

    def __call__(self, x, key=None):
        jjm = jax.vmap(jax.vmap((self.mlp)))
        v = x
        for i in range(self.nl):
            y = jnp.pad(v, ((0, 0), (0, 0), (0, 1)), constant_values=i)
            y = self.cnv(y)
            y = jjm(y)
            a, b = jax.vmap(jax.vmap(state_to_metric))(y)
            v = (1 + self.mass) * v + 0.5 * Laplace_Beltrami(a, v) / b

        return v

    def metric(
        self, x: Float[Array, "w h c"], key=None
    ) -> tuple[
        Float[Array, "w h c"], list[Float[Array, "w h 3"]], list[Float[Array, "w h 1"]]
    ]:
        jjm = jax.vmap(jax.vmap((self.mlp)))
        v = x
        A = []
        B = []
        for i in range(self.nl):
            y = jnp.pad(v, ((0, 0), (0, 0), (0, 1)), constant_values=i)
            y = self.cnv(y)
            y = jjm(y)
            a, b = jax.vmap(jax.vmap(state_to_metric))(y)
            v = (1 + self.mass) * v + 0.5 * Laplace_Beltrami(a, v) / b
            A.append(a)
            B.append(b)

        return v, A, B


def scale_model(m: SigmaFlow, scale: float) -> SigmaFlow:
    weights = lambda mp: jax.tree_util.tree_leaves(
        mp, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    tr = weights(m)
    rescaled = jax.tree.map(lambda x: scale * x if isinstance(x, jax.Array) else x, tr)
    return eqx.tree_at(weights, m, rescaled)


def sigmasimple(
    nl: int,
    dim1: int,
    dim2: int,
    ks: int,
    mass: float | int,
    scale: float,
    seed=13812378,
    **kwargs,
) -> SigmaFlow:
    key = jax.random.key(seed)
    m = SigmaFlow(key, nl, dim1, dim2, ks, mass)
    return scale_model(m, scale)
