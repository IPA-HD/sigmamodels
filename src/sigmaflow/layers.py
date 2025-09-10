import equinox as eqx
import jax
import jax.numpy as jnp
from .flow import Laplace_Beltrami
from einops import rearrange
import numpy as np


def patch_factory(features, labels, size):
    w = np.max(features.shape)

    @jax.vmap
    def _ret(n):
        i, j = np.random.randint(0, w - size, 2)
        k = np.random.randint(0, 1)
        return (
            features[k, i : i + size, j : j + size],
            labels[k, i : i + size, j : j + size],
        )

    return _ret


class Aggregator(eqx.Module):
    cnv: eqx.nn.Conv2d

    def __init__(self, dim1, dim2, ks, key):
        self.cnv = eqx.nn.Conv2d(dim1, dim2, kernel_size=ks, padding="same", key=key)

    def __call__(self, x):
        x = rearrange(x, "w h c -> c w h")
        x = self.cnv(x)
        return rearrange(x, "c w h -> w h c")


# class Aggregator(eqx.Module):
#     # cnv: eqx.nn.MultiheadAttention
#     # norm: eqx.nn.LayerNorm

#     def __init__(self, dim1, dim2, key):
#         self.attn = eqx.nn.MultiheadAttention(1, dim, key=key)
#         self.norm = eqx.nn.LayerNorm(dim)

#     def __call__(self, x):
#         w, h, c = x.shape
#         x = rearrange(x, "w h c -> (w h) c")
#         x = jax.vmap(self.norm)(x + self.attn(x, x, x))
#         return x.reshape(w, h, c)


def state_to_metric(x):
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

    def __init__(self, key, dim1, dim2, ks, mass):
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

    def __call__(self, x, key=None):
        y = self.cnv(x)
        jjm = jax.vmap(jax.vmap((self.mlp)))
        y = jjm(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        return (1 + self.mass) * x + 0.5 * Laplace_Beltrami(a, x) / b

    def metric(self, x, key=None):
        y = self.agg(x)
        y = jax.vmap(jax.vmap((self.mlp)))(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        return a, b


class AFLayer(eqx.Module):
    cnv: Aggregator
    mass: jax.Array
    mlp: eqx.nn.Sequential
    noise: float
    key: int

    def __init__(self, key, dim1, dim2, ks, mass, noise=0):
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

    def __call__(self, x, key=None):
        y = self.cnv(x)
        jjm = jax.vmap(jax.vmap((self.mlp)))
        y = jjm(y)
        a, b = jax.vmap(jax.vmap(state_to_metric))(y)
        q = x + jax.random.normal(jax.random.key(self.key), x.shape) * self.noise
        p = jax.nn.softmax(x, axis=-1)
        return (1 + self.mass) * x + 0.5 * Laplace_Beltrami(a, p) / b

    def metric(self, x, key=None):
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

    def metric(self, x, key=None):
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


def scale_model(m, scale):
    weights = lambda mp: jax.tree_util.tree_leaves(
        mp, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    tr = weights(m)
    rescaled = jax.tree.map(lambda x: scale * x if isinstance(x, jax.Array) else x, tr)
    return eqx.tree_at(weights, m, rescaled)


def sigmasimple(nl, dim1, dim2, ks, mass, scale, seed=13812378, **kwargs):
    key = jax.random.key(seed)
    m = SigmaFlow(key, nl, dim1, dim2, ks, mass)
    return scale_model(m, scale)


if __name__ == "__test__":
    #

    from brute import *
    from optax import softmax_cross_entropy_with_integer_labels as ce
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.load("ex5/features.npz")
    x = jnp.array([x["l1"], x["l3"]])[0]
    l = np.load("ex5/labels.npz")
    l = jnp.array([l["l1"], l["l3"]])[0]
    m = sigmasimple(20, 9, 32, 15, 1, 0)
    # m = sigmalayers(20, 9, 32, 15, 1, 1.0)
    # n = sigmalayers(10, 9, 32, 15, 1, 2.0)
    plt.imshow(m(x).argmax(-1))
    # len(m)

    eqx.tree_serialise_leaves("ex5/tst", m)
    r = eqx.tree_deserialise_leaves("ex5/tst", n)
    len(m)

    # plt.imshow(m(x).argmax(-1), cmap="jet")
    np.mean(l != x.argmax(-1))
    ce(x, l).mean() / jnp.log(9)

    import h5py

    f = h5py.File("ex5/featureVectors.h5", "r")
    fs = f["Dataset1"]
    plt.imshow(fs[30:180, 30:180].argmax(-1), cmap="tab20")
    plt.imshow(x.argmax(-1)[30:180, 30:180], cmap="tab20")
    plt.imshow(l[30:180, 30:180], cmap="tab20")

    q = fs[30:180, 30:180]
    q.max(), q.min()
    r = x[30:180, 30:180]
    r.max(), r.min()

    plt.imshow((fs[30:180, 30:180] - x[30:180, 30:180]).argmax(-1))

    plt.imshow(m(x).argmax(-1), cmap="jet")
    plt.imshow(x.argmax(-1), cmap="jet")
    plt.imshow(l, cmap="jet")

    l.shape
    mp.conv.weight.shape
