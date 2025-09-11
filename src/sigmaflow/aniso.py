import jax
import jax.numpy as jnp
from einops import rearrange
from collections.abc import Callable
import equinox as eqx
from functools import reduce
import numpy as np
import optax
import os
import diffrax as dx
from .flow import derivative, Pi_0, sigmaflow


class CellData:
    def __init__(self):
        dn = os.path.dirname
        path = dn(dn(dn(__file__))) + "/data/voronoi.npz"
        self.dat = np.load(path)["arr_0"]

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        return self.dat[idx][None]


@jax.jit
def norm(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def make_x0(batch, noise, alpha=0.9, filter=None):
    gt = batch
    gt = rearrange(gt, "1 w  h 1 -> w h")
    x = jnp.log(optax.smooth_labels(jnp.eye(20)[gt], alpha))
    noise = noise * np.random.randn(*x.shape)
    if filter is not None:
        return norm(filter(norm(x) + noise)), gt
    else:
        return norm(norm(x) + noise), gt


@jax.jit
def dt_from_metric(a, b, c):
    dg = rearrange(jnp.sqrt(a * c - b**2 + 1e-6), "w h -> w h 1")
    dt = jnp.stack([c, a, -b], axis=-1)
    dt = dt / dg
    hinv = dt / (dg**2)
    return dt, dg, hinv


def conservative_conversion(x):
    l1, l2, alpha = x
    l1 = jax.nn.softplus(l1 + 0.9) + 0.1
    l2 = jax.nn.softplus(l2 + 0.9) + 0.1
    alpha = jax.nn.tanh(alpha)
    cos = alpha  # jnp.cos(alpha)
    sin = jnp.sqrt(1 - cos**2)  # jnp.sin(alpha)
    a = (1 / l1) * (cos**2) + (1 / l2) * (sin**2)
    c = (1 / l1) * (sin**2) + (1 / l2) * (cos**2)
    b = ((1 / l2) - (1 / l1)) * cos * sin
    hinv = jnp.array([a, c, b])
    diffusion_tensor = jnp.array(
        [
            jnp.sqrt(l2 / l1) * (cos**2) + jnp.sqrt(l1 / l2) * (sin**2),
            jnp.sqrt(l2 / l1) * (sin**2) + jnp.sqrt(l1 / l2) * (cos**2),
            (jnp.sqrt(l1 / l2) - jnp.sqrt(l2 / l1)) * cos * sin,
        ]
    )
    deth = l1 * l2
    return diffusion_tensor, deth[..., jnp.newaxis], hinv


def struct_tensor(x, t):
    dt = jnp.linalg.norm(derivative(x), axis=-1)
    aa = dt[0] ** 2 + 1
    bb = dt[0] * dt[1]
    cc = dt[1] ** 2 + 1
    return dt_from_metric(aa, bb, cc)


def conversion_experiment(x):
    l1, l2, alpha = x
    l1 = jax.nn.softplus(l1 + 0.9) + 0.1
    l2 = jax.nn.softplus(l2 + 0.9) + 0.1
    cos2 = jax.nn.sigmoid(alpha + 1)
    cos = jnp.sqrt(cos2)
    a = jnp.arccos(cos)
    sincos = jnp.tan(a) * cos2
    a = (1 / l1) * (cos2) + (1 / l2) * (1 - cos2)
    c = (1 / l1) * (1 - cos2) + (1 / l2) * (cos2)
    b = ((1 / l2) - (1 / l1)) * sincos
    hinv = jnp.array([a, c, b])
    diffusion_tensor = jnp.array(
        [
            jnp.sqrt(l2 / l1) * (cos2) + jnp.sqrt(l1 / l2) * (1 - cos2),
            jnp.sqrt(l2 / l1) * (1 - cos2) + jnp.sqrt(l1 / l2) * (cos2),
            (jnp.sqrt(l1 / l2) - jnp.sqrt(l2 / l1)) * sincos,
        ]
    )
    deth = l1 * l2
    return diffusion_tensor, deth[..., jnp.newaxis], hinv


def conversion_best_experiment(x):
    v, scale, alpha = x
    scale = jax.nn.sigmoid(scale + 1) + 0.01  # * 0.0 + 1
    l1 = jax.nn.sigmoid(v + 1) + 0.01
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
    return diff_tens, root_det_h, diff_tens / root_det_h


def conversion_best(x):
    v, scale, alpha = x
    scale = jax.nn.sigmoid(scale + 1)  # * 0.0 + 1
    l1 = jax.nn.softplus(v + 0.9) + 0.1
    cos2 = jax.nn.sigmoid(alpha + 1)
    cos = jnp.sqrt(cos2)
    a = jnp.arccos(cos)
    sincos = jnp.tan(a) * cos2
    Delta = (1 - l1**2) / l1
    l2 = l1 + Delta
    a = l2 - cos2 * Delta
    c = l1 + cos2 * Delta
    b = Delta * sincos
    diff_tens = jnp.array([a, c, b])
    det = scale[..., jnp.newaxis]
    return diff_tens, det, diff_tens * det


class metric_network(eqx.Module):
    conv: None
    mlp: None
    converter: None

    def __init__(self, key, dim, conv_dim=64, ks=15, converter=conversion_best):
        keys = iter(jax.random.split(key, 4))
        self.conv = eqx.nn.Conv2d(
            dim, conv_dim, key=next(keys), kernel_size=ks, padding=int((ks - 1) / 2)
        )
        dims = [conv_dim + 1] + [int(conv_dim / (2**k)) for k in [1, 2]] + [3]
        self.converter = jax.vmap(jax.vmap(converter))

        lst = reduce(
            lambda l, d: l
            + [
                eqx.nn.LayerNorm(d[0]),
                eqx.nn.Linear(d[0], d[1], key=next(keys)),
                eqx.nn.Lambda(jax.nn.gelu),
            ],
            zip(dims[:-1], dims[1:]),
            [],
        )

        self.mlp = eqx.nn.Sequential(lst[:-1] + [eqx.nn.LayerNorm(3)])

    @eqx.filter_jit
    def __call__(self, x, t):
        x = rearrange(x, "w h c -> c w h")
        x = self.conv(x)
        V = rearrange(x, "c w h -> w h c")
        V = jnp.pad(V, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=t)
        V = jax.vmap(jax.vmap(self.mlp))(V)
        return self.converter(V)

    def read_out(self, x, t):
        x = rearrange(x, "w h c -> c w h")
        x = self.conv(x)
        V = rearrange(x, "c w h -> w h c")
        V = jnp.pad(V, pad_width=((0, 0), (0, 0), (0, 1)), constant_values=t)
        V = self.mlp(V)
        return V


@jax.jit
def dt_to_matrix(dt):
    dt00 = dt
    dt01 = jnp.roll(dt00, 1, axis=1)
    dt10 = jnp.roll(dt00, 1, axis=0)
    dt11 = jnp.roll(dt01, 1, axis=0)
    dt_10 = jnp.roll(dt00, -1, axis=0)
    dt0_1 = jnp.roll(dt00, -1, axis=1)
    dt_1_1 = jnp.roll(dt_10, -1, axis=1)
    dt_11 = jnp.roll(dt_10, 1, axis=1)
    dt1_1 = jnp.roll(dt0_1, 1, axis=0)
    a00, c00, b00 = jnp.split(dt00, 3, axis=-1)
    a01, c01, b01 = jnp.split(dt01, 3, axis=-1)
    a10, c10, b10 = jnp.split(dt10, 3, axis=-1)
    a11, c11, b11 = jnp.split(dt11, 3, axis=-1)
    a0_1, c0_1, b0_1 = jnp.split(dt0_1, 3, axis=-1)
    a_10, c_10, b_10 = jnp.split(dt_10, 3, axis=-1)
    a_1_1, c_1_1, b_1_1 = jnp.split(dt_1_1, 3, axis=-1)
    a_11, c_11, b_11 = jnp.split(dt_11, 3, axis=-1)
    a1_1, c1_1, b1_1 = jnp.split(dt1_1, 3, axis=-1)
    A0 = (jnp.abs(b_11) - b_11 + jnp.abs(b00) - b00) / 4.0
    A1 = (c01 + c00 - jnp.abs(b01) - jnp.abs(b00)) / 2.0
    A2 = (jnp.abs(b11) + b11 + jnp.abs(b00) + b00) / 4.0
    A3 = (a_10 + a00 - jnp.abs(b_10) - jnp.abs(b00)) / 2.0
    A4 = (
        -(a_10 + 2 * a00 + a10) / 2.0
        - (jnp.abs(b_11) - b_11 + jnp.abs(b11) + b11) / 4.0
        - (jnp.abs(b_1_1) + b_1_1 + jnp.abs(b1_1) - b1_1) / 4.0
        + (
            jnp.abs(b_10)
            + jnp.abs(b10)
            + jnp.abs(b0_1)
            + jnp.abs(b01)
            + 2 * jnp.abs(b00)
        )
        / 2.0
        - (c0_1 + 2 * c00 + c01) / 2.0
    )
    A5 = (a10 + a00 - jnp.abs(b10) - jnp.abs(b00)) / 2.0
    A6 = (jnp.abs(b_1_1) + b_1_1 + jnp.abs(b00) + b00) / 4.0
    A7 = (c0_1 + c00 - jnp.abs(b0_1) - jnp.abs(b00)) / 2.0
    A8 = (jnp.abs(b1_1) - b1_1 + jnp.abs(b00) - b00) / 4.0
    A = jnp.stack([A0, A1, A2, A3, A4, A5, A6, A7, A8], axis=-1)
    return A


def LB(dt, X):
    A = dt_to_matrix(dt)
    return jax.lax.conv_general_dilated_local(
        X[jnp.newaxis, ...],
        A,
        filter_shape=(3, 3),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("CHWN", "HWOI", "CHWN"),
    )[0]


def nrm(dt, v):
    return (
        dt[:, :, 0] * (v[0] ** 2)
        + 2 * dt[:, :, 2] * (v[1] * v[0])
        + dt[:, :, 1] * (v[1] ** 2)
    )


nrm2 = jax.vmap(nrm, [None, -1], -1)


def aniso_sigma(
    v0,
    metric_predictor,
    t,
    *,
    nabla=derivative,
    dt=0.2,
    msq=0,
    alpha=0,
    ctrl=dx.ConstantStepSize(),
    solver=dx.Euler(),
):
    @jax.jit
    def RHS(t, v, args):
        logp = jax.nn.log_softmax(v, axis=-1)
        # p = sm(v)
        Dt, dg, hinv = metric_predictor(v, t)
        return jnp.clip(
            Pi_0(
                (LB(Dt, v) / dg + ((1 - alpha) / 2) * (nrm2(hinv, nabla(logp))))
                + msq * (v)
            ),
            -1e8,
            1e8,
        )

    n = int(t / dt)
    sol = reduce(lambda x, t: x + t * RHS(t, x, 0), n * [dt], v0)[np.newaxis]

    return sol


class flow_module(eqx.Module):
    t: float
    m: float
    alpha: float = 0
    dt: float = 0.2
    ctrl: dx.AbstractStepSizeController = dx.ConstantStepSize()
    solver: dx.AbstractItoSolver = dx.Euler()
    nabla: Callable = derivative

    def __call__(self, v, mp):
        return aniso_sigma(
            v,
            mp,
            self.t,
            nabla=self.nabla,
            dt=self.dt,
            msq=self.m,
            alpha=self.alpha,
            ctrl=self.ctrl,
            solver=self.solver,
        )[-1]

    def integrate(self, v, mp, t):
        return aniso_sigma(
            v,
            mp,
            t,
            nabla=self.nabla,
            dt=self.dt,
            msq=self.m,
            alpha=self.alpha,
            ctrl=self.ctrl,
            solver=self.solver,
        )[-1]


class n_sigma_model(eqx.Module):
    mp: Callable
    flow: flow_module

    def __call__(self, v):
        return self.flow(v, self.mp)

    def integrate(self, v, t):
        return self.flow.integrate(v, self.mp, t)


def aniso_score(model, y, t=0):
    t = model.flow.t * t
    a, c, b = jnp.split(model.mp(y, t)[2], 3, axis=-1)
    Delta = jnp.sqrt((a - c) ** 2 + 4 * b**2 + 1e-5)
    D = jnp.concatenate([(a + c + Delta) * 0.5, (a + c - Delta) * 0.5], axis=-1)
    D = jnp.log(D)
    return 0.5 * np.sum(D**2, axis=-1) - D.prod(-1)


def base_m(x):
    return sigmaflow(x, 2, m=4)[-1]
