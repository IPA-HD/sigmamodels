import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dx
from functools import reduce
from .matrices import D, Lweights
from .lb import Laplace_Beltrami, norm_cotangent


def convolving(x, y):
    """
    Convolve signal with filter y along first two dimensions with circular padding.
    """
    padding = int((y.shape[-1] - 1) / 2)
    x = jax.vmap(lambda x: jnp.pad(x, padding, mode="wrap"), -1, -1)(x)[
        jnp.newaxis, ...
    ]
    res = jax.lax.conv_general_dilated(
        x,
        y,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("CHWN", "IOHW", "CHWN"),
    )
    return res


def derivative(x):
    """
    Convolve with the derivative Filter D.
    """
    return convolving(x, y=jnp.array(D))


def laplacian(x):
    """
    Convolve with the derivative Filter D.
    """
    return convolving(x, y=jnp.array(Lweights))[0]


def Pi_0(x):
    """
    Project onto tangent space T_0.
    """
    return x - x.mean(-1, keepdims=True)


def sm(x):
    """
    Softmax function.
    """
    return jax.nn.softmax(x, axis=-1)


def sigmaflow(
    v0,
    t,
    nabla=derivative,
    *,
    dt=0.2,
    ctrl=dx.ConstantStepSize(),
    solver=dx.Euler(),
    alpha=0,
    m=0,
    mode="adaptive",
):
    """
    Integrate the sigma flow PDE with a constant flat metric with starting point v0.
    """

    def RHS(t, v, args):
        logp = jax.nn.log_softmax(v, axis=-1)
        return jnp.clip(
            Pi_0(
                laplacian(v) + ((1 - alpha) / 2) * (nabla(logp) ** 2).sum(0) + m * (v)
            ),
            -1e9,
            1e9,
        )

    if mode == "adaptive":
        f = dx.ODETerm(RHS)
        sol = dx.diffeqsolve(
            f,
            solver,
            t0=0,
            t1=t,
            y0=v0,
            dt0=dt,
            saveat=dx.SaveAt(ts=np.linspace(0, t, 10)),
            stepsize_controller=ctrl,
        ).ys

    if mode == "fast":
        # note: only the endpoint is returned in this integration method
        n = int(t / dt)
        sol = reduce(lambda x, t: x + t * RHS(0, x, 0), n * [dt], v0)[np.newaxis]

    return sol


def sigmaflow_anisotropic(
    v0,
    metric,
    *,
    t=1,
    nabla=derivative,
    dt=0.2,
    m=0,
    alpha=0,
    ctrl=dx.ConstantStepSize(),
    solver=dx.Euler(),
    mode="adaptive",
):
    """
    Intergrate the sigma flow PDE with starting point v0 and potentially variable metric.
    """

    @jax.jit
    def RHS(t, v, args):
        logp = jax.nn.log_softmax(v, axis=-1)
        diffusion_tensor, deth, hinv = metric(v, t)
        return jnp.clip(
            Pi_0(
                (
                    Laplace_Beltrami(diffusion_tensor, v) / deth
                    + ((1 - alpha) / 2) * (norm_cotangent(hinv, nabla(logp)))
                )
                + m * v
            ),
            -1e8,
            1e8,
        )

    if mode == "adaptive":
        f = dx.ODETerm(RHS)
        sol = dx.diffeqsolve(
            f,
            solver,
            t0=0,
            t1=t,
            y0=v0,
            dt0=dt,
            saveat=dx.SaveAt(ts=jnp.linspace(0, t, 10)),
            stepsize_controller=ctrl,
        ).ys

    if mode == "fast":
        n = int(t / dt)
        sol = jnp.array(
            reduce(lambda ls, t: ls + [ls[-1] + t * RHS(t, ls[-1], 0)], n * [dt], [v0])
        )

    return sol


def sigmaflow_anisotropic_static(
    v0,
    metric,
    *,
    t=1,
    nabla=derivative,
    dt=0.2,
    msq=0,
    alpha=0,
    ctrl=dx.ConstantStepSize(),
    solver=dx.Euler(),
    mode="adaptive",
):
    """
    Anisotropic sigma flow but with a constant local metric tensor.
    """
    metric = metric()

    def RHS(t, v, args):
        diff_t, scale = args
        logp = jax.nn.log_softmax(v, axis=-1)
        term = Pi_0(
            scale
            * (
                Laplace_Beltrami(diff_t, v)
                + ((1 - alpha) / 2) * (norm_cotangent(diff_t, nabla(logp)))
            )
            + msq * (v)
        )
        return jnp.clip(term, -1e8, 1e8)

    if mode == "adaptive":
        f = dx.ODETerm(RHS)
        sol = dx.diffeqsolve(
            f,
            solver,
            t0=0,
            t1=t,
            y0=v0,
            dt0=dt,
            saveat=dx.SaveAt(ts=jnp.linspace(0, t, 10)),
            stepsize_controller=ctrl,
            args=metric,
        ).ys

    if mode == "fast":
        diff_t, scale = metric
        steps = int(t / dt)
        sol = reduce(
            lambda x, y: x + y * RHS(0, x, (diff_t, scale)),
            steps * [dt],
            v0,
        )[np.newaxis]
    return sol


def sigmaflow_anisotropic_static_jax49(
    v0,
    metric,
    *,
    t=1,
    nabla=derivative,
    dt=0.2,
    msq=0,
    alpha=0,
    ctrl=dx.ConstantStepSize(),
    solver=dx.Euler(),
    mode="adaptive",
):
    """
    Anisotropic sigma flow but with a constant local metric tensor.
    """

    # def RHS(t, v, args):
    #     diffusing_tensor, det_h, h_inv = args
    #     logp = jax.nn.log_softmax(v, axis=-1)
    #     term = Pi_0(
    #         Laplace_Beltrami(diffusing_tensor, v) / det_h
    #         + ((1 - alpha) / 2) * (norm_cotangent(h_inv, nabla(logp)))
    #         + msq * sm(v)
    #     )
    #     return jnp.clip(term, -1e8, 1e8)
    metric = metric()

    def RHS(t, v, args):
        diff_t, scale = args
        logp = jax.nn.log_softmax(v, axis=-1)
        term = Pi_0(
            scale
            * (
                Laplace_Beltrami(diff_t, v)
                + ((1 - alpha) / 2) * (norm_cotangent(diff_t, nabla(logp)))
            )
            + msq * sm(v)
        )
        return jnp.clip(term, -1e8, 1e8)

    if mode == "adaptive":
        f = dx.ODETerm(RHS)
        sol = dx.diffeqsolve(
            f,
            solver,
            t0=0,
            t1=t,
            y0=v0,
            dt0=dt,
            saveat=dx.SaveAt(ts=jnp.linspace(0, t, 10)),
            stepsize_controller=ctrl,
            args=metric,
        ).ys

    if mode == "fast":
        # diffusion_tensor, det_h, h_inv = metric
        diff_t, scale = metric
        steps = int(t / dt)
        sol = reduce(
            # lambda x, y: x + y * RHS(0, x, (diffusion_tensor, det_h, h_inv)),
            lambda x, y: x + y * RHS(0, x, (diff_t, scale)),
            steps * [dt],
            v0,
        )[np.newaxis]
    return sol
