from collections.abc import Callable
from typing import Tuple
from dataclasses import dataclass
from dataclasses import field
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray, Key, UInt8
from optax import softmax_cross_entropy_with_integer_labels as ce

from .metrics import metric_generator_cells, metric_generator_baboon
from .flow import sigmaflow_anisotropic_static


########################## CONSTANTS ##########################
Ar = Array | np.ndarray
JV = jax.vmap


@dataclass
class Constants:
    L_CELLS: UInt8[Ar, "512 512"]
    L_BABOON: Int[Ar, "512 512"]
    KEY: PRNGKeyArray
    DIR: str
    NOISE: Float[Ar, "512 512 20"]
    COLORCODE: Float[Ar, "20 3"]


###############################################################


class Diffusion_Tensor(eqx.Module):
    rgb: Float[Ar, "w h 3"]
    mg: Callable[[Float[Ar, "3"]], Tuple[Float[Ar, "3"], Float[Ar, "1"]]]

    def __init__(
        self,
        size: tuple[int, ...],
        key: PRNGKeyArray,
        metric_generator: Callable[
            [Float[Ar, "3"]], Tuple[Float[Ar, "3"], Float[Ar, "1"]]
        ],
        rgb: Float[Ar, "w h 3"] = None,
    ):
        if rgb is None:
            self.rgb = jax.random.normal(key, size) * 0.1
        else:
            self.rgb = rgb
        self.mg = metric_generator

    @eqx.filter_jit
    def __call__(self, x=0):  # variable x is unused
        V = self.rgb
        dt, scale = JV(JV(self.mg))(V)
        return dt, scale


class static_flow_module(eqx.Module):
    params: dict = field(
        default_factory=lambda: dict(mode="fast", t=3, msq=10, alpha=0)
    )

    def set_params(self, value: dict):
        """Change the parameters of the flow"""
        object.__setattr__(self, "params", value)

    def __call__(
        self, v: Float[Ar, "w h c"], mp: Diffusion_Tensor
    ) -> Float[Ar, "w h c"]:
        return sigmaflow_anisotropic_static(v, mp, **self.params)[-1]


class static_sigma_model(eqx.Module):
    DT: Diffusion_Tensor
    flow: static_flow_module

    def __call__(self, v: Float[Ar, "w h c"]) -> Float[Ar, "w h c"]:
        return self.flow(v, self.DT)


def rnd(x: Float[Ar, "... c"]) -> Int[Ar, "..."]:
    return x.argmax(-1)


def plot(ax: plt.Axes, x: Float[Ar, "w h *c"]):
    ax.imshow(x)
    ax.axis("off")


def anisotropy_index(metric: Float[Ar, "w h 3"]) -> Float[Ar, "w h"]:
    """
    calculate the anisoptropy index of given metric
    """
    a, c, b = jnp.split(metric, 3, axis=-1)
    Delta = jnp.sqrt((a - c) ** 2 + 4 * b**2)
    laambda = (jnp.concatenate([a + c + Delta, a + c - Delta], axis=-1)) * 0.5
    D = jnp.log(laambda)
    return 0.5 * np.sum(D**2, axis=-1) - D.prod(-1)


def plot_metric(m: static_sigma_model) -> plt.Figure:
    """
    plotting the anisotropy index and scale factor of a given model
    """
    # preprocessing
    dt, scale = m.DT()
    z = np.sqrt(anisotropy_index(dt)[..., np.newaxis])
    max = np.percentile(np.ravel(z), 85)
    im = jax.nn.relu(z - max) + max

    # plotting
    fig, ax = plt.subplots(1, 2)
    cax = ax[0].imshow(im, cmap="jet", norm=colors.PowerNorm(gamma=0.5))
    ax[0].axis("off")
    fig.colorbar(cax, pad=0.05, shrink=0.5)
    cax = ax[1].imshow(scale, cmap="jet_r", norm=colors.PowerNorm(gamma=1))
    ax[1].axis("off")
    fig.colorbar(cax, pad=0.05, shrink=0.5)
    plt.tight_layout(w_pad=0.9)
    return fig


def eval_model(C: Constants, m: static_sigma_model, gt: UInt8[Ar, "w h"]) -> plt.Figure:
    res = m(C.NOISE)
    im = np.where(
        (res.argmax(-1) != gt)[..., np.newaxis], (0, 0, 0), 0.6 * C.COLORCODE[gt]
    )
    fig, ax = plt.subplots()
    ax.imshow(im)
    ax.axis("off")
    plt.tight_layout(w_pad=0.9)
    return fig
