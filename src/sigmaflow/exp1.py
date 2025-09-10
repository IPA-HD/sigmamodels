from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from itertools import product

import jax
import jax.numpy as jnp
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import numpy as np
import diffrax as dx
from einops import rearrange
from jaxtyping import Array, Float, Float64, jaxtyped
from scipy.special import softmax

from .flow import sigmaflow, Pi_0

########################## CONSTANTS ##########################
AR = Array | np.ndarray  # ty: ignore


@dataclass
class coordinates:
    CORNERS: Float[AR, "3 4"]
    TRI: mtri.Triangulation
    U: Float[AR, "23 23"]
    V: Float[AR, "23 23"]


###############################################################


def mean_entropy(
    p: Float[AR, "... c"],
    agg: Callable[[Float[AR, "..."]], Float[AR, ""]] = jnp.mean,
) -> Float[AR, ""]:
    c = float(p.shape[-1])
    return agg((-jax.scipy.special.entr(p)).sum(-1) + jnp.log(c)) / jnp.log(c)


def plot_figures(
    axs: list[plt.Axes],
    figures: list[AR],
    pf: Callable[[plt.Axes, AR], None],
    **kwargs,
) -> None:
    reduce(lambda x, y: pf(y[0], y[1], **kwargs), zip(axs, figures), 0)


def segre() -> Float[AR, "4 20 20"]:
    p = np.linspace(0, 1, 20)
    q = np.linspace(0, 1, 20)
    pq = np.dstack([np.outer(x, y) for x, y in product([1 - p, p], [1 - q, q])])
    pq = pq.swapaxes(-1, 0)
    return pq


def plot_simplex(C: coordinates, ax: plt.Axes, alpha: float = 0.1, c: str = "grey"):
    triangles = [(0, 1, 2), (1, 2, 3), (0, 1, 3), (0, 2, 3)]
    ax.plot_trisurf(
        *C.CORNERS,
        triangles=triangles,
        edgecolors="k",
        lw=0.6,
        alpha=alpha,
        color=c,
    )


def sm(x: Float[AR, "..."]) -> Float[AR, "..."]:
    """
    softmax along last dimension
    """
    return softmax(x, axis=-1)


def torus_20(C: coordinates) -> Float[AR, "4 23 23"]:
    """
    generate torus in 4d tangent space to simplex, with 20 samples along each dimension
    """
    x = (3 + (np.cos(C.V))) * np.cos(C.U) * 0.2
    y = (3 + (np.cos(C.V))) * np.sin(C.U) * 0.2
    z = np.sin(C.V) * 0.2
    theta = np.stack([x, y, z])
    v = np.stack([x, y, z, theta.sum(0)], axis=0)
    return softmax(v, axis=0)


def torus_80() -> Float[AR, "4 82 82"]:
    """
    generate torus in 4d tangent space to simplex, with 80 samples along each dimension
    """
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, 2 * np.pi, 80)
    u, v = np.meshgrid(u, v)
    u, v = map(lambda xx: np.pad(xx, 1, mode="wrap"), (u, v))
    # _ = mtri.Triangulation(u.ravel(), v.ravel())
    x = (3 + (np.cos(v))) * np.cos(u) * 0.2
    y = (3 + (np.cos(v))) * np.sin(u) * 0.2
    z = np.sin(v) * 0.2
    theta = np.stack([x, y, z])
    v = np.stack([x, y, z, theta.sum(0)], axis=0)
    # V += np.random.randn(*V.shape) * 0.05
    return v


def plot_trajectories(
    C: coordinates, p: Float[AR, "t 3 w h"], ax: plt.Axes, az: float, elev: float = 15
):
    """
    plot particle trajectories through 3 dimensional simplex
    """
    ax.plot_surface(*p[0])
    for pp in rearrange(p, "t d w h -> (w h) d t")[::2]:
        x, y, z = pp
        ax.plot(x, y, z, "-", alpha=0.5, linewidth=0.4, c="g")
    plot_simplex(C, ax)
    ax.view_init(azim=az, elev=elev)
    ax.set_axis_off()


def plot_p(X: Float[AR, "3 w h"], ax: plt.Axes) -> None:
    """
    plot a rasterized surface
    """
    x, y, z = X
    x, y, z = map(lambda xx: np.pad(xx, 1, mode="wrap"), (x, y, z))
    ax.plot_surface(x, y, z)
    ax.view_init(azim=25)
    ax.set_xlim(0.1, 0.4)
    ax.set_ylim(0.1, 0.4)
    ax.set_zlim(0.1, 0.4)
    ax.set_axis_off()


def gen_v0(C: coordinates) -> Float[AR, "23 23 4"]:
    """
    generate a torus in the tangent space
    """
    p = torus_20(C)
    v0 = np.log(p)
    v0 = Pi_0(v0.swapaxes(0, -1))
    return v0
