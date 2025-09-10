import optax
import equinox as eqx
from tqdm import tqdm
import numpy as np
import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sigmaflow.exp2 import (
    Diffusion_Tensor,
    Constants,
    static_flow_module,
    static_sigma_model,
    metric_generator_cells,
    metric_generator_baboon,
)

ce = optax.softmax_cross_entropy_with_integer_labels


@eqx.filter_jit
def loss(model, X, GT):
    y = model(X)
    return ce(y, GT).mean()


def train(m, x, gt, optim, epochs, os=None):
    @eqx.filter_jit
    def train_step(model, x, gt, opt_state):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, gt)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, loss_value, opt_state

    if os is None:
        params = eqx.filter(m, eqx.is_array)
        os = optim.init(params)

    params = eqx.filter(m, eqx.is_array)
    os = optim.init(params)
    pbar = tqdm(range(epochs))
    for _ in pbar:
        X = x  # + 0.1 * np.random.randn(*x.shape)
        m, ls, os = train_step(m, X, gt, os)
        pbar.set_description(f"loss {ls:.5f}")
        pbar.refresh()

    return m, os


if __name__ == "__main__":
    import argparse
    import os

    dn = os.path.dirname
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--capture", action="store_true")
    capture: bool = parser.parse_args().capture
    path = dn(dn(__file__)) + "/artifacts"

    key = jax.random.PRNGKey(250)
    noise = jax.random.normal(key, (512, 512, 20))
    noise = jnp.log(0.1 * np.eye(20)[noise.argmax(-1)] + 0.9 / 20.0)
    noise /= 0.1 * np.linalg.norm(noise, axis=-1, keepdims=True)

    dr = dn(dn(__file__)) + "/data"
    C = Constants(
        DIR=dr,
        L_CELLS=np.load(dr + "/cells.npy"),
        L_BABOON=np.load(dr + "/baboon.npy"),
        KEY=key,
        NOISE=noise,
        COLORCODE=plt.cm.tab20(np.arange(20))[..., :-1],
    )

    # %% -----------------------------------------------------------
    mp = Diffusion_Tensor((512, 512, 3), C.KEY, metric_generator=metric_generator_cells)
    fm = static_flow_module(dict(t=3, msq=1, mode="fast", alpha=0))
    m = static_sigma_model(mp, fm)
    optim = optax.adabelief(optax.cosine_decay_schedule(0.05, 2000))
    m, os = train(m, C.NOISE, C.L_CELLS, optim, 2000)
    if capture:
        eqx.tree_serialise_leaves(path + "/self_trained_cells.eqx", m)

    # %% -----------------------------------------------------------
    mp = Diffusion_Tensor((512, 512, 3), key, metric_generator=metric_generator_baboon)
    fm = static_flow_module(dict(t=3, msq=1))
    m = static_sigma_model(mp, fm)
    optim = optax.adabelief(0.05)
    l = C.L_BABOON
    w = C.NOISE
    m, os = train(m, w, l, optim, 2000)
    if capture:
        eqx.tree_serialise_leaves("self_trained_baboon", m)
