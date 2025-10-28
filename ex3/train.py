import argparse
import datetime
import json
import os as os
import subprocess
import uuid
from typing import TextIO
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sigmaflow.layers import sigmalayers, sigmasimple
from sigmaflow.unet import unet
from sigmaflow.flow import laplacian
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from PIL import Image
from tqdm import tqdm


@eqx.filter_jit
def loss(
    model,
    features: Float[Array, "b w h c"],
    labels: Int[Array, "b w h"],
    weight: float,
    pad: int,
) -> Float[Array, "1"]:
    y = jax.vmap(model)(features)
    kl = optax.softmax_cross_entropy_with_integer_labels(y, labels) / jnp.log(
        features.shape[-1]
    )
    return jnp.mean(kl)


def train(
    model,
    data: tuple[int, Array],
    optim: optax.GradientTransformationExtraArgs,
    opt_state: optax.OptState,
    logfile: str,
    weight: float,
    bs: int,
    size: int,
    num_iter: int,
    ks: int,
    nl: int,
    dt: str,
):
    pad = int(ks - 1) // 2
    pad *= nl

    @eqx.filter_jit
    def train_step(model, features, labels, opt_state: optax.OptState):
        loss_value, grads = eqx.filter_value_and_grad(loss)(
            model, features, labels, weight, pad
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, loss_value, opt_state

    nl = np.max(labels) + 1
    w = np.min(labels.shape[1:])
    if dt == "voronoi":
        nk = 501
    else:
        nk = 2

    @jax.vmap
    def ret(n):
        i, j = np.random.randint(0, w - size, 2)
        k = np.random.randint(0, nk)
        l = labels[k, i : i + size, j : j + size]
        rs = np.log(np.eye(nl)[l] * 0.2 + 0.8)
        rs += np.random.randn(*rs.shape) * 0.2
        rs = (rs - rs.min(-1, keepdims=True)) / (
            rs.max(-1, keepdims=True) - rs.min(-1, keepdims=True)
        )
        return rs, l

    pbar = range(num_iter)
    losses = []
    t0 = time.time()
    told = t0
    tnew = t0
    for i, batch in enumerate(pbar):
        f, l = ret(jnp.empty(bs))
        model, ls, opt_state = train_step(model, f, l, opt_state)
        losses.append(ls.item())
        if i % 100 == 0:
            tnew = time.time()
            tt = int((num_iter - i) * (tnew - told) / 100)
            ttm = tt / 60
            delta = tnew - t0
            told = tnew
            perf = np.mean(jax.vmap(model)(f).argmax(-1) == l)
            print(
                f"iteration: {i:8d}/{num_iter} | time: {delta:8.2f}s / {ttm:4.1f} min remaining | loss: {ls.item():.5f} | acc: {perf:.4f}"
            )

    if capture:
        np.savetxt(logfile, losses)
    return model, opt_state, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--capture", action="store_true")
    parser.add_argument("-m", "--model", type=str, default="sigmaflowv")
    parser.add_argument("-d", "--data", type=str, default="voronoi")
    args = parser.parse_args()
    capture = args.capture
    model = args.model
    data = args.data

    dn = os.path.dirname
    np.random.seed(131823718)
    hps = json.load(open(dn(dn(__file__)) + f"/data/{model}/hps.json"))
    hps["capture"] = capture

    hps["date"] = str(datetime.datetime.now())
    print(json.dumps(hps, sort_keys=True, indent=2))

    path = hps["path"]
    capture = hps["capture"]
    path = dn(dn(__file__)) + "/data/"
    logdir = path + f"/{uuid.uuid1()}"
    logfile = logdir + "/epoch" + str(0) + "_loss.csv"
    checkpoint = logdir + "/epoch" + str(0) + ".eqx"
    m = eval(hps["model"])(**hps)

    optimizer = eval(f"optax.{hps['optim']}")
    opt_state: optax.OptState = optimizer.init(eqx.filter(m, eqx.is_array))

    if data == "voronoi":
        labels = np.load(f"{path}/voronoi.npz")["arr_0"]
    elif data == "house":
        ds = np.load(f"{path}/rgb_capped_labeled.npz")
        labels = ds["l"]
    else:
        raise ValueError(
            f"Can only operate on data = voronoi or data = house, got data = {data}"
        )

    m, opt_state, losses = train(
        m,
        (0, labels),
        optimizer,
        opt_state,
        logfile,
        weight=hps["weight"],
        bs=hps["batch_size"],
        size=hps["size"],
        num_iter=hps["num_iter"],
        ks=hps["ks"],
        nl=hps["nl"],
        dt=data,
    )
    if capture:
        hps["mean"] = np.array(losses)[-100:].mean().item()
        hps["uuid"] = logdir
        os.makedirs(logdir)
        with open(logdir + "/hps.json", "a") as f:
            json.dump(hps, f)
            eqx.tree_serialise_leaves(checkpoint, m)
