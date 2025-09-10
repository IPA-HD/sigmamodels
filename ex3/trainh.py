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
    return jnp.mean(kl)  # "[:, pad:-pad, pad:-pad])
    return jnp.mean((kl * (1 + weight * edges))[:, pad:-pad, pad:-pad])


def train(
    model,
    data: Array,
    optim: optax.GradientTransformationExtraArgs,
    opt_state: optax.OptState,
    logfile: str,
    weight: float,
    bs: int,
    size: int,
    num_iter: int,
    ks: int,
    nl: int,
):
    # dim = model.mp.conv.in_channels

    # pad = int(model.mp.conv.weight.shape[-1] - 1) // 2
    pad = int(ks - 1) // 2
    pad *= nl

    @eqx.filter_jit
    def train_step(model, features, labels, opt_state: optax.OptState):
        # gt = rearrange(gt, "(x w) (y h) -> (x y) w h", x=bs, y=bs)
        # x = jax.vmap(partial(corrupt, dim=dim, noise=noise, alpha=alpha))(gt)
        loss_value, grads = eqx.filter_value_and_grad(loss)(
            model, features, labels, weight, pad
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, loss_value, opt_state

    # features, labels = data
    w = np.min(labels[0].shape)
    nl = np.max(labels) + 1
    # rs = np.log(np.eye(nl)[labels] * 0.2 + 0.8)
    # rs += np.random.randn(*rs.shape) * 0.2
    # rs = (rs - rs.min(-1, keepdims=True)) / (
    #     rs.max(-1, keepdims=True) - rs.min(-1, keepdims=True)
    # )
    # features = rs

    @jax.vmap
    def ret(n):
        i, j = np.random.randint(0, w - size, 2)
        k = np.random.randint(0, 1)
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
        # f = features[N]
        # l = labels[None]
        # f = 1 * jnp.vstack(f)
        # l = jnp.vstack(l)
        # f = (f - f.mean(-1, keepdims=True)) / (f.std(-1, keepdims=True) + 1e-6)
        # f /= 1 * jnp.linalg.norm(f, axis=-1, keepdims=True)
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
            # pbar.set_description(f"loss: {ls.item():.5f}")
            # pbar.refresh()

    if capture:
        np.savetxt(logfile, losses)
    return model, opt_state, losses


def sync_git():
    if len(s := subprocess.check_output(["git", "diff", "--stat"])) > 0:
        print("won't run on dirty git tree")
        print(f"{s.decode()}")
        msg = f"backup for run {time.strftime('%d.%m.%y-%H:%M', time.gmtime(time.time() + 7200))}"
        subprocess.run(["git", "commit", "-am", msg])
        subprocess.run(["git", "push"])


def hparams(parser):
    # parser.add_argument("-A", "--alpha_sigma", type=float, default=0)
    parser.add_argument("-I", "--num_iter", type=int, default=500)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-c", "--capture", action="store_true")
    parser.add_argument("-g", "--git_track", action="store_true")
    parser.add_argument("-k", "--ks", type=int, default=15)
    # parser.add_argument("-l", "--lr", type=float, default=1e-4)
    parser.add_argument("-m", "--mass", type=float, default=1)
    parser.add_argument("-d", "--dim1", type=int, default=9)
    parser.add_argument("-D", "--dim2", type=int, default=32)
    parser.add_argument("-S", "--scale", type=float, default=0.1)
    parser.add_argument("-o", "--optim", type=str, default="adabelief(1e-4)")
    parser.add_argument("-p", "--path", type=str, default="artifacts")
    parser.add_argument("-r", "--resumeId", type=int, default=None)
    parser.add_argument("-s", "--size", type=int, default=128)
    parser.add_argument("-n", "--nl", type=int, default=2)
    parser.add_argument("-w", "--weight", type=float, default=0)
    parser.add_argument("-M", "--model", type=str, default="sigmalayers")
    parser.add_argument("-q", "--type", type=str, default="voronoi-house")


if __name__ == "__main__":
    np.random.seed(131823718)
    parser = argparse.ArgumentParser()
    hparams(parser)
    args = parser.parse_args()
    hps = vars(args)

    if args.git_track:
        sync_git()

    hps["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    hps["date"] = str(datetime.datetime.now())
    print(pd.DataFrame([hps]))
    # print(json.dumps(hps, sort_keys=True, indent=2))

    dir: str
    path: str = hps["path"]
    capture: bool = hps["capture"]
    # optimizer: optax.GradientTransformationExtraArgs = eval(f"optax.{hps['optim']}")(
    #     hps["lr"]
    # )
    logdir: str = path + f"/ex5/{uuid.uuid1()}"
    logfile: str = logdir + "/epoch" + str(0) + "_loss.csv"
    checkpoint: str = logdir + "/epoch" + str(0) + ".eqx"

    # key: PRNGKeyArray = jax.random.key(3123123)
    # mp: metric_network = metric_network(key, ks=hps["ks"])
    # fm: flow_module = flow_module(
    #     t=hps["t"], m=hps["m"], alpha=hps["alpha_sigma"], dt=hps["dt"]
    # )
    # m: sigma_module = init_sigma(0.01, fm, mp)
    # m = sigmalayers(**hps)
    m = eval(hps["model"])(**hps)

    optimizer = eval(f"optax.{hps['optim']}")
    # optimizer = optax.adabelief(hps["lr"])
    opt_state: optax.OptState = optimizer.init(eqx.filter(m, eqx.is_array))

    if args.resumeId:
        ...
        # from handler import load_eqx, PATHS
        # m = load_eqx(dim, idx=args.resumeId)

    # labels = np.load(dir + f"/labels.npz")
    # labels = jnp.array([labels["l1"], labels["l3"]])
    # features = np.load(dir + f"/rgb.npz")
    # features = jnp.array([features["l1"], features["l3"]])

    ds = np.load(f"{dir}/rgb_capped_labeled.npz")
    labels = ds["l"]
    # labels = np.load(f"{dir}/voronoi.npz")["arr_0"]

    # optimizer = optax.lion(hps["lr"], weight_decay=5e-3)
    # optimizer = optax.adam(hps["lr"])

    if capture:
        os.makedirs(logdir)
        with open(logdir + "/hps.json", "a") as f:
            json.dump(hps, f)

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
    )
    if capture:
        eqx.tree_serialise_leaves(checkpoint, m)

        hps["mean"] = np.array(losses)[-100:].mean().item()
        hps["uuid"] = logdir
        row = pd.DataFrame([hps])
        path = PATHS[5]
        df = pd.read_csv(path + "/summaryv2.csv", index_col=0)
        df = pd.concat([df, row], ignore_index=True)
        print(df)
        df.to_csv(path + "/summaryv2.csv")
