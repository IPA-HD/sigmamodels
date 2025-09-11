import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import sys
import os
from sigmaflow.aniso import (
    metric_network,
    make_x0,
    CellData,
    conversion_best_experiment,
    flow_module,
    n_sigma_model,
    aniso_score,
    base_m,
)
from optax import softmax_cross_entropy_with_integer_labels as ce


def gen_ax(n):
    return plt.subplots(1, n, figsize=(5 * int(n), 5))[1]


def plot(figs):
    return [
        ax.imshow(x) and ax.set_axis_off() for ax, x in zip(gen_ax(len(figs)), figs)
    ]


def eval_model(model, inp, ref=base_m, P=plt.cm.tab20(np.arange(0, 20))[..., :-1]):
    x, gt = inp
    sigma = ref(x)
    metric = model(x)
    print(
        f"Sigmaflow: {ce(sigma, gt)[10:-10, 10:-10].mean():.4f}, Metric {ce(metric, gt)[10:-10, 10:-10].mean():.4f}"
    )
    cc = ce(sigma, gt)[..., np.newaxis]
    cc = (cc - cc.min()) / (cc.max() - cc.min())
    cc = cc ** (0.3)
    im1 = P[gt] * (1 - cc)

    cc = ce(metric, gt)[..., np.newaxis]
    cc = (cc - cc.min()) / (cc.max() - cc.min())
    cc = cc ** (0.3)
    im2 = P[gt] * (1 - cc)

    plot([im1, im2, P[gt]])


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--capture", action="store_true")
args = parser.parse_args()
capture: bool = args.capture  # whether to save artifacts


print("...loading dataset...")
dn = os.path.dirname
pth = dn(dn(__file__)) + "/data"
batch_air = np.load(pth + "/airplane.npy")[10:-10, 10:-10][None, ..., None]
ite = CellData()
bp = ite[np.random.randint(0, len(ite))][..., None]

print("...loading model...")
key = jax.random.key(371283789)
mp = metric_network(key, 20, converter=conversion_best_experiment)
fm = flow_module(2, 4, alpha=0)
m = n_sigma_model(mp, fm)
m = eqx.tree_deserialise_leaves(f"{pth}/full.eqx", m)
# mn, n, a, f = extract_from_folder(path + "/models/15_08/17_48/", m)


## plot53a
f1, axs = plt.subplots(1, 4, figsize=(20, 5))
# ite = iter(train_dataloader)
# ite = bp
for a, f in zip(axs, ite):
    a.imshow(plt.cm.tab20(f[0, :, :]))
    a.axis("off")
plt.tight_layout(pad=1)

## plot 53b
P = plt.cm.tab20(range(20))[..., :-1]
x, gt = make_x0(bp, 0.2, 0.8)
plt.figure(figsize=(5, 5))
plt.imshow(P[x.argmax(-1)])
plt.axis("off")
plt.tight_layout(pad=0)
f2 = plt.gcf()

## plot 53c
inp = make_x0(bp, 0.2, 0.8)  # alpha=0.1,filter=uf(5))
x, gt = inp
eval_model(m, inp, base_m)
plt.tight_layout(w_pad=0.5)
f3 = plt.gcf()

## plot 53d
inp = make_x0(batch_air, 0.2, 0.8)  # alpha=0.1,filter=uf(5))
x, gt = inp
eval_model(m, inp, base_m)
plt.tight_layout(w_pad=0.5)
plt.axis("off")
f4 = plt.gcf()

## plot 53e
print("...plotting anisotropy...")
x, gt = make_x0(batch_air, 0.0, 0.8)
z = np.array([np.sqrt(aniso_score(m, x, t)) for t in np.arange(0, 1, 0.1)]).mean(0)[
    10:-10, 10:-10
]
# z = np.maximum(z - np.percentile(z, 70),0)
f5, ax = plt.subplots(dpi=100)
bar = ax.imshow(z, cmap="rainbow", norm=colors.PowerNorm(1), alpha=1)
plt.colorbar(bar, fraction=0.03, pad=0.01, shrink=0.7)
plt.axis("off")
plt.tight_layout(w_pad=1)

## plot 53f
print("...plotting scale...")
f6, ax = plt.subplots(dpi=100)
z = np.array([np.reciprocal(m.mp(x, t)[1]) for t in np.arange(0, 1, 0.1)]).mean(0)[
    10:-10, 10:-10
]
z = 1 / z
bar = ax.imshow(z, cmap="rainbow", norm=colors.PowerNorm(1), alpha=1)
plt.colorbar(bar, fraction=0.03, pad=0.01, shrink=0.7)
plt.axis("off")
plt.tight_layout(w_pad=1)

if capture:
    path = dn(dn(__file__)) + "/artifacts"
    for i, f in enumerate([f1, f2, f3, f4, f5, f6]):
        f.savefig(path + f"/legacy{i}.png")
else:
    plt.show()
