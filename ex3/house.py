import json
import os
import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

try:
    from jaxtyping import install_import_hook

    with install_import_hook("sigmaflow", "beartype.beartype"):
        from sigmaflow.layers import sigmasimple
        from sigmaflow.unet import unet
except:
    from sigmaflow.layers import sigmasimple
    from sigmaflow.unet import unet

plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"font.family": "libertinus"})
plt.rcParams.update({"font.family": "bold"})
dn = os.path.dirname
figs = []

if __name__ != "__main__":
    __file__ = os.path.abspath("ex4/experiment5.py")

pth = dn(dn(__file__)) + "/data"
hp = json.load(open(pth + "/sigmaflowh/hps.json"))
print(json.dumps(hp, indent=2))
m = sigmasimple(**hp)
sm = eqx.tree_deserialise_leaves(pth + "/sigmaflowh/model.eqx", m)

hp = json.load(open(pth + "/uneth/hps.json"))
print(json.dumps(hp, indent=2))
m = unet(**hp)
mu = eqx.tree_deserialise_leaves(pth + "/uneth/model.eqx", m)

ds = np.load(pth + "/rgb_capped_labeled.npz")
features = ds["f"]
labels = ds["l"]
size = 128
ell = lambda x: x.argmax(-1)
w = np.min(labels[0].shape)
nl = np.max(labels) + 1


def ret():
    i, j = np.random.randint(0, w - size, 2)
    k = np.random.randint(0, 2)
    l = labels[k, i : i + size, j : j + size]
    rs = np.log(np.eye(nl)[l] * 0.2 + 0.8)
    rs += np.random.randn(*rs.shape) * 0.2
    rs = jnp.array(rs)
    rs = (rs - rs.min(-1, keepdims=True)) / (
        rs.max(-1, keepdims=True) - rs.min(-1, keepdims=True)
    )
    return rs, l


rr, ll = zip(*[ret() for _ in range(32)])
rr = jnp.array(rr)
ll = jnp.array(ll)
perf1 = np.mean(ell(jax.vmap(sm)(rr)) == ll)
perf2 = np.mean(ell(jax.vmap(mu)(rr)) == ll)
print(f"Train perf house fronts Sigmaflow {perf1}, UNet {perf2}")

np.random.seed(48913794)
rr, l = ret()
figs += [plt.figure(figsize=plt.figaspect(1 / 4.0))]
plt.subplot(141).imshow(ell(rr), cmap="tab20", vmin=0, vmax=8)
plt.title("Input")
plt.axis("off")
plt.subplot(142).imshow(l, cmap="tab20", vmin=0, vmax=8)
plt.title("Target")
plt.axis("off")
plt.subplot(143).imshow(ell(sm(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("Sigmaflow")
plt.axis("off")
plt.subplot(144).imshow(ell(mu(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("UNet")
plt.axis("off")
plt.tight_layout(pad=0.5)

np.random.seed(57285479)
_, ll = zip(*[ret() for _ in range(3)])
l = np.array(ll)
figs += [plt.figure(figsize=plt.figaspect(1 / 3.0), layout="tight")]
plt.subplot(131).imshow(ll[0], cmap="tab20", vmin=0, vmax=8)
plt.axis("off")
plt.subplot(132).imshow(ll[1], cmap="tab20", vmin=0, vmax=8)
plt.axis("off")
plt.subplot(133).imshow(ll[2], cmap="tab20", vmin=0, vmax=8)
plt.axis("off")
plt.tight_layout(pad=0.5)

ll = labels[0:2]
rr = np.log(np.eye(nl)[ll] * 0.2 + 0.8)
rr += np.random.randn(*rr.shape) * 0.2
rr = jnp.array(rr)
rr3 = rr / jnp.linalg.norm(rr, axis=-1, keepdims=True)
rr2 = (rr - rr.min(-1, keepdims=True)) / (
    rr.max(-1, keepdims=True) - rr.min(-1, keepdims=True)
)
print(f"""
Performance sigma flow test large house seen unit cube normalization   {np.mean(ell(jax.vmap(sm)(rr2)) == ll)}
Performance UNet       test large house seen unit cube normalization   {np.mean(ell(jax.vmap(mu)(rr2)) == ll)}
Performance sigma flow test large house seen unit sphere normalization {np.mean(ell(jax.vmap(sm)(rr3)) == ll)}
Performance UNet       test large house seen unit sphere normalization {np.mean(ell(jax.vmap(mu)(rr3)) == ll)}
""")

ll = labels[1]
rr = np.log(np.eye(nl)[ll] * 0.2 + 0.8)
rr += np.random.randn(*rr.shape) * 0.2
rr /= jnp.linalg.norm(rr, axis=-1, keepdims=True)
figs += [plt.figure(figsize=plt.figaspect(1 / 4.0))]
plt.subplot(141).imshow(ell(rr), cmap="tab20", vmin=0, vmax=8)
plt.title("Input")
plt.axis("off")
plt.subplot(142).imshow(ll, cmap="tab20", vmin=0, vmax=8)
plt.title("Target")
plt.axis("off")
plt.subplot(143).imshow(ell(sm(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("Sigma Flow")
plt.axis("off")
plt.subplot(144).imshow(ell(mu(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("UNet")
plt.axis("off")
plt.tight_layout(pad=0.5)

ll = labels[2]
rr = np.log(np.eye(nl)[ll] * 0.2 + 0.8)
rr += np.random.randn(*rr.shape) * 0.2
rr = jnp.array(rr)
rr3 = rr / jnp.linalg.norm(rr, axis=-1, keepdims=True)
rr2 = (rr - rr.min(-1, keepdims=True)) / (
    rr.max(-1, keepdims=True) - rr.min(-1, keepdims=True)
)
print(f"""
Performance sigma flow test large house unseen unit cube normalization   {np.mean(ell(sm(rr2)) == ll)}
Performance UNet       test large house unseen unit cube normalization   {np.mean(ell(mu(rr2)) == ll)}
Performance sigma flow test large house unseen unit sphere normalization {np.mean(ell(sm(rr3)) == ll)}
Performance UNet       test large house unseen unit sphere normalization {np.mean(ell(mu(rr3)) == ll)}
""")

ll = labels[2]
rr = np.log(np.eye(nl)[ll] * 0.2 + 0.8)
rr += np.random.randn(*rr.shape) * 0.2
rr /= jnp.linalg.norm(rr, axis=-1, keepdims=True)
figs += [plt.figure(figsize=plt.figaspect(1 / 4.0))]
plt.subplot(141).imshow(ell(rr), cmap="tab20", vmin=0, vmax=8)
plt.title("Input")
plt.axis("off")
plt.subplot(142).imshow(ll, cmap="tab20", vmin=0, vmax=8)
plt.title("Target")
plt.axis("off")
plt.subplot(143).imshow(ell(sm(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("Sigma Flow")
plt.axis("off")
plt.subplot(144).imshow(ell(mu(rr)), cmap="tab20", vmin=0, vmax=8)
plt.title("UNet")
plt.axis("off")
plt.tight_layout(pad=0.5)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--capture", action="store_true")
capture = parser.parse_args().capture
if capture:
    dr = dn(dn(__file__)) + "/artifacts"
    for i, f in enumerate(figs):
        f.savefig(f"{dr}/house{i}")

else:
    plt.show()
