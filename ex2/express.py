import matplotlib.pyplot as plt
import argparse
import os

try:
    from jaxtyping import install_import_hook

    with install_import_hook("sigmaflow", "beartype.beartype"):
        from sigmaflow.exp2 import *
except:
    from sigmaflow.exp2 import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--capture", action="store_true")
capture = parser.parse_args().capture
figures = []

dn = os.path.dirname
dr = dn(dn(__file__)) + "/data"
key = jax.random.PRNGKey(250)
noise = jax.random.normal(key, (512, 512, 20))
noise = jnp.log(0.1 * np.eye(20)[noise.argmax(-1)] + 0.9 / 20.0)
noise /= 0.1 * np.linalg.norm(noise, axis=-1, keepdims=True)
C = Constants(
    DIR=dr,
    L_CELLS=np.load(dr + "/cells.npy"),
    L_BABOON=np.load(dr + "/baboon.npy"),
    KEY=key,
    NOISE=noise,
    COLORCODE=plt.cm.tab20(np.arange(20))[..., :-1],
)

# %% ===========================================================
mp = Diffusion_Tensor((512, 512, 3), C.KEY, metric_generator_cells)
fm = static_flow_module()
m = static_sigma_model(mp, fm)
m_cells = m
m_cells = eqx.tree_deserialise_leaves(C.DIR + "/cellsjax53.eqx", m)
m_cells.flow.set_params(dict(t=3.0, dt=0.2, msq=1.0, alpha=0.0, mode="adaptive"))

f1, axs = plt.subplots(1, 2)
figures.append(f1)
rect = patches.Rectangle(
    (100, 100), 64, 64, linewidth=1, edgecolor="r", facecolor="none"
)
figs = [C.COLORCODE[rnd(C.NOISE)], C.COLORCODE[rnd(C.NOISE)][:64, :64]]
[plot(ax, x) for x, ax in zip(figs, axs)]
axs[0].add_patch(rect)
rect = patches.Rectangle((0, 0), 63, 63, linewidth=1, edgecolor="r", facecolor="none")
axs[1].add_patch(rect)
plt.title("starter configuration")

# %% ===========================================================
f2, axs = plt.subplots(1, 1)
figures.append(f2)
plot(axs, C.COLORCODE[C.L_CELLS])
plt.title("target voronoi")

# %% ===========================================================
print(
    f"Number of mislabled pixels for the simple texture: {(m_cells(C.NOISE).argmax(-1) != C.L_CELLS).sum()}"
)

# %% ===========================================================
mp = Diffusion_Tensor((512, 512, 3), C.KEY, metric_generator_baboon)
fm = static_flow_module(dict(t=3.0, msq=1.0))
m = static_sigma_model(mp, fm)
m_bab = eqx.tree_deserialise_leaves(C.DIR + "/self_trained_baboon.eqx", m)

# %% ===========================================================
figures += [plt.figure(figsize=plt.figaspect(1))]
plt.imshow(C.L_BABOON, cmap="tab20")
plt.axis("off")
plt.title("target config for mandrill")

figures += [plt.figure(figsize=plt.figaspect(1))]
plt.imshow(m_bab(C.NOISE).argmax(-1), cmap="tab20")
plt.axis("off")
plt.title("labeling result for mandrill")

figures += [plt.figure(figsize=(5, 5))]
plt.imshow(m_bab(C.NOISE).argmax(-1) != C.L_BABOON, cmap="gray")
plt.axis("off")
plt.title("binary mask of mislabeled pixels")

print(
    f"Percentage of mislabeled pixels for mandrill target: {(m_bab(C.NOISE).argmax(-1) != C.L_BABOON).mean() * 100}"
)

# %% ===========================================================
if capture:
    path = dn(dn(__file__)) + "/artifacts"
    for i, f in enumerate(figures):
        f.savefig(path + f"/express{i}.png")
else:
    plt.show()
