from sigmaflow.exp1 import *
import argparse

plt.rcParams["text.usetex"] = True
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--capture", action="store_true")
capture = parser.parse_args().capture

u = np.append(np.linspace(0, 2 * np.pi, 20), 0)
v = np.append(np.linspace(0, 2 * np.pi, 20), 0)
u, v = np.meshgrid(u, v)
u, v = map(lambda xx: np.pad(xx, 1, mode="wrap"), (u, v))

C = coordinates(
    U=u,
    V=v,
    TRI=mtri.Triangulation(u.ravel(), v.ravel()),
    CORNERS=np.vstack([np.zeros(3), np.eye(3)]).T,
)

f1, axs = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
plot_simplex(C, axs)
p = torus_20(C)
axs.plot_surface(*p[:-1])
axs.view_init(azim=25, elev=20)
axs.set_xticks([0, 0.5, 1])
axs.set_yticks([0, 0.5, 1])
axs.set_zticks([0, 0.5, 1])

# %% -----------------------------------------------------------
f2, axs = plt.subplots(1, 4, subplot_kw=dict(projection="3d"), figsize=(35, 15))
axit = iter(axs)
v0 = gen_v0(C)
v1 = sigmaflow(v0, 80, m=0)
p = sm(v1)[..., :-1]
p = p.swapaxes(-1, 1)
for X, ax in zip(p[(0, 1, 2, 3), :, :], axit):
    plot_p(X, ax)
f2.tight_layout(pad=0.1, w_pad=0.0)

# %% -----------------------------------------------------------
v = torus_80()
v0 = v.swapaxes(0, -1)
v1 = sigmaflow(v0, 5, m=1, alpha=0, solver=dx.Dopri5())
p = sm(v1)[..., :-1]
p = p.swapaxes(-1, 1)

f3, axs = plt.subplots(1, 6, figsize=(50, 10), subplot_kw=dict(projection="3d"))
ax = axs[0]
plot_simplex(C, ax)
ax.plot_surface(*p[0])
ax.view_init(azim=25)

for X, ax in zip(p[::2], axs.flat[1:]):
    ax.scatter(*X)
    ax.view_init(azim=25)
    plot_simplex(C, ax)
    ax.set_axis_off()
f3.tight_layout(pad=0.1, w_pad=0.0)

# %% -----------------------------------------------------------
f4, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"), layout="tight")
plot_trajectories(C, p, ax, -15.0, 20.0)

# %% -----------------------------------------------------------
f5, ax = plt.subplots(subplot_kw=dict(projection="3d"))
plot_simplex(C, ax)
pq = segre()
col = jax.scipy.special.entr(pq).sum(0)
col = (col - col.min()) / (col.max() - col.min())
ax.plot_surface(*pq[1:], lw=2.0, alpha=0.7, linewidth=1, color="orange")
p = torus_20(C)
ax.view_init(azim=-15, elev=20)
ax.axis("off")

# %% -----------------------------------------------------------
ind = iter((1, 2))
for m in (0, 0.1):
    if m == 0:
        f6, ax = plt.subplots(1, 1)
    else:
        f7, ax = plt.subplots(1, 1)
    for a in (-1, 0, 1):
        v0 = 5 * v.swapaxes(0, -1)
        v1 = sigmaflow(v0, 20, m=m, alpha=a)
        p = sm(v1)
        ax.plot(
            np.linspace(0, 20, 10),
            jax.vmap(mean_entropy)(p),
            label=f"$\\alpha$ = {a:2d}",
        )
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.legend(fontsize=20)
    ax.grid(True)
    ax.set_xlabel("$t$", fontsize=20)
    ax.set_ylabel("$\\tilde\\varphi$", fontsize=20)
    plt.tight_layout()

if capture:
    import os

    dn = os.path.dirname
    pth = dn(dn(__file__)) + "/artifacts"
    for i, f in enumerate([f1, f2, f3, f4, f5, f6, f7]):
        f.savefig(pth + f"/simplex{i}.png")
else:
    plt.show()
