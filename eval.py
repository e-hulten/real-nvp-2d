import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os
from scipy.stats import kde
from old_model import OldRealNVP
from model import RealNVP
from data_loader import Dataset


# ------------ parameters ------------
density = "moons"
n_c_layers = 32
path = r"/Users/edvardhulten/real_nvp_2d/"  # change to your own path (unless your name is Edvard Hultén too)
# ------------------------------------

if not os.path.exists("evals"):
    os.makedirs("evals")

model_ntnu = torch.load(path + "model_ntnu.pt")
model = torch.load(path + "model.pt")


gridspec = dict(wspace=0, width_ratios=[1, 0.1, 1, 1, 0.1, 1, 1])
fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(12, 3), gridspec_kw=gridspec)
data = Dataset(density)
x = data.generate_data(n_samples=10000)
axes[0].scatter(x[:, 0], x[:, 1], s=6, color="darkblue")
if density == "ntnu":
    axes[0].set_xlim(2.1, 8.2)
    axes[0].set_ylim(2.1, 8.2)
elif density == "moons":
    axes[0].set_xlim(-1.7, 2.6)
    axes[0].set_ylim(-2.1 + 0.225, 2.65 - 0.225)
axes[0].set_aspect(1)

x = data.generate_data(n_samples=2000)
sns.scatterplot(x[:, 0], x[:, 1], s=6, ax=axes[2], color="darkblue")
if density == "ntnu":
    axes[2].set_xlim(2.1, 8.2)
    axes[2].set_ylim(2.1, 8.2)
elif density == "moons":
    axes[2].set_xlim(-1.7, 2.6)
    axes[2].set_ylim(-2.1 + 0.225, 2.65 - 0.225)
axes[2].set_aspect(1)


# perform data -> noise mapping, should look like an isotropic Gaussian
z, _ = model(torch.from_numpy(x))
z = z.detach().numpy()

sns.scatterplot(z[:, 0], z[:, 1], s=6, color="darkred", ax=axes[3])
axes[3].set_xlim(-5, 5)
axes[3].set_ylim(-5, 5)
axes[3].set_aspect(1)


# generate standard Gaussian samples
z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 2000).astype(np.float32)
sns.scatterplot(z[:, 0], z[:, 1], s=6, color="darkblue", ax=axes[5])
axes[5].set_xlim(-5, 5)
axes[5].set_ylim(-5, 5)
axes[5].set_aspect(1)

# generate data from noise
x, _ = model.reverse(torch.from_numpy(z))
x = x.detach().numpy()
sns.scatterplot(x[:, 0], x[:, 1], s=6, color="darkred", ax=axes[6])
if density == "ntnu":
    axes[6].set_xlim(2.1, 8.2)
    axes[6].set_ylim(2.1, 8.2)
elif density == "moons":
    axes[6].set_xlim(-1.7, 2.6)
    axes[6].set_ylim(-2.1 + 0.225, 2.65 - 0.225)

for i in range(7):
    axes[i].set_xticklabels([])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_yticklabels([])
    axes[i].set_aspect(1)
    plt.tick_params(
        top="off",
        bottom="off",
        left="off",
        right="off",
        labelleft="off",
        labelbottom="off",
    )
    plt.tight_layout()

    # axes[i].set_frame_on(True)

plt.subplots_adjust(hspace=0.0)
axes[1].set_visible(False)
axes[4].set_visible(False)

plt.savefig("evals/" + "noise_to_data.png", bbox_inches="tight", pad_inches=0)
plt.savefig("evals/" + "noise_to_data.pdf", bbox_inches="tight", pad_inches=0, dpi=300)

# generate nice density plot
data = Dataset("ntnu")
orig_ntnu = data.generate_data(n_samples=10000)
data = Dataset("moons")
orig_moons = data.generate_data(n_samples=10000)

z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 2000).astype(np.float32)
x_ntnu, _ = model_ntnu.reverse(torch.from_numpy(z))
x_ntnu = x_ntnu.detach().numpy()
x_moons, _ = model.reverse(torch.from_numpy(z))
x_moons = x_moons.detach().numpy()


def nice_plot(x, ax, str="ntnu", cmap="Blues"):
    blues = cm.get_cmap(cmap, 512)
    nbins = 400
    k = kde.gaussian_kde([x[:, 0], x[:, 1]])
    xi, yi = np.mgrid[
        x[:, 0].min() - 1 : x[:, 0].max() + 1 : nbins * 1j,
        x[:, 1].min() - 1 : x[:, 1].max() + 1 : nbins * 1j,
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    ax.set_facecolor(blues(0))
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap, rasterized=True)
    if str == "ntnu":
        ax.set_xlim(2.0, 8.1)
        ax.set_ylim(2.0, 8.1)
    elif str == "moons":
        ax.set_xlim(-1.7, 2.6)
        ax.set_ylim(-2.1 + 0.225, 2.65 - 0.225)
    ax.set_aspect(1)
    return ax


gridspec = dict(wspace=0, width_ratios=[1, 1, 0.1, 1, 1])
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3), gridspec_kw=gridspec)

nice_plot(orig_ntnu, str="ntnu", ax=axes[3], cmap="Blues")
nice_plot(x_ntnu, str="ntnu", ax=axes[4], cmap="Reds")  # gist_heat_r
nice_plot(orig_moons, str="moons", ax=axes[0], cmap="Blues")
nice_plot(x_moons, str="moons", ax=axes[1], cmap="Reds")  # afmhot_r

for i in range(5):
    axes[i].set_xticklabels([])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_yticklabels([])
    axes[i].set_aspect(1)
    plt.tick_params(
        top="off",
        bottom="off",
        left="off",
        right="off",
        labelleft="off",
        labelbottom="off",
    )
    plt.tight_layout()
axes[2].set_visible(False)
plt.subplots_adjust(wspace=0.0)
plt.savefig("evals/" + "both" + ".png", bbox_inches="tight")
plt.savefig("evals/" + "both" + ".pdf", bbox_inches="tight", dpi=300)


"""
##################
model.eval()

x = data.generate_data(n_samples=1000)
z_norm = torch.distributions.Normal(0, 1).sample([1000, 2])

z, _ = model(torch.from_numpy(x))
z = z.detach().numpy()

np.random.seed(0)
x_n, _ = model.reverse(z_norm)
x_n = x_n.detach().numpy()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10),)
fig.subplots_adjust(left=0, right=1)

sns.scatterplot(
    x[:, 0], x[:, 1], ax=ax[0, 0], color="steelblue",
)
ax[0, 0].set_title(
    "Input: $\mathbf{{x}}\sim p_{{data}}(\mathbf{{x}})$, $n = $ {}".format(x.shape[0]),
    size=14,
)
ax[0, 0].set_xlim(-2.2, 3)
ax[0, 0].set_ylim(-2.3, 2.9)
ax[0, 0].set_aspect(1)

sns.scatterplot(z[:, 0], z[:, 1], ax=ax[0, 1], color="darkred")
ax[0, 1].set_title(
    r"Output: $\mathbf{{u}} = f^{-1}(\mathbf{{x}})$" + ", $n = $ {}".format(x.shape[0]),
    size=14,
)
ax[0, 1].set_xlim(-4, 4)
ax[0, 1].set_ylim(-4, 4)
ax[0, 1].set_aspect(1)


sns.scatterplot(
    z_norm[:, 0], z_norm[:, 1], ax=ax[1, 0], color="steelblue",
)
ax[1, 0].set_title(
    "Input: $\mathbf{{u}}\sim\mathcal{{N}}(0,I)$, $n = $ {}".format(z_norm.shape[0]),
    size=14,
)
ax[1, 0].set_xlim(-4, 4)
ax[1, 0].set_ylim(-4, 4)
ax[1, 0].set_aspect(1)


sns.scatterplot(x_n[:, 0], x_n[:, 1], ax=ax[1, 1], color="darkred")
ax[1, 1].set_title(
    "Output: $\mathbf{{x}} = f(\mathbf{{u}})$, $n = $ {}".format(x_n.shape[0]), size=14
)
ax[1, 1].set_xlim(-2.2, 3)
ax[1, 1].set_ylim(-2.3, 2.9)
ax[1, 1].set_aspect(1)

plt.suptitle("", size=18)

for (m, n), subplot in np.ndenumerate(ax):
    subplot.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )

plt.savefig(
    "results/" + "result_square.pdf", bbox_inches="tight", pad_inches=0.2, dpi=300
)
plt.close()

##################
model.eval()

x = data.generate_data(n_samples=1000)
z_norm = torch.distributions.Normal(0, 1).sample([1000, 2])

z, _ = model(torch.from_numpy(x))
z = z.detach().numpy()

np.random.seed(0)
x_n, _ = model.reverse(z_norm)
x_n = x_n.detach().numpy()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 8),)
fig.subplots_adjust(left=0, right=1)

sns.scatterplot(
    x[:, 0], x[:, 1], ax=ax[0, 0], color="steelblue",
)

ax[0, 0].set_xlim(-1.7, 2.9)
ax[0, 0].set_ylim(-1.6, 2.2)
ax[0, 0].set_aspect(1)

sns.scatterplot(z[:, 0], z[:, 1], ax=ax[0, 1], color="darkred")

ax[0, 1].set_xlim(-4, 4)
ax[0, 1].set_ylim(-4, 4)
ax[0, 1].set_aspect(1)


sns.scatterplot(
    z_norm[:, 0], z_norm[:, 1], ax=ax[1, 0], color="steelblue",
)

ax[1, 0].set_xlim(-4, 4)
ax[1, 0].set_ylim(-4, 4)
ax[1, 0].set_aspect(1)


sns.scatterplot(x_n[:, 0], x_n[:, 1], ax=ax[1, 1], color="darkred")

ax[1, 1].set_xlim(-1.7, 2.9)
ax[1, 1].set_ylim(-1.6, 2.2)
ax[1, 1].set_aspect(1)

for (m, n), subplot in np.ndenumerate(ax):
    subplot.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    subplot.set_frame_on(False)

plt.savefig(
    "results/" + "result_square2.pdf", bbox_inches="tight", pad_inches=0.2, dpi=300
)
plt.close()
"""

