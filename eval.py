import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import os
from scipy.stats import kde
from model import RealNVP
from data_loader import Dataset

# ------------ parameters ------------
density = "moons"
n_c_layers = 12
path = r"/Users/edvardhulten/real_nvp_2d/"  # change to your own path (unless your name is Edvard Hultén too)
# ------------------------------------

if not os.path.exists("evals"):
    os.makedirs("evals")

model = RealNVP(2, n_c_layers)
checkpoint = torch.load(path + "model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

data = Dataset(density)
x = data.generate_data()

# perform data -> noise mapping, should look like an isotropic Gaussian
z, _ = model(torch.from_numpy(x))
z = z.detach().numpy()
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.scatter(z[:, 0], z[:, 1], s=4)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect(1)
plt.savefig("evals/" + "data_to_noise.png")


# generate standard Gaussian samples
z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 10000).astype(np.float32)
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
sns.scatterplot(z[:, 0], z[:, 1], s=20, alpha=0.5, color="darkblue", ax=ax)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect(1)

# generate data from noise
x, _ = model.reverse(torch.from_numpy(z))
x = x.detach().numpy()
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
sns.scatterplot(x[:, 0], x[:, 1], s=10, color="darkblue", alpha=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect(1)
plt.savefig("evals/" + "noise_to_data.png")

# generate nice density plot
blues = cm.get_cmap("Blues", 512)
nbins = 300
k = kde.gaussian_kde([x[:, 0], x[:, 1]])
xi, yi = np.mgrid[
    x[:, 0].min() - 1 : x[:, 0].max() + 1 : nbins * 1j,
    x[:, 1].min() - 1 : x[:, 1].max() + 1 : nbins * 1j,
]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_facecolor(blues(1))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Blues")


ax.set_aspect(1)
plt.savefig("evals/" + "ntnu_logo_density.png")

