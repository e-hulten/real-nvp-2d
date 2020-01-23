import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from model import RealNVP, NegLogLik
from data_loader import Dataset
from gif import make_gif
import time

# ------------ parameters ------------
continue_training = True  # set to false if you want to train a new model
gif = True
density = "moons"  # set to true if you want to use the two moons dataset
n_c_layers = 12  # number of coupling layers
epochs = 100  # number of epochs to train for
batch_size = 100  # set batch size
lr = 5e-4  # set the learning rate of the adam optimiser
plot_interval = 1
path = r"/Users/edvardhulten/real_nvp_2d/"  # change to your own path (unless your name is Edvard Hultén too)
distr_name = "two_moons"
duration = 0.1
# ------------------------------------

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("gifs"):
    os.makedirs("gifs")

data = Dataset(density)
x = data.generate_data()

model = RealNVP(data_dim=x.shape[1], n_c_layers=n_c_layers)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if continue_training is True:
    checkpoint = torch.load(path + "model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint['scheduler'])

    tot_epochs = checkpoint["epoch"]
    loss = checkpoint["loss"]
else:
    tot_epochs = 0


loss_log_det_J = NegLogLik(model)
train_loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=True)
# for plotting
z_norm = np.random.multivariate_normal(np.zeros(2), np.eye(2), 2000).astype(np.float32)

start = time.time()
for epoch in range(tot_epochs + 1, tot_epochs + epochs + 1):
    for batch in train_loader:
        model.train()
        z, sum_log_det_J = model(batch)
        loss = loss_log_det_J(z, sum_log_det_J)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        stop = time.time()
        print("(epoch %s/%s) loss : %.3f" % (epoch, tot_epochs + epochs, loss.item()))
        print("time elapsed previous 10 epochs: {0:.2f} sec".format(stop - start))
        start = stop

    if epoch % plot_interval == 0:
        model.eval()
        z, _ = model(torch.from_numpy(x))
        z = z.detach().numpy()

        np.random.seed(0)
        x_n, _ = model.reverse(torch.from_numpy(z_norm))
        x_n = x_n.detach().numpy()

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),)
        fig.subplots_adjust(left=0, right=1)

        sns.scatterplot(
            x[:, 0], x[:, 1], ax=ax[0, 0], color="steelblue",
        )
        ax[0, 0].set_title("One batch $x$, size: {}".format(x.shape[0]))
        # ax[0, 0].set_xlim(-1.5, 2.5)
        ax[0, 0].set_xlim(-2.2, 3)
        # ax[0, 0].set_ylim(-1, 1.5)
        ax[0, 0].set_ylim(-2.4, 3)

        sns.scatterplot(z[:, 0], z[:, 1], ax=ax[0, 1], color="darkred")
        ax[0, 1].set_title(r"$u = f^{-1}(x)$" + ", size: {}".format(x.shape[0]))
        ax[0, 1].set_xlim(-3, 3)
        ax[0, 1].set_ylim(-3, 3)

        sns.scatterplot(
            z_norm[:, 0], z_norm[:, 1], ax=ax[1, 0], color="steelblue",
        )
        ax[1, 0].set_title(
            "Samples $u$ from a standard Gaussian, size: {}".format(z_norm.shape[0])
        )
        ax[1, 0].set_xlim(-3, 3)
        ax[1, 0].set_ylim(-3, 3)

        sns.scatterplot(x_n[:, 0], x_n[:, 1], ax=ax[1, 1], color="darkred")
        ax[1, 1].set_title("x = f(u), size: {}".format(x_n.shape[0]))
        ax[1, 1].set_xlim(-2.2, 3)
        ax[1, 1].set_ylim(-2.4, 3)
        plt.suptitle("Iteration: {:03d}".format(epoch), size=18)

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
            "results/" + "result_" + "{:03d}".format(epoch) + ".png",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.close()

if gif is True:
    make_gif(distr_name, duration=duration)

torch.save(
    {
        "epoch": tot_epochs + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
        #'scheduler': scheduler.state_dict()
    },
    (path + "model.pt"),
)

