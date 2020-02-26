import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from rnvp import RealNVP
from new_model import NewFlow
from utils import train_one_epoch, val, test
import time

# ------------ parameters ------------
dataset = "hepmass"
n_c_layers = 5  # number of coupling layers
epochs = 3000  # number of epochs to train for
batch_size = 100  # set batch size
lr = 1e-4  # set the learning rate of the adam optimiser
plot_interval = 12345234523456
path = r"/Users/edvardhulten/real_nvp_2d/"  # change to your own path (unless your name is Edvard Hultén too)
duration = 0.1
num_samples = 30  # must be multiple of 10
# ------------------------------------

if dataset == "mnist":
    from data.mnist import train, train_loader, val_loader, test_loader, n_in
elif dataset == "power":
    from data.power import train, train_loader, val_loader, test_loader, n_in
elif dataset == "hepmass":
    from data.hepmass import train, train_loader, val_loader, test_loader, n_in
else:
    raise ValueError(
        "Unknown dataset...\n\nPlease choose between 'mnist','power', and 'hepmass'."
    )


if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("gifs"):
    os.makedirs("gifs")

tot_epochs = 0

model = RealNVP(data_dim=n_in, n_c_layers=n_c_layers, n_hidden=512, hidden_dims=1)
# model = NewFlow(data_dim=n_in, n_c_layers=n_c_layers, n_hidden=700, hidden_dims=1)
print("# model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

# for plotting
mvn = torch.distributions.Normal(0, 1)
z_norm = mvn.sample([num_samples, np.ceil(n_in).astype(int)])
val_batch = next(iter(val_loader)).float()

start = time.time()
# for early stopping
i = 0
max_loss = np.inf
epochs_list = []
train_losses = []
val_losses = []
for epoch in range(1, epochs):
    epochs_list.append(epoch)
    train_loss = train_one_epoch(model, epoch, optimizer, train_loader)
    val_loss = val(model, train, val_loader)
    train_losses.append(train_loss)
    # val_losses.append(val_loss)
    val_loss = 100
    if val_loss < max_loss:
        max_loss = val_loss
        i = 0
        torch.save(
            model, (path + "model.pt"),
        )
    else:
        i += 1
    if i >= 30:
        break
    print("Patience counter: {}/30".format(i))

    if epoch % 1 == 0:
        stop = time.time()
        print("time elapsed previous epoch: {0:.2f} sec".format(stop - start))
        start = stop

    if epoch % plot_interval == 112837192873:
        model.eval()
        np.random.seed(0)
        samples, _ = model.reverse(z_norm)
        samples = (torch.sigmoid(samples) - 1e-6) / (1 - 2e-6)
        samples = samples.detach().cpu().view(num_samples, 28, 28)
        fig, axes = plt.subplots(ncols=10, nrows=int(num_samples / 10))
        ax = axes.ravel()
        for i in range(num_samples):
            ax[i].imshow(
                np.transpose(samples[i], (0, 1)), cmap="gray", interpolation="none"
            )
            ax[i].axis("off")
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_frame_on(False)

        plt.savefig(
            "results/" + "result_" + "{:03d}".format(epoch) + ".png",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.close()

test_loss = test(model, epochs)

