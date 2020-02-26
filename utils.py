import torch
import torch.nn as nn
import numpy as np
import math


def train_one_epoch(model, epoch, optimizer, train_loader):
    model.train()
    train_loss = 0
    for batch in train_loader:
        u, log_det = model.forward(batch.float())

        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = negloglik_loss.mean()
        train_loss += negloglik_loss.item()

        optimizer.zero_grad()
        negloglik_loss.backward()
        optimizer.step()

    avg_loss = np.sum(train_loss) / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss


def val(model, train, val_loader):
    model.eval()
    val_loss = []
    with torch.no_grad():
        _, _ = model(train.float())
        for batch in val_loader:
            u, log_det = model(batch.float())

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det
            val_loss.extend(negloglik_loss.detach().tolist())

    print(
        "Validation loss: {:.4f} +/- {:.4f}".format(
            np.mean(val_loss), 2 * np.std(val_loss) / np.sqrt(len(val_loss))
        )
    )
    return np.mean(val_loss)


def test(model, train, test_loader):
    model.eval()
    test_loss = []
    with torch.no_grad():
        _, _ = model.forward(train)
        for batch in test_loader:
            u, log_det = model.forward(batch.float())

            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det

            test_loss.extend(negloglik_loss)

    N = len(test_loss)
    print(
        "Test loss: {:.4f} +/- {:.4f}".format(
            np.mean(test_loss), 2 + np.std(test_loss) / np.sqrt(N)
        )
    )
