import torch
from torch.utils import data
import torch.nn as nn

import os.path
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns


class Dataset:
    def __init__(
        self, density="ntnu", fname=r"/Users/edvardhulten/real_nvp_2d/ntnu.jpg"
    ):
        self.density = density
        if self.density is "ntnu":
            self.img = mpimg.imread(r"/Users/edvardhulten/real_nvp_2d/ntnu.jpg")
            self.img = np.round(self.col2gray(self.img))
            # create grid and get coordinates
            ys, xs = np.mgrid[0 : self.img.shape[0], 0 : self.img.shape[1]]
            # find idx for coloured pixels
            idx = self.img == 0
            xs = xs[idx]
            ys = ys[idx]
            xs = self.normalise(xs) + 5
            ys = self.normalise(ys) + 5
            self.x = np.vstack((xs.flatten(), ys.flatten())).T
            self.x = self.x.astype(np.float32)

    def generate_data(self, n_samples=100):
        if self.density == "moons":
            self.x = datasets.make_moons(n_samples=n_samples, noise=0.05)[0].astype(
                np.float32
            )

        elif self.density == "ntnu":
            self.x = self.x[np.random.choice(self.x.shape[0], n_samples, replace=True)]
            self.x += (np.random.rand(*self.x.shape,) - 0.5) / 40

        return self.x

    def plot_original(self):
        fig, axes = plt.subplots(ncols=1, nrows=1)
        axes.scatter(self.x[:, 0], self.x[:, 1], c="darkblue")
        axes.set_aspect("equal")
        plt.show()

    def col2gray(self, img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray / 255

    def normalise(self, x):
        return (x - np.mean(x)) / np.std(x)

