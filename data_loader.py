import torch
from torch.utils import data
import torch.nn as nn

import os.path
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns


class Dataset:
    def __init__(
        self, density=False, fname=r"/Users/edvardhulten/real_nvp_2d/ntnu.jpg"
    ):
        self.density = density
        if self.density is "ntnu":
            self.img = plt.imread(fname) / 255
            # reduce the number of pixels by a factor 4
            step_length = 4
            self.img = self.img[0:-1:step_length, 0:-1:step_length]
            # binarise image
            self.img = self.img.sum(axis=2)
            self.img[self.img < 2] = 0  # colour
            self.img[self.img >= 2] = 1  # white
            # create grid
            self.ys, self.xs = np.mgrid[
                self.img.shape[0] : 0 : -1, 0 : self.img.shape[1]
            ]
            # move grid to unit square around (0,0)
            self.xs = (self.xs - np.mean(self.xs)) / np.std(self.xs)
            self.ys = (self.ys - np.mean(self.ys)) / np.std(self.ys)

    def generate_data(self, n_samples=5000):

        if self.density == "moons":
            np.random.seed(0)
            x = datasets.make_moons(n_samples=n_samples, noise=0.05)[0].astype(
                np.float32
            )

        elif self.density == "ntnu":
            # find idx for coloured pixels
            idx = self.img == 0
            xs = self.xs[idx]
            ys = self.ys[idx]
            x = np.vstack((xs.flatten(), ys.flatten())).T
            x = x.astype(np.float32)

        return x

    def plot_original(self):

        plt.scatter(
            self.xs.flatten(), self.ys.flatten(), s=4, c=im.flatten().reshape(-1, 3)
        )
        plt.show()
