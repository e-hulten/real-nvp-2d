import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from rnvp import RealNVP
from new_model import NewFlow
from utils import train_one_epoch, val, test
from data.hepmass import train
from data.hepmass import val_loader as test_loader

from gif import make_gif
import time


model = torch.load("/Users/edvardhulten/real_nvp_2d/model.pt")
v = val(model, train, test_loader)
test_loss = test(model, train, test_loader)
