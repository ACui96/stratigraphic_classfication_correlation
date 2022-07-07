import pickle

import pandas as pd

from models.utils.dataprocessor import plot_acc_loss

with open('../data/history.txt') as f:
    history = pickle.load(f)
plot_acc_loss(history)