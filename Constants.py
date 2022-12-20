import os
import numpy as np

DATA_DIR = "data"
DEFAULT_INPUT_FNAME = "default.csv"
DEFAULT_INPUT_FILE = os.path.join(DATA_DIR, DEFAULT_INPUT_FNAME)

ERROR_MSG_TEMPLATE = """
Tried to render graph of column {}, but encountered the following error.
Remember that changes made to the dataset don't propagate until the Update Data button is clicked,
and make sure you haven't applied transformations on columns that result in undefined data! (e.g. logarithm of negative numbers).
"""

FONTSIZE = 22
# taken from matplotlib
COLORMAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

GRAPH_TYPES = ["Threshold Testing", "Single Variable Analysis"]


# LAMBDAS
perc2num = lambda data, p: np.percentile(data, p)
num2perc = lambda data, n: np.sum(data < n) / len(data) * 100

num2std = lambda std, n: n / std
std2num = lambda std, n: n * std
