from sklearn.datasets import make_blobs, make_moons, make_circles
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn.datasets import make_blobs, make_moons, make_circles
import h5py
from sklearn.linear_model import SGDOneClassSVM 
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import classification_report, confusion_matrix
import time
plt.rcParams.update({
    'font.family': 'sans-serif',     # Font family (serif, sans-serif, etc.)
    'font.size': 10,            # Font size
    'font.weight': 'normal',      # Font weight (normal, bold, etc.)
    'axes.titlesize': 12,       # Title font size
    'axes.labelsize': 10,       # Axis labels font size
    'xtick.labelsize': 6,      # X-axis tick label size
    'ytick.labelsize': 6       # Y-axis tick label size
})
