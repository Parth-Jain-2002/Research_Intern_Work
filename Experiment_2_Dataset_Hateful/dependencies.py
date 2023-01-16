# accuracies = [0.80, 0.77, 0.75, 0.72, 0.70, 0.67, 0.65, 0.63, 0.60, 0.57, 0.55, 0.53, 0.50] # must be in non-increasing order
# accuracies = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50] # must be in non-increasing order
# accuracies = [0.80, 0.75, 0.70, 0.65] # must be in non-increasing order
# accuracies = [0.7] * 5
# accuracies = [0.7] * 10
# accuracies = [0.7] * 15
# accuracies = [0.7] * 20
all_accuracies = [
                    [0.80, 0.77, 0.75, 0.72, 0.70, 0.67, 0.65, 0.63, 0.60, 0.57, 0.55, 0.53, 0.50],
                    [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50],
                    [0.80, 0.75, 0.70, 0.65],
                    [0.7] * 5,
                    [0.7] * 10,
                    [0.7] * 15,
                    [0.7] * 20
                ]

# sort all accuries in all_accuraries in non-increasing order
for accuracies in all_accuracies:
    accuracies.sort(reverse=1)

# NUM_HUMANS = len(accuracies)

# print hello world


import os
import deepdish
import time
import warnings
import csv

from collections.abc import Mapping
import matplotlib.pyplot as plt

from   collections import defaultdict

import pandas as pd
import numpy as np

import torch
from   torch import nn
from   torch import nn, optim
from   torch.distributions.log_normal import LogNormal
from   torch.nn.functional import softmax

import scipy.cluster.vq
import scipy
import scipy.stats
import scipy.integrate as integrate
import scipy.sparse as sp
from   scipy import optimize

from   sklearn.isotonic import IsotonicRegression
from   sklearn.utils.extmath import stable_cumsum, row_norms  
from   sklearn.metrics.pairwise import euclidean_distances
from   sklearn.metrics import confusion_matrix
from   sklearn.model_selection import GridSearchCV, StratifiedKFold
from   sklearn.linear_model import LogisticRegression
from   sklearn.model_selection import train_test_split

import pyro
import pyro.distributions as dist
from   pyro.infer import MCMC, NUTS

import calibration as cal
import contextlib

from tqdm import tqdm

from policy import *

EPS = 1e-50
rng = np.random.default_rng(1234)
PROJECT_ROOT = "."
warnings.filterwarnings("ignore")
