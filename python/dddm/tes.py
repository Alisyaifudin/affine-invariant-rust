from matplotlib import pyplot as plt
import numpy as np
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from mcmc import dm
from time import time
import pandas as pd
from mcmc.utils import plot_chain, plot_corner

rhob = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018,
    0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015
]
sigmaz = [
    3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0, 
    18.0, 18.5, 18.5, 20.0, 20.0]
rhoDM = [0.016]
nu0 = [1]
R = [3.4E-3]
zsun = [30]
w0 = [-7.]
sigmaw = [5.]
a = [1.]
sigmaw2 = [10.]
a2 = [0.2]

theta = np.array([rhob + sigmaz + rhoDM + nu0 + R + zsun+w0 + sigmaw + a + sigmaw2 + a2]).flatten()

N = 16

z = np.random.randn(N)*500
w = np.random.randn(N)*20
dz = 1
pos = np.array([z, w]).T

nwalkers = pos.shape[0]
ndim = pos.shape[1]

t0 = time()
chain = dm.sample(100, nwalkers, pos, theta, dz=1., verbose=True)
print(time() - t0, "s")