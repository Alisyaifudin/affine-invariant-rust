import warnings
warnings.filterwarnings("ignore")
import numpy as np
from time import time
import os
import sys
from os.path import join, abspath
parent_dir = os.path.dirname(os.getcwd())
root_dir = abspath(join(parent_dir, '..'))
current_dir = os.path.curdir
sys.path.append(root_dir)
from mcmc import dm
from mcmc import utils
from init import init

model_kind = int(sys.argv[1])
ndim, nwalkers = init(model_kind)
data_kind = int(sys.argv[2])
add_name = sys.argv[3] if len(sys.argv) == 4 else ""

utils.style()

print('load chain...')
sampler = np.load(join(current_dir, 'data', f'chain-({data_kind})-{add_name}{model_kind}-dm.npy'))
chain, probs = sampler[:,:,:-3], sampler[:,:,-3:]

zdata = np.loadtxt(join(parent_dir, 'data', f'z{data_kind}.csv'), skiprows=1, delimiter=',')
zdata = zdata[:, 0], zdata[:, 1], zdata[:, 2]

wdata = np.loadtxt(join(parent_dir, 'data', f'w{data_kind}.csv'), skiprows=1, delimiter=',')
wdata = wdata[:, 0], wdata[:, 1], wdata[:, 2]
print("plot fit curve...")
t0 = time()
utils.plot_fit(dm, zdata, wdata, chain, ndim, path=join(current_dir, 'plots', f'fit-({data_kind})-{add_name}{model_kind}-dm.png'))
print(f'plotting took {time()-t0:.2f} seconds')

BIC = -2*np.max(probs[:, 1]) + ndim*np.log(len(zdata)+len(wdata))
print(f'BIC = {BIC}')