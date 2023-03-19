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
from init import init

steps = int(sys.argv[1])
model_kind = int(sys.argv[2])
ndim, nwalkers = init(model_kind)
data_kind = int(sys.argv[3])
add_name = sys.argv[4] if len(sys.argv) == 5 else ""



locs = np.load(join(current_dir, 'data', f'locs-{model_kind}-dm.npy'))
scales = np.load(join(current_dir, 'data', f'scales-{model_kind}-dm.npy'))

sampler = np.load(join(current_dir, 'data', f'chain-({data_kind})-{add_name}{model_kind}-0-dm.npy'))
p0_next = sampler[-1, :, :-3]

zdata = np.loadtxt(join(parent_dir, 'data', f'z{data_kind}.csv'), skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]
dz = zmid[1] - zmid[0]

wdata = np.loadtxt(join(parent_dir, 'data', f'w{data_kind}.csv'), skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]
dw = wmid[1] - wmid[0]
zbound = 50

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

print("running mcmc...")
t0 = time()
sampler = dm.run_mcmc(steps, nwalkers, p0_next, zdata, wdata, locs, scales, dz=1, verbose=True)
print(time() - t0, "s")

np.save(join(current_dir, 'data', f'chain-({data_kind})-{add_name}{model_kind}-dm.npy'), sampler)