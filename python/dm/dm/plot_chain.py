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
from mcmc import utils
from init import init

model_kind = int(sys.argv[1])
ndim, nwalkers = init(model_kind)
data_kind = int(sys.argv[2])
add_name = sys.argv[3] if len(sys.argv) == 4 else ""

utils.style()

ndim = 31 if model_kind == 1 else 33
nwalkers = 2*ndim+2

print('load chain...')
sampler = np.load(join(current_dir, 'data', f'chain-({data_kind})-{add_name}{model_kind}-0-dm.npy'))
chain = sampler[:,:,:-3]
# plot chain
rhob =  chain[1:, :, :12].sum(axis=2).T
sigmaz = chain[1:, :, 12:24].sum(axis=2).T
rhoDM = chain[1:, :, 24].T
nu0 = chain[1:, :, 25].T
R = chain[1:, :, 26].T
zsun = chain[1:, :, 27].T
w0 = chain[1:, :, 28].T
log_sigmaw1 = chain[1:, :, 29].T
log_a1 = chain[1:, :, 30].T
if model_kind == 2:
    log_sigmaw2 = chain[1:, :, 31].T
    log_a2 = chain[1:, :, 32].T

params = [rhob, sigmaz, rhoDM, nu0, R, zsun, w0, log_sigmaw1, log_a1]
if model_kind == 2:
    params += [log_sigmaw2, log_a2]

params = np.stack(params).T
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$']
if model_kind == 2:
    labels += [r'$\log \sigma_{w2}$', r'$\log a_2$']
print("plot the chains...")
t0 = time()
utils.plot_chain(params, labels, figsize=(10, 14), path=join(current_dir, 'plots', f'chain-({data_kind})-{add_name}{model_kind}-0-dm.png'))
print(f'plotting took {time()-t0:.2f} seconds')