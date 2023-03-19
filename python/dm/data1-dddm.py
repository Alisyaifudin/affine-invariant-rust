from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from mcmc import dddm
from mcmc import utils
from tqdm import tqdm
import pandas as pd

tes = True if (len(sys.argv) == 2) and (sys.argv[1] == "True") else False

plt.style.use('seaborn-v0_8-deep') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern"]}
plt.rcParams.update(params)

ndim = 33
nwalkers = 2*ndim+2

locs = dict(
    rhoDM=-0.02,
    sigmaDD=1,
    log_hDD=np.log(1),
    log_nu0=-3,
    R=3.4E-3,
    zsun=-50,
    w0=-10,
    log_sigmaw=1,
    log_a=-2
)

scales = dict(
    rhoDM=0.08,
    sigmaDD=30,
    log_hDD=np.log(100),
    log_nu0=2,
    R=0.6E-3,
    zsun=100,
    w0=5,
    log_sigmaw=2,
    log_a=3
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

p0 = dddm.generate_p0(nwalkers, locs, scales, kind=1)

zdata = np.loadtxt('data/z1.csv', skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]
dz = zmid[1] - zmid[0]

wdata = np.loadtxt('data/w1.csv', skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]
dw = wmid[1] - wmid[0]
zbound = 50

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

t0 = time()
sampler = dddm.run_mcmc(500, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True)
print(time() - t0, "s")
chain, probs = sampler[:,:,:-3], sampler[:,:,-3:]

# plot first 500 steps
rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
sigmaDD = chain[:, :, 25].T
log_hDD = chain[:, :, 26].T
nu0 = chain[:, :, 27].T
R = chain[:, :, 28].T
zsun = chain[:, :, 29].T
w0 = chain[:, :, 30].T
log_sigmaw = chain[:, :, 31].T
log_a = chain[:, :, 32].T

params = np.array([rhob, sigmaz, rhoDM, sigmaDD, log_hDD, nu0, R, zsun, w0, log_sigmaw, log_a]).T

labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\sigma_{\textup{DD}}$', r'$h_{\textup{DD}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
print("plot chain...")
if tes:
    utils.plot_chain(params[200:], labels, figsize=(10, 14), path="data/chain-test.png")
    sys.exit()
else:
    utils.plot_chain(params, labels, figsize=(10, 14), path='data/chain-dddm-1-0.png')

# run again
p0_next = chain[-1]
t0 = time()
sampler = dddm.run_mcmc(5000, nwalkers, p0_next, zdata, wdata, locs, scales, dz=1, verbose=True)
print(time() - t0, "s")
chain, probs = sampler[:,:,:-3], sampler[:,:,-3:]

# plot chain
rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
sigmaDD = chain[:, :, 25].T
log_hDD = chain[:, :, 26].T
nu0 = chain[:, :, 27].T
R = chain[:, :, 28].T
zsun = chain[:, :, 29].T
w0 = chain[:, :, 30].T
log_sigmaw = chain[:, :, 31].T
log_a = chain[:, :, 32].T

params = np.array([rhob, sigmaz, rhoDM, sigmaDD, log_hDD, nu0, R, zsun, w0, log_sigmaw, log_a]).T

labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\sigma_{\textup{DD}}$', r'$h_{\textup{DD}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
print("plot chain...")
utils.plot_chain(params, labels, figsize=(10, 14), path='data/chain-dddm-1-1.png')
# # chain.shape
# plot corner
rhob_f = rhob/1E-2
sigmaz_f = sigmaz
rhoDM_f = rhoDM/1E-2
sigmaDD_f = sigmaDD
log_hDD_f = log_hDD
nu0_f = nu0
R_f = R/1E-3
zsun_f = zsun
w0_f = w0
log_sigmaw_f = log_sigmaw
log_a_f = log_a

samples = np.array([rhob_f, sigmaz_f, rhoDM_f, sigmaDD_f, log_hDD_f, nu0_f, R_f, zsun_f, w0_f, log_sigmaw_f, log_a_f]).T

labels = [r'$\rho_b\times 10^2$', r'$\sigma_z$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\sigma_{\textup{DD}}$', r'$h_{\textup{DD}}$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
print("plot corner...")
utils.plot_corner(samples, labels, path="data/corner-dddm-1.png")

# fit
print("plot fitting...")
utils.plot_fit(dddm, zdata, wdata, chain, ndim, path="data/fitting-dddm-1.png")
# save chain
# df_com = []
# df_dict = {}

# for i in tqdm(range(nwalkers)):
#     chain = sampler[:, i, :]
#     for j in range(12):
#         df_dict[f'rhob_{j}'] = chain[:, j]
#         df_dict[f'sigmaz_{j}'] = chain[:, j+12]
#     df_dict['rhoDM'] = chain[:, 24]
#     df_dict['log_nu0'] = chain[:, 25]
#     df_dict['R'] = chain[:, 26]
#     df_dict['zsun'] = chain[:, 27]
#     df_dict['w0'] = chain[:, 28]
#     df_dict['sigmaw'] = chain[:, 29]
#     df_dict['log_a'] = chain[:, 30]
#     df_dict['walker'] = np.repeat(i, len(chain))
#     df = pd.DataFrame(df_dict)
#     if len(df_com) == 0:
#         df_com = df
#     else:
#         df_com = pd.concat([df_com, df], ignore_index=True)
BIC = -2*np.max(probs[:, 1]) + ndim*np.log(len(zdata)+len(wdata))
print(f'BIC = {BIC}')
# df_com.to_csv('chain.csv', index=False)