from matplotlib import pyplot as plt
import numpy as np
from time import time
import os
import sys
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from mcmc import dm
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
    log_nu0=-2,
    R=3.4E-3,
    zsun=-50,
    w0=-10,
    log_sigmaw1=np.log(5)-1,
    log_a1=0,
    log_sigmaw2=np.log(10)-0.5,
    log_a2=-3
)

scales = dict(
    rhoDM=0.08,
    log_nu0=2,
    R=0.6E-3,
    zsun=100,
    w0=5,
    log_sigmaw1=2,
    log_a=1,
    log_sigmaw2=1.5,
    log_a2=5
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

p0 = dm.generate_p0(nwalkers, locs, scales, kind=2)

zdata = np.loadtxt('data/z2.csv', skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]
dz = zmid[1] - zmid[0]

wdata = np.loadtxt('data/w2.csv', skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]
dw = wmid[1] - wmid[0]
zbound = 50

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

t0 = time()
sampler = dm.run_mcmc(500, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True)
print(time() - t0, "s")
chain, probs = sampler[:,:,:-3], sampler[:,:,-3:]
# plot chain
rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
nu0 = chain[:, :, 25].T
R = chain[:, :, 26].T
zsun = chain[:, :, 27].T
w0 = chain[:, :, 28].T
log_sigmaw1 = chain[:, :, 29].T
log_a1 = chain[:, :, 30].T
log_sigmaw2 = chain[:, :, 31].T
log_a2 = chain[:, :, 32].T

params = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, log_sigmaw1, log_a1, log_sigmaw2, log_a2]).T

labels = labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$', r'$\log \sigma_{w2}$', r'$\log a_2$']
if tes:
    utils.plot_chain(params[200:], labels, path="data/chain-test.png")
    sys.exit()
else:
    utils.plot_chain(params, labels, path="data/chain-2-0.png")
# run again
p0_next = chain[-1, :, :]
t0 = time()
sampler = dm.run_mcmc(2000, nwalkers, p0_next, zdata, wdata, locs, scales, dz=1, verbose=True)
print(time() - t0, "s")
chain, probs = sampler[:,:,:-3], sampler[:,:,-3:]
# plot chain
rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
nu0 = chain[:, :, 25].T
R = chain[:, :, 26].T
zsun = chain[:, :, 27].T
w0 = chain[:, :, 28].T
log_sigmaw1 = chain[:, :, 29].T
log_a1 = chain[:, :, 30].T
log_sigmaw2 = chain[:, :, 31].T
log_a2 = chain[:, :, 32].T

params = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, log_sigmaw1, log_a1, log_sigmaw2, log_a2]).T

labels = labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$', r'$\log \sigma_{w2}$', r'$\log a_2$']
utils.plot_chain(params, labels, path="data/chain-2-1.png")
# corner
rhob_f = rhob/1E-2
sigmaz_f = sigmaz
rhoDM_f = rhoDM/1E-2
nu0_f = nu0
R_f = R/1E-3
zsun_f = zsun
w0_f = w0
log_sigmaw1_f = log_sigmaw1
log_a1_f = log_a1
log_sigmaw2_f = log_sigmaw2
log_a2_f = log_a2

flat_samples = np.array([rhob_f, sigmaz_f, rhoDM_f, nu0_f, R_f, zsun_f, w0_f, log_sigmaw1_f, log_a1_f, log_sigmaw2_f, log_a2_f]).T

labels = [r'$\rho_b\times 10^2$', r'$\sigma_z$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$', r'$\log \sigma_{w2}$', r'$\log a_2$']
utils.plot_corner(flat_samples, labels, path="data/corner-2.png")
# fitting
def plot_fit(zdata, wdata, chain, ndim, n=50000, alpha=0.2, path=None, dpi=100):
    zmid, znum, zerr = zdata
    wmid, wnum, werr = wdata
    
    flat_samples = chain.reshape((-1, ndim))
    zs = np.linspace(zmid.min()*1.1, zmid.max()*1.1, 100)
    ws = np.linspace(wmid.min()*1.1, wmid.max()*1.1, 100)
    fzs = np.empty((n, len(zs)))
    fws = np.empty((n, len(ws)))
    for i in tqdm(range(n)):
        ind = np.random.randint(len(flat_samples))
        theta = flat_samples[ind]
        fzs[i] = dm.fz(zs, theta ,1.)
        fws[i] = dm.fw(ws, theta ,1.)
    fz_log_mean = np.log(fzs).mean(axis=0)
    fz_log_std = np.log(fzs).std(axis=0)
    fz_mean = np.exp(fz_log_mean)

    fw_log_mean = np.log(fws).mean(axis=0)
    fw_log_std = np.log(fws).std(axis=0)
    fw_mean = np.exp(fw_log_mean)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].errorbar(zmid, znum, yerr=zerr, color='k', alpha=0.5, capsize=2, fmt=".")
    axes[0].fill_between(zs, np.exp(fz_log_mean - 1.9600*fz_log_std), np.exp(fz_log_mean + 1.9600*fz_log_std), alpha=alpha, color="C0")
    axes[0].fill_between(zs, np.exp(fz_log_mean - 1.6449*fz_log_std), np.exp(fz_log_mean + 1.6449*fz_log_std), alpha=alpha, color="C0")
    axes[0].fill_between(zs, np.exp(fz_log_mean - fz_log_std), np.exp(fz_log_mean + fz_log_std), alpha=alpha, color="C0")
    axes[0].plot(zs, fz_mean, c="C0", ls="--")
    axes[0].set_ylabel(r'$\nu(z)$')
    axes[0].set_xlabel(r'$z$ [pc]')
    axes[0].set_ylim(np.min(np.exp(fz_log_mean - 1.9600*fz_log_std)), np.max(np.exp(fz_log_mean)*1.5))
    axes[0].set_xlim(zs.min(), zs.max())
    axes[0].set_yscale('log')
    axes[1].errorbar(wmid, wnum, yerr=werr, color='k', alpha=0.5, capsize=2, fmt=".")
    axes[1].fill_between(ws, np.exp(fw_log_mean - 1.9600*fw_log_std), np.exp(fw_log_mean + 1.9600*fw_log_std), alpha=alpha, color="C0")
    axes[1].fill_between(ws, np.exp(fw_log_mean - 1.6449*fw_log_std), np.exp(fw_log_mean + 1.6449*fw_log_std), alpha=alpha, color="C0")
    axes[1].fill_between(ws, np.exp(fw_log_mean - fw_log_std), np.exp(fw_log_mean + fw_log_std), alpha=alpha, color="C0")
    axes[1].plot(ws, fw_mean, c="C0", ls="--")
    axes[1].set_ylabel(r'$f_0(w)$')
    axes[1].set_xlabel(r'$w$ [km/s]]')
    axes[1].set_ylim(np.min(np.exp(fw_log_mean - 1.9600*fw_log_std)), np.max(np.exp(fw_log_mean)*1.5))
    axes[1].set_xlim(ws.min(), ws.max())
    axes[1].set_yscale('log')
    if path is not None:
        fig.savefig(path, dpi=dpi)
plot_fit(zdata, wdata, chain, ndim, path="data/fitting-2.png")
# save chain
df_com = []
df_dict = {}

for i in tqdm(range(nwalkers)):
    chain = sampler[:, i, :]
    for j in range(12):
        df_dict[f'rhob_{j}'] = chain[:, j]
        df_dict[f'sigmaz_{j}'] = chain[:, j+12]
    df_dict['rhoDM'] = chain[:, 24]
    df_dict['log_nu0'] = chain[:, 25]
    df_dict['R'] = chain[:, 26]
    df_dict['zsun'] = chain[:, 27]
    df_dict['w0'] = chain[:, 28]
    df_dict['log_sigmaw1'] = chain[:, 29]
    df_dict['log_a1'] = chain[:, 30]
    df_dict['log_sigmaw2'] = chain[:, 31]
    df_dict['log_a2'] = chain[:, 32]
    df_dict['walker'] = np.repeat(i, len(chain))
    df = pd.DataFrame(df_dict)
    if len(df_com) == 0:
        df_com = df
    else:
        df_com = pd.concat([df_com, df], ignore_index=True)
print(df_com)
# bic
BIC = -2*np.max(probs[:, 1]) + ndim*np.log(len(zdata)+len(wdata))
print(f'BIC = {BIC}')
# df_com.to_csv('chain.csv', index=False)