import numpy as np
import affine_invariant as af
import pandas as pd
from matplotlib import pyplot as plt
import corner
from tqdm import tqdm
from time import time

plt.style.use('seaborn-v0_8-bright') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern"]}
plt.rcParams.update(params)

locs = dict(
    rhoDM=-0.02,
    log_nu0=5,
    R=3.4E-3,
    zsun=-20,
    w0=-10,
    sigmaw=6,
    log_a=5
)

scales = dict(
    rhoDM=0.08,
    log_nu0=4,
    R=0.6E-3,
    zsun=40,
    w0=5,
    sigmaw=6,
    log_a=4
)

locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

ndim = 31
nwalkers = 2*ndim+2

p0 = af.generate_p0_dm1(nwalkers, locs, scales)

zdata = np.loadtxt('z.csv', skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]
dz = zmid[1] - zmid[0]

wdata = np.loadtxt('w.csv', skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]
dw = wmid[1] - wmid[0]
zbound = 50

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

t0 = time()
sampler = af.mcmc_dm1(500, nwalkers, p0, zdata, wdata, locs, scales, dz=10, verbose=True)
print(time() - t0, "s")

# plot first 500 steps
rhob = sampler[:, :, :12].sum(axis=2)
sigmaz = sampler[:, :, 12:24].sum(axis=2)
rhoDM = sampler[:, :, 24]
nu0 = sampler[:, :, 25]
R = sampler[:, :, 26]
zsun = sampler[:, :, 27]
w0 = sampler[:, :, 28]
sigmaw = sampler[:, :, 29]
a = sampler[:, :, 30]


chain = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, sigmaw, a])
# # chain.shape
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\sigma_w$', r'$\log a$']
fig, axes = plt.subplots(9, 1, figsize=(10, 10), sharex=True)
for i in range(9):
    axes[i].plot(chain[i], color="k", alpha=0.1)
    axes[i].set_xlim(0, len(sampler)-1)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("step number")
plt.savefig('chain-0.png', dpi=70)
plt.show()

p0_next = sampler[-1]
sampler = af.mcmc_dm1(4000, nwalkers, p0_next, zdata, wdata, locs, scales, dz=10, verbose=True)
# plot chain
rhob = sampler[:, :, :12].sum(axis=2)
sigmaz = sampler[:, :, 12:24].sum(axis=2)
rhoDM = sampler[:, :, 24]
nu0 = sampler[:, :, 25]
R = sampler[:, :, 26]
zsun = sampler[:, :, 27]
w0 = sampler[:, :, 28]
sigmaw = sampler[:, :, 29]
a = sampler[:, :, 30]


chain = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, sigmaw, a])
# # chain.shape
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\sigma_w$', r'$\log a$']
fig, axes = plt.subplots(9, 1, figsize=(10, 10), sharex=True)
for i in range(9):
    axes[i].plot(chain[i], color="k", alpha=0.05)
    axes[i].set_xlim(0, len(sampler)-1)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("step number")
plt.savefig('chain-1.png', dpi=70)
plt.show()
# plot corner
rhob_f = rhob.reshape(-1)/1E-2
sigmaz_f = sigmaz.reshape(-1)
rhoDM_f = rhoDM.reshape(-1)/1E-2
nu0_f = nu0.reshape(-1)
R_f = R.reshape(-1)/1E-3
zsun_f = zsun.reshape(-1)
w0_f = w0.reshape(-1)
sigmaw_f = sigmaw.reshape(-1)
a_f = a.reshape(-1)


flat_samples = np.array([rhob_f, sigmaz_f, rhoDM_f, nu0_f, R_f, zsun_f, w0_f, sigmaw_f, a_f])
labels = [r'$\rho_b\times 10^2$', r'$\sigma_z$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\sigma_w$', r'$a$']

fig = corner.corner(
    flat_samples.T, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}
)
fig.savefig("corner.png", dpi=70)
plt.show()
# fit
zs = np.linspace(zmid.min(), zmid.max(), 100)
ws = np.linspace(wmid.min(), wmid.max(), 100)
flat_samples = sampler.reshape((-1, ndim))
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
axes[0].bar(zmid, znum, yerr=zerr, width=dz, color='k', alpha=0.5)
axes[0].set_ylabel(r'$N(z)$')
axes[0].set_xlabel(r'$z$')
axes[1].bar(wmid, wnum, yerr=werr, width=dw, color='k', alpha=0.5)
axes[1].set_ylabel(r'$N(w)$')
axes[0].set_xlabel(r'$w$')
print("fitting....")
for i in tqdm(range(1000)): 
    ind = np.random.randint(len(flat_samples))
    theta = flat_samples[ind]
    fz_mod = af.fz1(zs, theta)
    fw_mod = af.fw1(ws, theta)
    axes[0].plot(zs, fz_mod, color='r', alpha=0.005)
    axes[1].plot(ws, fw_mod, color='r', alpha=0.005)
plt.savefig('fit.png', dpi=70)
plt.show()
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
    df_dict['sigmaw'] = chain[:, 29]
    df_dict['log_a'] = chain[:, 30]
    df_dict['walker'] = np.repeat(i, len(chain))
    df = pd.DataFrame(df_dict)
    if len(df_com) == 0:
        df_com = df
    else:
        df_com = pd.concat([df_com, df], ignore_index=True)
df_com.to_csv('chain.csv', index=False)