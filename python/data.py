import numpy as np
import affine_invariant as af
from scipy.stats import norm, uniform
import pandas as pd
from matplotlib import pyplot as plt
import corner
from tqdm import tqdm

plt.style.use('seaborn-v0_8-bright') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern"]}
plt.rcParams.update(params)

rhob = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018,
    0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015
]
e_rhob = []

sigmaz = [
    3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0, 
    18.0, 18.5, 18.5, 20.0, 20.0]
e_sigmaz = []

rhoDM = [0.016]
e_rhoDM = [0.001]

R = [3.4E-3]
e_R = [0.1E-3]

zsun = [0]
e_zsun = [1]

sigmaw = [10.]
e_sigmaw = [1.]

w0 = [-7]
e_w0 = [1.]

theta = np.array([rhob + sigmaz + rhoDM + R + zsun + sigmaw + w0]).flatten()

ndim = len(theta)
nwalkers = 2*ndim+6
nsteps = 500

rhob = [0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018, 0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015]
e_rhob = [0.00312, 0.00554, 0.00070, 0.00003, 0.00006, 0.00018, 0.00018, 0.00029, 0.00072, 0.00280,
    0.00100, 0.00050]
sigmaz = [3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0, 18.0, 18.5, 18.5, 20.0, 20.0]
e_sigmaz = [0.2, 0.5, 2.4, 4.0, 1.6, 2.0, 2.4, 1.8, 1.9, 4.0, 5.0, 5.0]

locs = {
    'rhob' : rhob,
    'sigmaz': sigmaz,
    'rhoDM': -0.01,
    'R': 3.4E-3,
    'zsun': -50.,
    'sigmaw':1.,
    'w0': -15.
}
scales = {
    'rhob' : e_rhob,
    'sigmaz': e_sigmaz,
    'rhoDM': 0.15,
    'R': 0.6E-3,
    'zsun': 100.,
    'sigmaw': 20.,
    'w0': 10.
}

rhob_0 = norm.rvs(loc=locs['rhob'], scale=scales['rhob'], size=(nwalkers, 12))
sigmaz_0 = norm.rvs(loc=locs['sigmaz'], scale=scales['sigmaz'], size=(nwalkers, 12))

rhoDM_0 = uniform.rvs(loc=locs['rhoDM'], scale=scales['rhoDM'], size=nwalkers)
R_0 = uniform.rvs(loc=locs['R'], scale=scales['R'], size=nwalkers)
zsun_0 = uniform.rvs(loc=locs['zsun'], scale=scales['zsun'], size=nwalkers)
sigmaw_0 = uniform.rvs(loc=locs['sigmaw'], scale=scales['sigmaw'], size=nwalkers)
w0_0 = uniform.rvs(loc=locs['w0'], scale=scales['w0'], size=nwalkers)

p0 = np.array([*rhob_0.T, *sigmaz_0.T, rhoDM_0, R_0, zsun_0, sigmaw_0, w0_0]).T

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

zdata = (zmid, znum, zerr, dz)
wdata = (wmid, wnum, werr, dw, zbound)

sampler = af.mcmc(500, nwalkers, p0, zdata, wdata, True)

# plot first 500 steps
rhob = sampler[:, :, :12].sum(axis=2)
sigmaz = sampler[:, :, 12:24].sum(axis=2)
rhoDM = sampler[:, :, 24]
R = sampler[:, :, 25]
zsun = sampler[:, :, 26]
sigmaw = sampler[:, :, 27]
w0 = sampler[:, :, 28]

chain = np.array([rhob, sigmaz, rhoDM, R, zsun, sigmaw, w0])
# # chain.shape
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$R$', r'$z_{\odot}$', r'$\sigma_w$', r'$w_0$']
fig, axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
for i in range(7):
    axes[i].plot(chain[i], color="k", alpha=0.1)
    axes[i].set_xlim(0, len(sampler)-1)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("step number")
plt.savefig('chain-0.png', dpi=120)
plt.show()
# run 2000 again
nsteps = 2000
sampler = af.mcmc(nsteps, nwalkers, sampler[-1], zdata, wdata, True)
# plot chain
rhob = sampler[:, :, :12].sum(axis=2)
sigmaz = sampler[:, :, 12:24].sum(axis=2)
rhoDM = sampler[:, :, 24]
R = sampler[:, :, 25]
zsun = sampler[:, :, 26]
sigmaw = sampler[:, :, 27]
w0 = sampler[:, :, 28]

chain = np.array([rhob, sigmaz, rhoDM, R, zsun, sigmaw, w0])
# # chain.shape
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$R$', r'$z_{\odot}$', r'$\sigma_w$', r'$w_0$']
fig, axes = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
for i in range(7):
    axes[i].plot(chain[i], color="k", alpha=0.01)
    axes[i].set_xlim(0, len(sampler)-1)
    axes[i].set_ylabel(labels[i])
axes[-1].set_xlabel("step number")
plt.savefig('chain-1.png', dpi=120)
plt.show()
# plot corner
rhob_f = rhob.reshape(-1)/1E-2
sigmaz_f = sigmaz.reshape(-1)
rhoDM_f = rhoDM.reshape(-1)/1E-2
R_f = R.reshape(-1)/1E-3
zsun_f = zsun.reshape(-1)
sigmaw_f = sigmaw.reshape(-1)
w0_f = w0.reshape(-1)

flat_samples = np.array([rhob_f, sigmaz_f, rhoDM_f, R_f, zsun_f, sigmaw_f, w0_f])
labels = [r'$\rho_b\times 10^2$', r'$\sigma_z$', r'$\rho_{\textup{DM}}\times 10^2$', r'$R\times 10^3$', r'$z_{\odot}$', r'$\sigma_w$', r'$w_0$']

fig = corner.corner(
    flat_samples.T, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}
)
fig.savefig("corner.png", dpi=120)
plt.show()
# fit
Nz = znum.sum()
Nw = wnum.sum()
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
for i in tqdm(range(300)):
    ind = np.random.randint(len(flat_samples))
    theta = flat_samples[ind]
    Nz_mod = af.Nz1(zs, dz, Nz, theta)
    Nw_mod = af.Nw1(ws, dw, Nw, zbound, theta)
    axes[0].plot(zs, Nz_mod, color='r', alpha=0.01)
    axes[1].plot(ws, Nw_mod, color='r', alpha=0.01)
plt.savefig('fit.png', dpi=120)
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
    df_dict['R'] = chain[:, 25]
    df_dict['zsun'] = chain[:, 26]
    df_dict['sigmaw'] = chain[:, 27]
    df_dict['w0'] = chain[:, 28]
    df_dict['walker'] = np.repeat(i, nsteps+1)
    df = pd.DataFrame(df_dict)
    if len(df_com) == 0:
        df_com = df
    else:
        df_com = pd.concat([df_com, df], ignore_index=True)
df_com.to_csv('chain.csv', index=False)