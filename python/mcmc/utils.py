from matplotlib import pyplot as plt
import corner
import numpy as np
from tqdm import tqdm

def plot_chain(chain, labels, figsize=(10, 10), alpha=0.1, path=None):
    fig, ax = plt.subplots(len(labels), 1, figsize=figsize)
    ax[0].set_title("MCMC Chain")
    for i, label in enumerate(labels):
        ax[i].plot(chain[:, :, i], alpha=alpha, color="k")
        ax[i].set_ylabel(label)
        ax[i].set_xlim(0, len(chain)-1)
        ax[i].grid(False)
    ax[-1].set_xlabel("step number")
    if path is not None:
        fig.savefig(path)
    fig.tight_layout()


def plot_corner(chain, labels, burn=0, truths=None, path=None):
    ndim = chain.shape[2]
    flat = chain[burn:].reshape((-1, ndim)).copy()
    if flat.shape[1] != len(labels):
        raise ValueError("labels must have same length as chain dimension")
    fig = corner.corner(
        flat, 
        labels=labels, 
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        truths=truths,
        title_kwargs={"fontsize": 12},
    )
    if path is not None:
        fig.savefig(path)
    fig.tight_layout()

def plot_fit(func, zdata, wdata, chain, ndim, n=50000, alpha=0.2, path=None, dpi=100):
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
        fzs[i] = func.fz(zs, theta ,1.)
        fws[i] = func.fw(ws, theta ,1.)
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
    axes[1].set_xlabel(r'$w$ [km/s]')
    axes[1].set_ylim(np.min(np.exp(fw_log_mean - 1.9600*fw_log_std)), np.max(np.exp(fw_log_mean)*1.5))
    axes[1].set_xlim(ws.min(), ws.max())
    axes[1].set_yscale('log')
    if path is not None:
        fig.savefig(path, dpi=dpi)

def style(name='seaborn-v0_8-deep'):
    plt.style.use(name)
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern"]}
    plt.rcParams.update(params)