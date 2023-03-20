import numpy as np
from time import time
import warnings
warnings.filterwarnings("ignore")
import os
import sys
from os.path import join, abspath
parent_dir = os.path.dirname(os.getcwd())
root_dir = abspath(join(parent_dir, '..'))
current_dir = os.path.curdir
sys.path.append(root_dir)
from mcmc import no
from init import init
from scipy.stats import norm
from scipy.optimize import curve_fit

model_kind = int(sys.argv[1])
data_kind = int(sys.argv[2])
ndim, nwalkers = init(model_kind)

zdata = np.loadtxt(join(parent_dir, 'data', f'z{data_kind}.csv'), skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]

wdata = np.loadtxt(join(parent_dir, 'data', f'w{data_kind}.csv'), skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]

def gauss(x, mu, sigma, A):
    return A * norm.pdf(x, mu, sigma)

def kind1():
    poptz, _ = curve_fit(gauss, zmid, znum, p0=[0, 100, 1])
    poptw, _ = curve_fit(gauss, wmid, wnum, p0=[0, 10, 1])

    log_nu0 = np.log(poptz[2]/(np.sqrt(2*np.pi)*poptz[1]))
    print(poptw)
    log_sigmaw1 = np.log(poptw[1])
    log_a1 = np.log(poptw[2])

    locs = dict(
        log_nu0=log_nu0-1,
        R=3.4E-3,
        zsun=-50,
        w0=-10,
        log_sigmaw1=log_sigmaw1-1,
        log_a1=log_a1-1,
    )

    scales = dict(
        log_nu0=2,
        R=0.6E-3,
        zsun=100,
        w0=5,
        log_sigmaw1=2,
        log_a1=2,
    )
    return locs, scales

def dgauss(x, mu, sigma1, A1, sigma2, A2):
    return A1 * norm.pdf(x, mu, sigma1) + A2 * norm.pdf(x, mu, sigma2)
def kind2():
    for i in range(100):
        p0 = np.array([0])
        sigma = np.random.uniform(1, 20, size=2)
        a = np.min(sigma)/sigma
        p0 = np.append(p0, sigma)
        p0 = np.append(p0, a)
        try:
            poptz, _ = curve_fit(gauss, zmid, znum, p0=[0, 100, 1])
            poptw, _ = curve_fit(dgauss, wmid, wnum, p0=p0)
            log_nu0 = np.log(poptz[2]/(np.sqrt(2*np.pi)*poptz[1]))
            log_sigmaw1 = np.log(poptw[1])
            log_a1 = np.log(poptw[2])
            log_sigmaw2 = np.log(poptw[3])
            log_a2 = np.log(poptw[4])
            if np.isnan(log_a1) or np.isnan(log_a2) or log_sigmaw1 < 0 or log_sigmaw2 < -0:
                print(f"ooppss... try again... {i}")
                continue
            locs = dict(
                log_nu0=log_nu0-1,
                R=3.4E-3,
                zsun=-50,
                w0=-10,
                log_sigmaw1=log_sigmaw1-1,
                log_a1=log_a1-1.5,
                log_sigmaw2=log_sigmaw2-1,
                log_a2=log_a2-1.5,
            )

            scales = dict(
                log_nu0=2,
                R=0.6E-3,
                zsun=100,
                w0=5,
                log_sigmaw1=2,
                log_a1=3,
                log_sigmaw2=2,
                log_a2=3,
            )
            
            return locs, scales
        except RuntimeError:
            print("RuntimeError, try again...")
            continue
locs, scales = locals()[f'kind{model_kind}']()
print(locs)
print(scales)
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

print("generate p0...")
t0 = time()
p0 = no.generate_p0(nwalkers, locs, scales, kind=model_kind)
print(f"generating p0 took {time()-t0:.2f} seconds")

np.save(join(current_dir, 'data', f'locs-{model_kind}-no.npy'), locs)
np.save(join(current_dir, 'data', f'scales-{model_kind}-no.npy'), scales)
np.save(join(current_dir, 'data', f'p0-{model_kind}-no.npy'), p0)
