from typing import List
import affine_invariant as af
import numpy as np

def generate_data(n: int, locs: List[float]) -> np.ndarray:
    """
    Generate data for line example.

    Parameters
    ----------
        n (`int`): number data points
        locs (`List[float]`): the value of m, b, and f

    Returns
    -------
        data (`np.ndarray`): x, y, and yerr
    """
    data = af.line.generate_data(n, locs)
    return data

def generate_p0(nwalkers: int, locs: List[float], scales: List[float]) -> np.ndarray:
    """
    Generate initial positions for walkers. The walkers are sampled from uniform distributions
    from `loc` to `loc` + `scale`.

    Parameters
    ----------
        nwalkers (`int`): number of walkers
        locs (`List[float]`): the loc of m, b, and f
        scales (`List[float]`): the scale of m, b, and f

    Returns
    -------
        p0 (`np.ndarray`): initial positions for walkers with shape of (nwalkers, 3)
    """
    p0 = af.line.generate_p0(nwalkers, locs, scales)
    return p0

def log_prob(theta: np.ndarray, data: np.ndarray, locs: List[float], scales: List[float]) -> float:
    """
    Log probability function for the line example.

    Parameters
    ----------
        theta (`np.ndarray`): parameters of the model
        data (`np.ndarray`): x, y, and yerr
        locs (`List[float]`): the loc of m, b, and f
        scales (`List[float]`): the scale of m, b, and f

    Returns
    -------
        log_prob (`float`): log probability
    """
    log_prob = af.line.log_prob(theta, data, locs, scales)
    return log_prob

def run_mcmc(nsteps: int, nwalkers: int, p0: np.ndarray, data: np.ndarray, locs: List[float], scales: List[float], parallel=False, batch=2, verbose=False) -> np.ndarray:
    """
    Run the MCMC sampler.

    Parameters
    ----------
        nsteps (`int`): number of steps to run
        nwalkers (`int`): number of walkers to use
        p0 (`np.ndarray`): initial positions for walkers with shape of (nwalkers, 3)
        data (`np.ndarray`): x, y, and yerr
        locs (`List[float]`): the loc of m, b, and f
        scales (`List[float]`): the scale of m, b, and f
        parallel (`bool`): whether to run in parallel
        batch (`int`): number of batch to run in parallel
        verbose (`bool`): whether to print out the acceptance rate
    Returns
    -------
        samples (`np.ndarray`): samples from the posterior distribution
    """
    res = af.line.mcmc(nsteps, nwalkers, p0, data, locs, scales, parallel=parallel, batch=batch, verbose=verbose)
    return res