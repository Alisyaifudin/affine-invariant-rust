import affine_invariant as af
import numpy as np

def run_mcmc(nsteps: int, nwalkers: int, parallel=False) -> np.ndarray:
    """
    Run the MCMC sampler.

    Parameters
    ----------
        nsteps (`int`): number of steps to run
        nwalkers (`int`): number of walkers to use
        parallel (`bool`): whether to run in parallel

    Returns
    -------
        samples (`np.ndarray`): samples from the posterior distribution
    """
    res = af.mcmc(nsteps, nwalkers, parallel=parallel)
    return res