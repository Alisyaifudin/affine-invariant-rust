import affine_invariant as af
import numpy as np

def zu(n: int) -> np.ndarray:
    """
    Inverse of the transformation function z(u) = (1/a)*(1+(a-1)*u)**2.
    Return n random numbers from this distribution.

    Parameters
    ----------
        n (`int`): number of random numbers to generate

    Returns
    -------
        z (`np.ndarray`): n random numbers from the distribution
    """
    return af.sample_z(n)

def log_prob(x: np.ndarray) -> np.ndarray:
    """
    Logarithm of the probability density function of the 1D distribution
    p(x) = 1/(2*sqrt(2*pi))*exp(-x**2/2).

    Parameters
    ----------
        x (`np.ndarray`): x values

    Returns
    -------
        log_p (`np.ndarray`): log(p(x))
    """
    return af.log_prob(x)