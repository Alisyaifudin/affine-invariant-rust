import affine_invariant as af
from numpy import ndarray
from typing import Tuple

def f(z: ndarray, u: ndarray, theta: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Calculate the gradients of the CBE for axisymmetric distribution near
    galactic plane.

    Parameters
    ----------
        z (`ndarray[1]`): the distance from the galactic plane [pc]
        u (`ndarray[2]`): the gradients of the potential (Kz) and gradient of Kz.
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
    """
    val = af.dm.f(z, u, theta)
    return val

def solve_potential(theta: ndarray, z_start: float, z_end: float, dz: float) -> ndarray:
    """
    Solve the differential equation for the potential of axisymmetric distribution near galactic plane.

    Parameters
    ----------
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
        z_start (`float`): the initial distance from the galactic plane [pc]
        z_end (`float`): the final distance from the galactic plane [pc]
        dz (`float`): the step size of the distance from the galactic plane [pc]
    """
    val = af.dm.solve_potential(theta, z_start, z_end, dz)
    return val

def potential(z: ndarray, theta: ndarray, dz=10.) -> ndarray:
    """
    Calculate the potential of axisymmetric distribution near galactic plane.

    Parameters
    ----------
        z (`ndarray[1]`): the distance from the galactic plane [pc]
        theta (`ndarray[2]`): parameters of the model, compose of
            `theta[:, :12]`: baryonic density at midplane [Msun/pc^3]
            `theta[:, 12:24]`: baryonic dispersion (km/s)
            `theta[:, 24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[:, 25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[:, 26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[:, 27]`: sun height from midplane [pc]
            `theta[:, 28]`: the average vertical velocity of the sun [km/s]
            `theta[:, 29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[:, 30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[:, 31]`: the second dispersion velocity term [km/s]
            `theta[:, 32]`: the second normalization constant for the dispersion velocity
        dz (`Optional[float]`): the step size of the distance from the galactic plane [pc] (default: 10.)
    """
    val = af.dm.potential(z, theta, dz)
    return val

def log_prob(theta: ndarray, zdata: Tuple[ndarray, ndarray, ndarray], wdata: Tuple[ndarray, ndarray, ndarray], locs: ndarray, scales: ndarray, dz=10.) -> ndarray:
    """
    Calculate the log probability of the model.

    Parameters
    ----------
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
        zdata (`Tuple[ndarray[1], ndarray[1], ndarray[1]]`): the distance from the galactic plane [pc], the number density [pc^-3], and the error of the number density [pc^-3]
        wdata (`Tuple[ndarray[1], ndarray[1], ndarray[1]]`): the distance from the galactic plane [pc], the number density of velocity [s/km], and the error of the number density of velocity [s/km]
        locs (`ndarray[1]`): the boundaries for the prior
        scales (`ndarray[1]`): the scales for the prior
        dz (`Optional[float]`): the step size of the distance from the galactic plane [pc] (default: 10.)
    """
    val = af.dm.log_prob(theta, zdata, wdata, locs, scales, dz)
    return val

def generate_p0(nwalkers: int, locs: ndarray, scales: ndarray, kind: int) -> ndarray:
    """
    Generate the initial position of walkers. The initial position is generated from the prior.
    The prior is either uniform or Gaussian.

    Parameters
    ----------
        nwalkers (`int`): the number of walkers
        locs (`ndarray[1]`): the boundaries for the prior
        scales (`ndarray[1]`): the scales for the prior
        kind (`int`): either for 31 (=1) or 33 (=2) parameters
    """
    val = af.dm.generate_p0(nwalkers, locs, scales, kind=kind)
    return val

def fz(z: ndarray, theta: ndarray, dz=10.) -> ndarray:
    """
    Calculate the number density of stars at the distance from the galactic plane.

    Parameters
    ----------
        z (`ndarray[1]`): the distance from the galactic plane [pc]
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
        dz (`Optional[float]`): the step size of the distance from the galactic plane [pc] (default: 10.)
    """
    val = af.dm.fz(z, theta, dz)
    return val

def fw(w: ndarray, theta: ndarray, dz=10.) -> ndarray:
    """
    Calculate the number density of velocity near the midplane.

    Parameters
    ----------
        w (`ndarray[1]`): the vertical velocity [km/s]
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
        dz (`Optional[float]`): the step size of the distance from the galactic plane (for integration) [pc] (default: 10.)
    """
    val = af.dm.fw(w, theta, dz)
    return val

def run_mcmc(nsteps: int, nwalkers: int, p0: ndarray, zdata: Tuple[ndarray, ndarray, ndarray], wdata: Tuple[ndarray, ndarray, ndarray], locs: ndarray, scales: ndarray, parallel=False, dz=10., batch=2, verbose=False) -> ndarray:
    """
    Run the MCMC.

    Parameters
    ----------
        nsteps (`int`): the number of steps
        nwalkers (`int`): the number of walkers
        p0 (`ndarray[1]`): the initial position of walkers
        zdata (`Tuple[ndarray[1], ndarray[1], ndarray[1]]`): the distance from the galactic plane [pc], the number density [pc^-3], and the error of the number density [pc^-3]
        wdata (`Tuple[ndarray[1], ndarray[1], ndarray[1]]`): the distance from the galactic plane [pc], the number density of velocity [s/km], and the error of the number density of velocity [s/km]
        locs (`ndarray[1]`): the boundaries for the prior
        scales (`ndarray[1]`): the scales for the prior
        parallel (`Optional[bool]`): whether to use parallel algorithm (default: False)
        dz (`Optional[float]`): the step size of the distance from the galactic plane (for integration) [pc] (default: 10.)
        batch (`Optional[int]`): the number of batches (default: 2)
        verbose (`Optional[bool]`): whether to print the acceptance rate (default: False)
    """
    val = af.dm.mcmc(nsteps, nwalkers, p0, zdata, wdata, locs, scales, parallel=parallel, dz=dz, batch=batch, verbose=verbose)
    return val

def fzw(pos: ndarray, theta: ndarray, dz=1.) -> ndarray:
    """
    Calculate the unnormalized distribution function fzw

    Parameters
    ----------
        pos (`ndarray[2]`): the distance from the galactic plane [pc] and the vertical velocity [km/s] (num, 2)
        theta (`ndarray[1]`): parameters of the model, compose of
            `theta[:12]`: baryonic density at midplane [Msun/pc^3]
            `theta[12:24]`: baryonic dispersion (km/s)
            `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
            `theta[25]`: number density of dark matter halo near midplane [pc^-3]
            `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
            `theta[27]`: sun height from midplane [pc]
            `theta[28]`: the average vertical velocity of the sun [km/s]
            `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
            `theta[30]`: normalization constant for the dispersion velocity
        -- `optional` --
            `theta[31]`: the second dispersion velocity term [km/s]
            `theta[32]`: the second normalization constant for the dispersion velocity
        dz (`Optional[float]`): the step size of the distance from the galactic plane (for integration) [pc] (default: 1.)
    """
    val = af.dm.fzw(pos, theta, dz)
    return val

def sample(nsteps: int, nwalkers: int, pos: ndarray, theta: ndarray, parallel=False, dz=1., batch=2, verbose=False) -> ndarray:
        """
        Sample the distribution function fzw
        
        Parameters
        ----------
            nsteps (`int`): the number of steps
            nwalkers (`int`): the number of walkers
            pos (`ndarray[2]`): the initial position of walkers (num, 2)
            theta (`ndarray[1]`): parameters of the model, compose of
                `theta[:12]`: baryonic density at midplane [Msun/pc^3]
                `theta[12:24]`: baryonic dispersion (km/s)
                `theta[24]`: dark matter halo density at midplane [Msun/pc^3]
                `theta[25]`: number density of dark matter halo near midplane [pc^-3]
                `theta[26]`: the radial term (related to Oort's constant) [Msun/pc^3]
                `theta[27]`: sun height from midplane [pc]
                `theta[28]`: the average vertical velocity of the sun [km/s]
                `theta[29]`: the dispersion of the vertical velocity stars analized [km/s]
                `theta[30]`: normalization constant for the dispersion velocity
            -- `optional` --
                `theta[31]`: the second dispersion velocity term [km/s]
                `theta[32]`: the second normalization constant for the dispersion velocity
            parallel (`Optional[bool]`): whether to use parallel algorithm (default: False)
            dz (`Optional[float]`): the step size of the distance from the galactic plane (for integration) [pc] (default: 1.)
            batch (`Optional[int]`): the number of batches (default: 2)
            verbose (`Optional[bool]`): whether to print the acceptance rate (default: False)
        """
        val = af.dm.sample(nsteps, nwalkers, pos, theta, parallel=parallel, dz=dz, batch=batch, verbose=verbose)
        return val