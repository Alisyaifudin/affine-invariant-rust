use ndarray::{s, Array1, Array2, Axis};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use ode_solvers::SVector;
use pyo3::{exceptions::PyValueError, prelude::*};

pub mod gravity;
pub mod prob;

use crate::ensemble;
use gravity::State;

use self::prob::{RHOB_INDEX, MU0_INDEX, R_INDEX, SIGMAZ_INDEX};

#[pyfunction]
pub fn f<'py>(
    py: Python<'py>,
    u: PyReadonlyArray2<f64>,
    theta: PyReadonlyArray1<f64>,
) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
    let u = u.as_array().to_owned();
    let u0 = u.slice(s![0, ..]).to_owned();
    let u1 = u.slice(s![1, ..]).to_owned();

    let theta = theta.as_array().to_owned();
    let rhob: SVector<f64, 12> = SVector::from_vec(theta.slice(s![RHOB_INDEX]).to_owned().to_vec());
    let sigmaz: SVector<f64, 12> =
        SVector::from_vec(theta.slice(s![SIGMAZ_INDEX]).to_owned().to_vec());
    let mu0 = theta[MU0_INDEX];
    let r = theta[R_INDEX];
    let res = u0.iter().zip(u1.iter())
        .map(|(u0, u1)| {
            gravity::dfz(
                &(*u0, *u1),
                &rhob,
                &sigmaz,
                &mu0,
                &r,
            )
        })
        .collect::<Vec<(f64, f64)>>();
    let phi = res.iter().map(|x| x.0).collect::<Vec<f64>>();
    let kz = res.iter().map(|x| x.1).collect::<Vec<f64>>();
    (phi.into_pyarray(py), kz.into_pyarray(py))
}

#[pyfunction]
pub fn solve_potential<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    z_start: f64,
    z_end: f64,
    dz: f64,
) -> PyResult<&'py PyArray2<f64>> {
    let theta = theta.as_array().to_owned();
    let theta = theta.insert_axis(Axis(0));
    let sol = gravity::solve(theta, z_start, z_end, dz);
    let res = sol
        .iter()
        .map(|s| match &s {
            Some((z, u)) => Ok((z.clone(), u.clone())),
            None => Err(PyValueError::new_err("Failed to solve potential")),
        })
        .collect::<Vec<PyResult<(Vec<f64>, Vec<State>)>>>();
    let integration = &res[0];
    match integration {
        Ok((z, u)) => {
            let phi = u.iter().map(|x| x[0]).collect::<Vec<f64>>();
            let kz = u.iter().map(|x| x[1]).collect::<Vec<f64>>();
            match ndarray::stack(
                Axis(0),
                &[
                    Array1::from_vec(z.clone()).view(),
                    Array1::from_vec(phi).view(),
                    Array1::from_vec(kz).view(),
                ],
            ) {
                Ok(res) => Ok(res.to_pyarray(py)),
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Failed to stack potential: {}",
                        e
                    )))
                }
            }
        }
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to solve potential: {}",
            e
        ))),
    }
}

#[pyfunction]
pub fn potential<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray1<f64>,
    dz: Option<f64>,
) -> &'py PyArray1<f64> {
    let theta = theta.as_array().to_owned();
    let theta = theta.insert_axis(Axis(0));
    let z = z.as_array().to_owned();
    let phi = gravity::potential(z, theta, dz);
    phi.row(0).to_owned().into_pyarray(py)
}

#[pyfunction]
pub fn fzw<'py>(
    py: Python<'py>,
    pos: PyReadonlyArray2<f64>,
    theta: PyReadonlyArray1<f64>,
    dz: Option<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let theta = theta.as_array().to_owned();
    let pos = pos.as_array().to_owned();
    let res = gravity::fzw(&pos, &theta, dz);
    match res {
        Ok(res) => Ok(res.into_pyarray(py)),
        Err(e) => Err(PyValueError::new_err(format!(
            "Failed to calculate fzw: {}",
            e
        ))),
    }
}

#[pyfunction]
pub fn log_prob<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    zdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    wdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
    dz: Option<f64>,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let theta = theta.as_array().to_owned();
    let ndim = theta.raw_dim()[1];
    let (zmid, znum, zerr) = zdata;
    let (zmid, znum, zerr) = (
        zmid.as_array().to_owned(),
        znum.as_array().to_owned(),
        zerr.as_array().to_owned(),
    );
    let (wmid, wnum, werr) = wdata;
    let (wmid, wnum, werr) = (
        wmid.as_array().to_owned(),
        wnum.as_array().to_owned(),
        werr.as_array().to_owned(),
    );
    if ndim != 31 || ndim != 33 {
        Err(PyValueError::new_err(format!(
            "Invalid number of dimensions: {}, expected 31 or 33",
            ndim
        )))?
    }
    let zdata = (zmid, znum, zerr);
    let wdata = (wmid, wnum, werr);
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();
    if ndim == 33 {
        let (prior, posterior) = prob::log_prob1(&theta, zdata, wdata, &locs, &scales, dz);
        Ok((prior.into_pyarray(py), posterior.into_pyarray(py)))
    } else if ndim == 35 {
        let (prior, posterior) = prob::log_prob2(&theta, zdata, wdata, &locs, &scales, dz);
        Ok((prior.into_pyarray(py), posterior.into_pyarray(py)))
    } else {
        Err(PyValueError::new_err(format!(
            "Invalid number of dimensions: {}, expected 31 or 33",
            ndim
        )))?
    }
}

#[pyfunction]
pub fn generate_p0<'py>(
    py: Python<'py>,
    nwalkers: usize,
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
    kind: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();
    if kind == 1 {
        let p0 = prob::generate_p0_1(nwalkers, locs, scales);
        let py_p0 = p0.into_pyarray(py);
        Ok(py_p0)
    } else if kind == 2 {
        let p0 = prob::generate_p0_2(nwalkers, locs, scales);
        let py_p0 = p0.into_pyarray(py);
        Ok(py_p0)
    } else {
        Err(PyValueError::new_err("kind must be either 1 or 2"))
    }
}

#[pyfunction]
pub fn fz<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray1<f64>,
    dz: Option<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let theta = theta.as_array().to_owned();
    let ndim = theta.len();
    let theta = theta.insert_axis(Axis(0));
    let z = z.as_array().to_owned();
    let fz = if ndim == 31 {
        gravity::fz1(z, theta, dz)
    } else {
        gravity::fz2(z, theta, dz)
    };
    let test = fz.sum();
    if test == f64::NAN {
        Err(PyValueError::new_err("Integration failed, curious huh?"))?
    }
    Ok(fz.row(0).to_owned().into_pyarray(py))
}

#[pyfunction]
pub fn fw<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray1<f64>,
    dz: Option<f64>,
) -> PyResult<&'py PyArray1<f64>> {
    let theta = theta.as_array().to_owned();
    let ndim = theta.len();
    let theta = theta.insert_axis(Axis(0));
    let w = w.as_array().to_owned();
    let fw = if ndim == 31 {
        gravity::fw1(w, theta, dz)
    } else {
        gravity::fw2(w, theta, dz)
    };
    let test = fw.sum();
    if test == f64::NAN {
        Err(PyValueError::new_err("Integration failed, curious huh?"))?
    }
    Ok(fw.row(0).to_owned().into_pyarray(py))
}

#[pyfunction]
pub fn mcmc<'py>(
    py: Python<'py>,
    nsteps: usize,
    nwalkers: usize,
    p0: PyReadonlyArray2<f64>,
    zdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    wdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
    parallel: Option<bool>,
    dz: Option<f64>,
    batch: Option<usize>,
    verbose: Option<bool>,
) -> PyResult<&'py PyArray3<f64>> {
    let batch = batch.unwrap_or(2);
    if nwalkers <= 31 {
        panic!("nwalkers must be greater than {} (ndims)", 31);
    }
    let parallel = parallel.unwrap_or(false);
    if parallel && (nwalkers % 2) != 0 {
        panic!("nwalkers must be even when using parallelization");
    }
    let (zmid, znum, zerr) = zdata;
    let (zmid, znum, zerr) = (
        zmid.as_array().to_owned(),
        znum.as_array().to_owned(),
        zerr.as_array().to_owned(),
    );
    let zdata = (zmid, znum, zerr);
    let (wmid, wnum, werr) = wdata;
    let (wmid, wnum, werr) = (
        wmid.as_array().to_owned(),
        wnum.as_array().to_owned(),
        werr.as_array().to_owned(),
    );
    let wdata = (wmid, wnum, werr);
    let p0 = p0.as_array().to_owned();
    let ndim = p0.raw_dim()[1];
    if ndim != 31 && ndim != 33 {
        Err(PyValueError::new_err("p0 must have dimension 31 or 33"))?
    }
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();
    let log_prob = move |theta: &Array2<f64>| {
        if ndim == 31 {
            prob::log_prob1(
                &theta,
                zdata.to_owned(),
                wdata.to_owned(),
                &locs,
                &scales,
                dz,
            )
        } else {
            prob::log_prob2(
                &theta,
                zdata.to_owned(),
                wdata.to_owned(),
                &locs,
                &scales,
                dz,
            )
        }
    };
    let mut sampler =
        ensemble::EnsembleSampler::new(ndim, nwalkers, p0, parallel, Box::new(log_prob));

    sampler.run_mcmc(nsteps, parallel, batch, verbose.unwrap_or(false), 2.);

    let array = sampler.get_chain();
    Ok(array.into_pyarray(py))
}

#[pyfunction]
pub fn sample<'py>(
    py: Python<'py>,
    nsteps: usize,
    nwalkers: usize,
    pos: PyReadonlyArray2<f64>,
    theta: PyReadonlyArray1<f64>,
    parallel: Option<bool>,
    dz: Option<f64>,
    batch: Option<usize>,
    verbose: Option<bool>,
) -> PyResult<&'py PyArray3<f64>> {
    let batch = batch.unwrap_or(2);
    if nwalkers <= 2 {
        panic!("nwalkers must be greater than {} (ndims)", 2);
    }
    let parallel = parallel.unwrap_or(false);
    if parallel && (nwalkers % 2) != 0 {
        panic!("nwalkers must be even when using parallelization");
    }
    let theta = theta.as_array().to_owned();

    let theta_dim = theta.len();
    if theta_dim != 31 && theta_dim != 33 {
        Err(PyValueError::new_err("theta must have dimension 31 or 33"))?
    }
    let pos = pos.as_array().to_owned();
    let ndim = pos.raw_dim()[1];
    let log_prob = move |pos: &Array2<f64>| prob::sample(&pos, &theta.to_owned(), dz);
    let mut sampler =
        ensemble::EnsembleSampler::new(ndim, nwalkers, pos, parallel, Box::new(log_prob));

    sampler.run_mcmc(nsteps, parallel, batch, verbose.unwrap_or(false), 2.);

    let array = sampler.get_chain();
    Ok(array.into_pyarray(py))
}
