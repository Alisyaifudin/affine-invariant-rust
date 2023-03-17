use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::ensemble;

pub mod prob;

#[pyfunction]
pub fn generate_data<'py>(
    py: Python<'py>,
    n: usize,
    locs: Vec<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let data = prob::generate_data(n, locs);
    match data {
        Ok(data) => Ok(data.into_pyarray(py)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}

#[pyfunction]
pub fn generate_p0<'py>(
    py: Python<'py>,
    nwalkers: usize,
    locs: Vec<f64>,
    scales: Vec<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let data = prob::generate_p0(nwalkers, locs, scales);
    match data {
        Ok(data) => Ok(data.into_pyarray(py)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
    }
}

#[pyfunction]
pub fn log_prob<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray2<f64>,
    data: PyReadonlyArray2<f64>,
    locs: Vec<f64>,
    scales: Vec<f64>,
) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
    let theta = theta.as_array().to_owned();
    let data = data.as_array().to_owned();
    let (prior, posterior) = prob::log_prob(&theta, &data, &locs, &scales);
    (prior.into_pyarray(py), posterior.into_pyarray(py))
}

#[pyfunction]
pub fn mcmc<'py>(
    py: Python<'py>,
    nsteps: usize,
    nwalkers: usize,
    p0: PyReadonlyArray2<f64>,
    data: PyReadonlyArray2<f64>,
    locs: Vec<f64>,
    scales: Vec<f64>,
    parallel: Option<bool>,
    batch: Option<usize>,
    verbose: Option<bool>,
) -> PyResult<&'py PyArray3<f64>> {
    let batch = batch.unwrap_or(2);
    if nwalkers <= 3 {
        Err(PyValueError::new_err("nwalkers must be greater than 3"))?
    }
    let parallel = parallel.unwrap_or(false);
    let p0 = p0.as_array().to_owned();
    let data = data.as_array().to_owned();
    let ndim = p0.raw_dim()[1];
    let log_prob = move |theta: &Array2<f64>| prob::log_prob(&theta, &data, &locs, &scales);
    let mut sampler =
        ensemble::EnsembleSampler::new(ndim, nwalkers, p0, parallel, Box::new(log_prob));

    sampler.run_mcmc(nsteps, parallel, batch, verbose.unwrap_or(false), 2.);
    let array = sampler.get_chain();
    Ok(array.into_pyarray(py))
}
