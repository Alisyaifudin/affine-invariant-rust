#![allow(non_snake_case)]

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;

mod ensemble;
mod prob;
mod stats;
mod utils;

const NDIM: usize = 3;

#[pyfunction]
fn sample_z<'py>(py: Python<'py>, size: usize) -> &PyArray1<f64> {
    let mut rng = rand::thread_rng();
    let u_vec: Vec<f64> = (0..size).map(|_| rng.gen()).collect();
    let u = Array1::from(u_vec);
    let array = utils::zu(u, 2.0);
    array.into_pyarray(py)
}

#[pyfunction]
fn log_prob<'py>(
    py: Python<'py>,
    th: PyReadonlyArray2<f64>,
    d: PyReadonlyArray2<f64>,
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
) -> &'py PyArray1<f64> {
    let theta = th.as_array().to_owned();
    let data = d.as_array().to_owned();
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();
    if theta.raw_dim()[1] != NDIM {
        panic!("theta must be a {}-dimensional vector", NDIM);
    }
    if data.raw_dim()[0] != 3 {
        panic!("data must be a {}-dimensional vector", 3);
    }
    let array = prob::log_prob(&theta, &data, &locs, &scales);
    array.into_pyarray(py)
}

#[pyfunction]
fn generate_data<'py>(py: Python<'py>, N: usize, m: f64, b: f64, f: f64) -> &'py PyArray2<f64> {
    let array = utils::generate_data(&N, &m, &b, &f);
    array.into_pyarray(py)
}

#[pyfunction]
fn generate_p0<'py>(
    py: Python<'py>,
    nwalkers: usize,
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
) -> &'py PyArray2<f64> {
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();

    let atan_m_0 = stats::uniform::rvs(&nwalkers, Some(locs[0]), Some(scales[0]));
    let b_0 = stats::uniform::rvs(&nwalkers, Some(locs[1]), Some(scales[1]));
    let log_f_0 = stats::uniform::rvs(&nwalkers, Some(locs[2]), Some(scales[2]));
    let p0 = ndarray::stack(
        ndarray::Axis(1),
        &[atan_m_0.view(), b_0.view(), log_f_0.view()],
    )
    .unwrap();
    p0.into_pyarray(py)
}

#[pyfunction]
fn mcmc<'py>(
    py: Python<'py>,
    nsteps: usize,
    nwalkers: usize,
    data: PyReadonlyArray2<f64>,
    p0: PyReadonlyArray2<f64>,
    locs: PyReadonlyArray1<f64>,
    scales: PyReadonlyArray1<f64>,
    parallel: Option<bool>,
) -> &'py PyArray3<f64> {
    let p0 = p0.as_array().to_owned();
    let data = data.as_array().to_owned();
    let locs = locs.as_array().to_owned();
    let scales = scales.as_array().to_owned();
    let mut sampler = ensemble::EnsembleSampler::new(NDIM, nwalkers, p0, false, data, locs, scales);
    if nwalkers <= NDIM {
        panic!("nwalkers must be greater than 3 (ndims)");
    }
    let parallel = parallel.unwrap_or(false);
    if parallel && (nwalkers % 2) != 0 {
        panic!("nwalkers must be even when using parallelization");
    }
    sampler.run_mcmc(nsteps, parallel, Some(true), None);
    let array = sampler.get_chain();
    array.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn affine_invariant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_z, m)?)?;
    m.add_function(wrap_pyfunction!(log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(generate_data, m)?)?;
    Ok(())
}
