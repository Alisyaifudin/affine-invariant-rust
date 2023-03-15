#![allow(non_snake_case)]
use ndarray::{Array1, Axis, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::Rng;

mod ensemble;
mod gravity;
mod stats;
mod utils;

const NDIM: usize = 31;

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
    zdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
    ),
    wdata: (
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        PyReadonlyArray1<f64>,
        f64,
    ),
) -> &'py PyArray1<f64> {
    let theta = th.as_array().to_owned();
    let (zmid, znum, zerr) = zdata;
    let (zmid, znum, zerr) = (
        zmid.as_array().to_owned(),
        znum.as_array().to_owned(),
        zerr.as_array().to_owned(),
    );
    let (wmid, wnum, werr, zbound) = wdata;
    let (wmid, wnum, werr) = (
        wmid.as_array().to_owned(),
        wnum.as_array().to_owned(),
        werr.as_array().to_owned(),
    );
    if theta.raw_dim()[1] != NDIM {
        panic!("theta must be a {}-dimensional vector", NDIM);
    }
    let zdata = (zmid, znum, zerr);
    let wdata = (wmid, wnum, werr, zbound);
    let array = gravity::prob::dm::log_prob1(&theta, &zdata, &wdata, &None);
    array.into_pyarray(py)
}

#[pyfunction]
fn generate_data<'py>(py: Python<'py>, N: usize, m: f64, b: f64, f: f64) -> &'py PyArray2<f64> {
    let array = utils::generate_data(&N, &m, &b, &f);
    array.into_pyarray(py)
}

// #[pyfunction]
// fn generate_p0<'py>(
//     py: Python<'py>,
//     nwalkers: usize,
//     locs: PyReadonlyArray1<f64>,
//     scales: PyReadonlyArray1<f64>,
// ) -> &'py PyArray2<f64> {
//     let locs = locs.as_array().to_owned();
//     let scales = scales.as_array().to_owned();

//     let atan_m_0 = stats::uniform::rvs(&nwalkers, locs[0], Some(scales[0]));
//     let b_0 = stats::uniform::rvs(&nwalkers, Some(locs[1]), Some(scales[1]));
//     let log_f_0 = stats::uniform::rvs(&nwalkers, Some(locs[2]), Some(scales[2]));
//     let p0 = ndarray::stack(
//         ndarray::Axis(1),
//         &[atan_m_0.view(), b_0.view(), log_f_0.view()],
//     )
//     .unwrap();
//     p0.into_pyarray(py)
// }

#[pyfunction]
fn mcmc<'py>(
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
        f64,
    ),
    parallel: Option<bool>,
) -> &'py PyArray3<f64> {
    if nwalkers <= NDIM {
        panic!("nwalkers must be greater than {} (ndims)", NDIM);
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
    let (wmid, wnum, werr, zbound) = wdata;
    let (wmid, wnum, werr) = (
        wmid.as_array().to_owned(),
        wnum.as_array().to_owned(),
        werr.as_array().to_owned(),
    );
    let wdata = (wmid, wnum, werr, zbound);
    let p0 = p0.as_array().to_owned();
    if (p0.raw_dim()[0], p0.raw_dim()[1]) != (nwalkers, NDIM) {
        panic!("p0 must have shape (nwalkers, ndims)");
    }
    let mut sampler = ensemble::EnsembleSampler1::new(NDIM, nwalkers, p0, parallel, zdata, wdata, None);
    sampler.run_mcmc(nsteps, parallel, Some(true), None);
    let array = sampler.get_chain();
    array.into_pyarray(py)
}

// #[pyfunction]
// fn f<'py>(
//     py: Python<'py>,
//     z: PyReadonlyArray1<f64>,
//     u: PyReadonlyArray2<f64>,
//     theta: PyReadonlyArray1<f64>,
//     nb: usize,
// ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
//     let z = z.as_array().to_owned();
//     let u = u.as_array().to_owned();
//     let u0 = u.slice(s![0, ..]).to_owned();
//     let u1 = u.slice(s![1, ..]).to_owned();

//     let theta = theta.as_array().to_owned();
//     let skip = 2 * nb;
//     let rhob: SVector<f64, 12> = SVector::from_vec(theta.slice(s![0..nb]).to_owned().to_vec());
//     let sigmaz: SVector<f64, 12> = SVector::from_vec(theta.slice(s![nb..skip]).to_owned().to_vec());
//     let rhoDM = theta[skip];
//     let sigmaDD = theta[skip + 1];
//     let hDD = theta[skip + 2];
//     let nu0 = theta[skip + 3];
//     let R = Some(theta[skip + 4]);
//     let res = z
//         .iter()
//         .zip(u0.iter().zip(u1.iter()))
//         .map(|(z, u)| gravity::dm::dfz(&(*u.0, *u.1), z, &rhob, &sigmaz, &rhoDM, &sigmaDD, &hDD, R))
//         .collect::<Vec<(f64, f64)>>();
//     let phi = res.iter().map(|x| x.0).collect::<Vec<f64>>();
//     let kz = res.iter().map(|x| x.1).collect::<Vec<f64>>();
//     (phi.into_pyarray(py), kz.into_pyarray(py))
// }

#[pyfunction]
fn solve_potential<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<f64>,
    z_start: f64,
    z_end: f64,
    dz: f64,
) -> (&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>) {
    let theta = theta.as_array().to_owned();
    let theta = theta.insert_axis(Axis(0));
    let res = gravity::dm::solve(theta, z_start, z_end, dz);
    let (z, u) = res[0].clone().unwrap();
    let phi = u.iter().map(|x| x[0]).collect::<Vec<f64>>();
    let kz = u.iter().map(|x| x[1]).collect::<Vec<f64>>();
    (
        Array1::from_vec(z).into_pyarray(py),
        Array1::from_vec(phi).into_pyarray(py),
        Array1::from_vec(kz).into_pyarray(py),
    )
}

#[pyfunction]
fn potential<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray2<f64>,
    dz: Option<f64>
) -> &'py PyArray2<f64> {
    let theta = theta.as_array().to_owned();
    let sh = theta.raw_dim();
    let z = z.as_array().to_owned();
    let phi = gravity::dm::potential(z, theta, dz);
    let mut tmp = Array2::zeros(sh);
    phi.iter().zip(tmp.axis_iter_mut(Axis(0))).for_each(|(phi, mut tmp)| {
        tmp.assign(phi);
    });
    tmp.into_pyarray(py)
}

#[pyfunction]
fn fz1<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<f64>,
    theta: PyReadonlyArray2<f64>,
    dz: Option<f64>
) -> &'py PyArray2<f64> {
    let theta = theta.as_array().to_owned();
    let z = z.as_array().to_owned();
    let fz = gravity::dm::fz1(z, theta, dz);
    fz.into_pyarray(py)
}

#[pyfunction]
fn fw1<'py>(
    py: Python<'py>,
    w: PyReadonlyArray1<f64>,
    zbound: f64,
    theta: PyReadonlyArray2<f64>,
    dz: Option<f64>
) -> &'py PyArray2<f64> {
    let theta = theta.as_array().to_owned();
    let w = w.as_array().to_owned();
    let fw = gravity::dm::fw1(w, zbound, theta, dz);
    fw.into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn affine_invariant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_z, m)?)?;
    m.add_function(wrap_pyfunction!(log_prob, m)?)?;
    // m.add_function(wrap_pyfunction!(generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(generate_data, m)?)?;
    m.add_function(wrap_pyfunction!(solve_potential, m)?)?;
    // m.add_function(wrap_pyfunction!(f, m)?)?;
    m.add_function(wrap_pyfunction!(potential, m)?)?;
    m.add_function(wrap_pyfunction!(fz1, m)?)?;
    m.add_function(wrap_pyfunction!(fw1, m)?)?;
    Ok(())
}
