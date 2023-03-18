use pyo3::prelude::*;
mod ensemble;
mod model;
mod stats;
mod utils;
use model::dddm;
use model::dm;
use model::line;
use model::mond;
use model::no;

/// A Python module implemented in Rust.
#[pymodule]
fn affine_invariant(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    dm(py, m)?;
    line(py, m)?;
    dddm(py, m)?;
    no(py, m)?;
    mond(py, m)?;
    Ok(())
}

fn dm(py: Python<'_>, affine_invariant: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "dm")?;
    m.add_function(wrap_pyfunction!(dm::solve_potential, m)?)?;
    m.add_function(wrap_pyfunction!(dm::f, m)?)?;
    m.add_function(wrap_pyfunction!(dm::potential, m)?)?;
    m.add_function(wrap_pyfunction!(dm::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(dm::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(dm::fz, m)?)?;
    m.add_function(wrap_pyfunction!(dm::fw, m)?)?;
    m.add_function(wrap_pyfunction!(dm::mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(dm::sample, m)?)?;
    m.add_function(wrap_pyfunction!(dm::fzw, m)?)?;
    affine_invariant.add_submodule(m)?;
    Ok(())
}

fn dddm(py: Python<'_>, affine_invariant: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "dddm")?;
    m.add_function(wrap_pyfunction!(dddm::solve_potential, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::f, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::potential, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::fz, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::fw, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::sample, m)?)?;
    m.add_function(wrap_pyfunction!(dddm::fzw, m)?)?;
    affine_invariant.add_submodule(m)?;
    Ok(())
}

fn no(py: Python<'_>, affine_invariant: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "no")?;
    m.add_function(wrap_pyfunction!(no::solve_potential, m)?)?;
    m.add_function(wrap_pyfunction!(no::f, m)?)?;
    m.add_function(wrap_pyfunction!(no::potential, m)?)?;
    m.add_function(wrap_pyfunction!(no::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(no::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(no::fz, m)?)?;
    m.add_function(wrap_pyfunction!(no::fw, m)?)?;
    m.add_function(wrap_pyfunction!(no::mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(no::sample, m)?)?;
    m.add_function(wrap_pyfunction!(no::fzw, m)?)?;
    affine_invariant.add_submodule(m)?;
    Ok(())
}

fn mond(py: Python<'_>, affine_invariant: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "mond")?;
    m.add_function(wrap_pyfunction!(mond::solve_potential, m)?)?;
    m.add_function(wrap_pyfunction!(mond::f, m)?)?;
    m.add_function(wrap_pyfunction!(mond::potential, m)?)?;
    m.add_function(wrap_pyfunction!(mond::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(mond::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(mond::fz, m)?)?;
    m.add_function(wrap_pyfunction!(mond::fw, m)?)?;
    m.add_function(wrap_pyfunction!(mond::mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(mond::sample, m)?)?;
    m.add_function(wrap_pyfunction!(mond::fzw, m)?)?;
    affine_invariant.add_submodule(m)?;
    Ok(())
}

fn line(py: Python<'_>, affine_invariant: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "line")?;
    m.add_function(wrap_pyfunction!(line::generate_data, m)?)?;
    m.add_function(wrap_pyfunction!(line::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(line::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(line::mcmc, m)?)?;
    affine_invariant.add_submodule(m)?;
    Ok(())
}
