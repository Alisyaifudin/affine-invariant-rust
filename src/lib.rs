use pyo3::prelude::*;
mod ensemble;
mod model;
mod stats;
mod utils;
use model::dm;
use model::line;

/// A Python module implemented in Rust.
#[pymodule]
fn affine_invariant(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sample_z, m)?)?;
    m.add_function(wrap_pyfunction!(dm::solve_potential, m)?)?;
    m.add_function(wrap_pyfunction!(dm::f, m)?)?;
    m.add_function(wrap_pyfunction!(dm::potential, m)?)?;
    m.add_function(wrap_pyfunction!(dm::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(dm::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(dm::fz, m)?)?;
    m.add_function(wrap_pyfunction!(dm::fw, m)?)?;
    m.add_function(wrap_pyfunction!(dm::mcmc, m)?)?;
    // // lineeee
    m.add_function(wrap_pyfunction!(line::generate_data, m)?)?;
    m.add_function(wrap_pyfunction!(line::generate_p0, m)?)?;
    m.add_function(wrap_pyfunction!(line::log_prob, m)?)?;
    m.add_function(wrap_pyfunction!(line::mcmc, m)?)?;
    Ok(())
}
