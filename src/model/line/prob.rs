use crate::stats::{normal, uniform};
use crate::utils;
use crate::utils::method::{Exponential, Power};
use ndarray::{s, Array1, Array2, Axis};

pub fn generate_data(n: usize, locs: Vec<f64>) -> Result<Array2<f64>, &'static str> {
    let (m_true, b_true, f_true) = (locs[0], locs[1], locs[2]);
    let x = 10. * uniform::rvs(&n, &0., &1.);
    let yerr = 0.1 + 0.5 * uniform::rvs(&n, &0., &1.);
    let y = m_true * x.clone() + b_true;
    let y = y.clone() + (y * f_true).mapv_into(|v| v.abs()) * normal::rvs(&n, &0., &1.);
    let y = y + yerr.clone() * normal::rvs(&n, &0., &1.);
    match ndarray::stack(Axis(1), &[x.view(), y.view(), yerr.view()]) {
        Ok(x) => Ok(x),
        Err(_) => Err("Failed to generate data"),
    }
}

pub fn generate_p0(
    nwalkers: usize,
    locs: Vec<f64>,
    scales: Vec<f64>,
) -> Result<Array2<f64>, &'static str> {
    let (atan_m_loc, b_loc, log_f_loc) = (locs[0], locs[1], locs[2]);
    let (atan_m_scale, b_scale, log_f_scale) = (scales[0], scales[1], scales[2]);
    let atan_m = uniform::rvs(&nwalkers, &atan_m_loc, &atan_m_scale);
    let b = uniform::rvs(&nwalkers, &b_loc, &b_scale);
    let log_f = uniform::rvs(&nwalkers, &log_f_loc, &log_f_scale);
    let p0 = ndarray::stack(Axis(1), &[atan_m.view(), b.view(), log_f.view()]);
    match p0 {
        Ok(p0) => Ok(p0),
        Err(_) => Err("Failed to generate p0"),
    }
}

fn log_prior(theta: &Array2<f64>, locs: &Vec<f64>, scales: &Vec<f64>) -> Array1<f64> {
    let (atan_m_loc, b_loc, log_f_loc) = (locs[0], locs[1], locs[2]);
    let (atan_m_scale, b_scale, log_f_scale) = (scales[0], scales[1], scales[2]);
    let atan_m = theta.slice(s![.., 0]).to_owned();
    let b = theta.slice(s![.., 1]).to_owned();
    let log_f = theta.slice(s![.., 2]).to_owned();
    let prob_m = uniform::log_pdf(&atan_m, &atan_m_loc, &atan_m_scale);
    let prob_b = uniform::log_pdf(&b, &b_loc, &b_scale);
    let prob_f = uniform::log_pdf(&log_f, &log_f_loc, &log_f_scale);
    let prob = prob_m + prob_b + prob_f;
    prob
}

fn log_likelihood(
    theta: &Array2<f64>,
    x: Array1<f64>,
    y: Array1<f64>,
    yerr: Array1<f64>,
) -> Array1<f64> {
    let nwalkers = theta.raw_dim()[0];
    let ndata = x.len();
    let m = theta.slice(s![.., 0]).to_owned().mapv_into(|v| v.tan());
    let m = utils::repeat_1d(&m, ndata);
    let b = theta.slice(s![.., 1]).to_owned();
    let b = utils::repeat_1d(&b, ndata);
    let f = theta.slice(s![.., 2]).to_owned().exp();
    let f = utils::repeat_1d(&f, ndata);
    let x = utils::repeat_1d(&x, nwalkers).t().to_owned();
    let y = utils::repeat_1d(&y, nwalkers).t().to_owned();
    let yerr2 = yerr.powi(2);
    let yerr2 = utils::repeat_1d(&yerr2, nwalkers).t().to_owned();
    let y_mod = m * x + b;
    let sigma2 = yerr2 + y_mod.powi(2) * f.powi(2);
    let res = -0.5 * ((y - y_mod).powi(2) / sigma2.clone() + sigma2.ln());
    res.sum_axis(ndarray::Axis(0))
}

pub fn log_prob(
    theta: &Array2<f64>,
    data: &Array2<f64>,
    locs: &Vec<f64>,
    scales: &Vec<f64>,
) -> Array1<f64> {
    let x = data.slice(s![0, ..]).to_owned();
    let y = data.slice(s![1, ..]).to_owned();
    let yerr = data.slice(s![2, ..]).to_owned();
    let prior = log_prior(theta, locs, scales);
    let likelihood = log_likelihood(theta, x, y, yerr);
    prior + likelihood
}
