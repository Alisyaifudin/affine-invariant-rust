use crate::stats::uniform;
use crate::utils;
use ndarray::{s, Array1, Array2};

fn log_prior(theta: &Array2<f64>, locs: &Array1<f64>, scales: &Array1<f64>) -> Array1<f64> {
    let atan_m = theta.slice(s![.., 0]).to_owned();
    let b = theta.slice(s![.., 1]).to_owned();
    let log_f = theta.slice(s![.., 2]).to_owned();

    let prob_m = uniform::log_pdf(&atan_m, Some(locs[0]), Some(scales[0]));
    let prob_b = uniform::log_pdf(&b, Some(locs[1]), Some(scales[1]));
    let prob_f = uniform::log_pdf(&log_f, Some(locs[2]), Some(scales[2]));
    let prob = prob_m + prob_b + prob_f;
    prob
}

fn log_likelihood(
    theta: &Array2<f64>,
    x: Array1<f64>,
    y: Array1<f64>,
    yerr: Array1<f64>,
) -> Array1<f64> {
    let atan_m = theta.slice(s![.., 0]).to_owned();
    let b = theta.slice(s![.., 1]).to_owned();
    let log_f = theta.slice(s![.., 2]).to_owned();
    let yerr2 = yerr.mapv(|v| v.powi(2));

    let N = atan_m.len();
    let M = x.len();
    let m_mat = utils::repeat_1d(&atan_m, M)
        .to_shape((N, M))
        .unwrap()
        .mapv_into(|v| v.tan())
        .to_owned();
    let b_mat = utils::repeat_1d(&b, M).to_shape((N, M)).unwrap().to_owned();
    let log_f_mat = utils::repeat_1d(&log_f, M)
        .to_shape((N, M))
        .unwrap()
        .to_owned();

    let x_mat = utils::repeat_1d(&x, N)
        .to_shape((M, N))
        .unwrap()
        .t()
        .to_owned();
    let y_mat = utils::repeat_1d(&y, N)
        .to_shape((M, N))
        .unwrap()
        .t()
        .to_owned();
    let yerr2_mat = utils::repeat_1d(&yerr2, N)
        .to_shape((M, N))
        .unwrap()
        .t()
        .to_owned();

    let y_mod = m_mat * x_mat + b_mat;
    let sigma2 = yerr2_mat
        + y_mod.clone().mapv_into(|v| v.powi(2)) * log_f_mat.mapv_into(|v| (2. * v).exp());
    let res = -0.5
        * ((y_mat - y_mod).mapv(|v| v.powi(2)) / sigma2.clone() + sigma2.mapv_into(|v| v.ln()));
    res.sum_axis(ndarray::Axis(1))
}

pub fn log_prob(
    theta: &Array2<f64>,
    data: &Array2<f64>,
    locs: &Array1<f64>,
    scales: &Array1<f64>,
) -> Array1<f64> {
    let x = data.slice(s![0, ..]).to_owned();
    let y = data.slice(s![1, ..]).to_owned();
    let yerr = data.slice(s![2, ..]).to_owned();

    let prior = log_prior(theta, locs, scales);
    let likelihood = log_likelihood(theta, x, y, yerr);
    prior + likelihood
}
