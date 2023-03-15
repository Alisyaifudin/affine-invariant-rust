use crate::gravity::dm;
use crate::stats::{normal, uniform};
use crate::utils::repeat_scalar;
use ndarray::{arr1, s, Array1, Array2, Axis};
use std::ops::Range;

const RHOB_LOCS: [f64; 12] = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018, 0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015,
];
const RHOB_SCALES: [f64; 12] = [
    0.00312, 0.00554, 0.00070, 0.00003, 0.00006, 0.00018, 0.00018, 0.00029, 0.00072, 0.00280,
    0.00100, 0.00050,
];

const SIGMAZ_LOCS: [f64; 12] = [
    3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0, 18.0, 18.5, 18.5, 20.0, 20.0,
];

pub const RHOB_INDEX: Range<usize> = 0..12;
pub const SIGMAZ_INDEX: Range<usize> = 12..24;
pub const RHO_DM_INDEX: usize = 24;
pub const LOG_NU0_INDEX: usize = 24 + 1;
pub const R_INDEX: usize = 24 + 2;
pub const ZSUN_INDEX: usize = 24 + 3;
pub const W0_INDEX: usize = 24 + 4;
pub const SIGMAW1_INDEX: usize = 24 + 5;
pub const LOG_A1_INDEX: usize = 24 + 6;
pub const SIGMAW2_INDEX: usize = 24 + 7;
pub const LOG_A2_INDEX: usize = 24 + 8;

const SIGMAZ_SCALES: [f64; 12] = [0.2, 0.5, 2.4, 4.0, 1.6, 2.0, 2.4, 1.8, 1.9, 4.0, 5.0, 5.0];

fn log_prior1(theta: &Array2<f64>) -> Array1<f64> {
    let rhob: Array2<f64> = theta.slice(s![.., RHOB_INDEX]).t().to_owned();
    let rhob = normal::log_pdf_array2(
        &rhob,
        &Array1::from_vec(RHOB_LOCS.to_vec()),
        &Array1::from_vec(RHOB_SCALES.to_vec()),
    );
    let sigmaz = theta.slice(s![.., SIGMAZ_INDEX]).to_owned();
    let sigmaz = normal::log_pdf_array2(
        &sigmaz,
        &Array1::from_vec(SIGMAZ_LOCS.to_vec()),
        &Array1::from_vec(SIGMAZ_SCALES.to_vec()),
    );
    let rhoDM = theta.slice(s![.., RHO_DM_INDEX]).to_owned();
    let rhoDM = uniform::log_pdf(&rhoDM, &-0.02, &0.12);
    let log_nu0 = theta.slice(s![.., LOG_NU0_INDEX]).to_owned();
    let log_nu0 = uniform::log_pdf(&log_nu0, &-15., &10.);
    let R = theta.slice(s![.., R_INDEX]).to_owned();
    let R = normal::log_pdf(&rhoDM, &3.4E-3, &0.6E-3);
    let zsun = theta.slice(s![.., ZSUN_INDEX]).to_owned();
    let zsun = uniform::log_pdf(&zsun, &-50., &100.);
    let w0 = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0 = uniform::log_pdf(&w0, &-15., &15.);
    let sigmaw = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let sigmaw = uniform::log_pdf(&sigmaw, &1., &19.);
    let log_a = theta.slice(s![.., LOG_A1_INDEX]).to_owned();
    let log_a = uniform::log_pdf(&log_a, &-5., &10.);
    let res = rhob.sum_axis(Axis(0))
        + sigmaz.sum_axis(Axis(0))
        + rhoDM
        + log_nu0
        + R
        + zsun
        + sigmaw
        + w0
        + log_a;
    res
}

fn log_likelihood1(
    theta: &Array2<f64>,
    zdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: &(Array1<f64>, Array1<f64>, Array1<f64>, f64),
    dz: Option<f64>
) -> Array1<f64> {
    let sh = theta.raw_dim();
    let nwalkers = sh[0];
    let (zmid, znum, zerr) = zdata;
    let (wmid, wnum, werr, zbound) = wdata;
    let zmod = dm::fz1(zmid.to_owned(), theta.clone(), dz);
    let wmod = dm::fw1(wmid.to_owned(), *zbound, theta.clone(), dz);
    let probz = normal::log_pdf_array2(&(zmod - znum), &repeat_scalar(0., zerr.len()), &zerr).sum_axis(Axis(1));
    let probw = normal::log_pdf_array2(&(wmod - wnum), &repeat_scalar(0., werr.len()), &werr).sum_axis(Axis(1));
    let res = probz + probw;
    res
}

pub fn log_prob1(
    theta: &Array2<f64>,
    zdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: &(Array1<f64>, Array1<f64>, Array1<f64>, f64),
    dz: &Option<f64>
) -> Array1<f64> {
    let prior = log_prior1(theta);
    let likelihood = log_likelihood1(theta, zdata, wdata, *dz);
    prior + likelihood
}

// fn log_likelihood1(
//     theta: &Array2<f64>,
//     zdata: &(Array1<f64>, Array1<f64>, Array1<f64>, f64),
//     wdata: &(Array1<f64>, Array1<f64>, Array1<f64>, f64, f64),
// ) -> Array1<f64> {
//     let sh = theta.raw_dim();
//     let nwalkers = sh[0];
//     let (zmid, znum, zerr, dz) = zdata;
//     let (wmid, wnum, werr, dw, zbound) = wdata;
//     let Nztot = znum.sum();
//     let Nwtot = wnum.sum();
//     let res = (0..nwalkers)
//         .map(|i| {
//             let Nz = dm::Nz1(zmid.clone(), *dz, Nztot, theta.slice(s![i, ..]).to_owned());
//             let Nw = dm::Nw1(
//                 wmid.clone(),
//                 *dw,
//                 Nwtot,
//                 *zbound,
//                 theta.slice(s![i, ..]).to_owned(),
//             );
//             let resz = Nz
//                 .iter()
//                 .zip(znum.iter())
//                 .zip(zerr.iter())
//                 .map(|((m, num), err)| {
//                     normal::log_pdf(&arr1(&vec![*m]), Some(*num), Some(*err)).to_vec()[0]
//                 })
//                 .sum::<f64>();
//             let resw = Nw
//                 .iter()
//                 .zip(wnum.iter())
//                 .zip(werr.iter())
//                 .map(|((m, num), err)| {
//                     normal::log_pdf(&arr1(&vec![*m]), Some(*num), Some(*err)).to_vec()[0]
//                 })
//                 .sum::<f64>();
//             resz + resw
//         })
//         .collect::<Vec<f64>>();
//     arr1(&res)
// }