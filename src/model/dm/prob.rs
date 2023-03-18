use super::gravity;
use crate::stats::{normal, uniform};
use crate::utils::method::Exponential;
use crate::utils::repeat_scalar;
use ndarray::{s, Array1, Array2, Axis};
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
const SIGMAZ_SCALES: [f64; 12] = [0.2, 0.5, 2.4, 4.0, 1.6, 2.0, 2.4, 1.8, 1.9, 4.0, 5.0, 5.0];

pub const RHOB_INDEX: Range<usize> = 0..12;
pub const SIGMAZ_INDEX: Range<usize> = 12..24;
pub const RHO_DM_INDEX: usize = 24;
pub const LOG_NU0_INDEX: usize = 24 + 1;
pub const R_INDEX: usize = 24 + 2;
pub const ZSUN_INDEX: usize = 24 + 3;
pub const W0_INDEX: usize = 24 + 4;
pub const LOG_SIGMAW1_INDEX: usize = 24 + 5;
pub const LOG_A1_INDEX: usize = 24 + 6;
pub const LOG_SIGMAW2_INDEX: usize = 24 + 7;
pub const LOG_A2_INDEX: usize = 24 + 8;

pub const ZBOUND: f64 = 50.0;

pub fn generate_p0_1(nwalkers: usize, locs: Array1<f64>, scales: Array1<f64>) -> Array2<f64> {
    let rhob = RHOB_LOCS
        .iter()
        .zip(RHOB_SCALES.iter())
        .map(|(loc, scale)| normal::rvs(&nwalkers, loc, scale).to_vec())
        .flatten()
        .collect::<Vec<f64>>();
    let sigmaz = SIGMAZ_LOCS
        .iter()
        .zip(SIGMAZ_SCALES.iter())
        .map(|(loc, scale)| normal::rvs(&nwalkers, loc, scale).to_vec())
        .flatten()
        .collect::<Vec<f64>>();
    let skip = 24;
    let rho_dm = uniform::rvs(
        &nwalkers,
        &locs[RHO_DM_INDEX - skip],
        &scales[RHO_DM_INDEX - skip],
    )
    .to_vec();
    let log_nu0 = uniform::rvs(
        &nwalkers,
        &locs[LOG_NU0_INDEX - skip],
        &scales[LOG_NU0_INDEX - skip],
    )
    .to_vec();
    let r = normal::rvs(&nwalkers, &locs[R_INDEX - skip], &scales[R_INDEX - skip]).to_vec();
    let zsun = uniform::rvs(
        &nwalkers,
        &locs[ZSUN_INDEX - skip],
        &scales[ZSUN_INDEX - skip],
    )
    .to_vec();
    let w0 = uniform::rvs(&nwalkers, &locs[W0_INDEX - skip], &scales[W0_INDEX - skip]).to_vec();
    let log_sigmaw = uniform::rvs(
        &nwalkers,
        &locs[LOG_SIGMAW1_INDEX - skip],
        &scales[LOG_SIGMAW1_INDEX - skip],
    )
    .to_vec();
    let log_a = uniform::rvs(
        &nwalkers,
        &locs[LOG_A1_INDEX - skip],
        &scales[LOG_A1_INDEX - skip],
    )
    .to_vec();
    // concatenate all the Vecs into a single Vec
    let p0 = rhob
        .into_iter()
        .chain(sigmaz.into_iter())
        .chain(rho_dm.into_iter())
        .chain(log_nu0.into_iter())
        .chain(r.into_iter())
        .chain(zsun.into_iter())
        .chain(w0.into_iter())
        .chain(log_sigmaw.into_iter())
        .chain(log_a.into_iter())
        .collect::<Vec<f64>>();
    let res = Array2::from_shape_vec((31, nwalkers), p0)
        .unwrap()
        .t()
        .to_owned();
    res
}

pub fn log_prior1(theta: &Array2<f64>, locs: &Array1<f64>, scales: &Array1<f64>) -> Array1<f64> {
    let rhob: Array2<f64> = theta.slice(s![.., RHOB_INDEX]).to_owned();
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
    let skip = 24;
    let rho_dm = theta.slice(s![.., RHO_DM_INDEX]).to_owned();
    let rho_dm = uniform::log_pdf(
        &rho_dm,
        &locs[RHO_DM_INDEX - skip],
        &scales[RHO_DM_INDEX - skip],
    );
    let log_nu0 = theta.slice(s![.., LOG_NU0_INDEX]).to_owned();
    let log_nu0 = uniform::log_pdf(
        &log_nu0,
        &locs[LOG_NU0_INDEX - skip],
        &scales[LOG_NU0_INDEX - skip],
    );
    let r = theta.slice(s![.., R_INDEX]).to_owned();
    let r = normal::log_pdf(&r, &locs[R_INDEX - skip], &scales[R_INDEX - skip]);
    let zsun = theta.slice(s![.., ZSUN_INDEX]).to_owned();
    let zsun = uniform::log_pdf(&zsun, &locs[ZSUN_INDEX - skip], &scales[ZSUN_INDEX - skip]);
    let w0 = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0 = uniform::log_pdf(&w0, &locs[W0_INDEX - skip], &scales[W0_INDEX - skip]);
    let log_sigmaw = theta.slice(s![.., LOG_SIGMAW1_INDEX]).to_owned();
    let log_sigmaw = uniform::log_pdf(
        &log_sigmaw,
        &locs[LOG_SIGMAW1_INDEX - skip],
        &scales[LOG_SIGMAW1_INDEX - skip],
    );
    let log_a = theta.slice(s![.., LOG_A1_INDEX]).to_owned();
    let log_a = uniform::log_pdf(
        &log_a,
        &locs[LOG_A1_INDEX - skip],
        &scales[LOG_A1_INDEX - skip],
    );
    let res = rhob.sum_axis(Axis(1))
        + sigmaz.sum_axis(Axis(1))
        + rho_dm
        + log_nu0
        + r
        + zsun
        + log_sigmaw
        + w0
        + log_a;
    res
}

fn log_likelihood1(
    theta: &Array2<f64>,
    zdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    dz: Option<f64>,
) -> Array1<f64> {
    let (zmid, znum, zerr) = zdata;
    let (wmid, wnum, werr) = wdata;
    let zmod = gravity::fz1(zmid.to_owned(), theta.clone(), dz);
    let wmod = gravity::fw1(wmid.to_owned(), theta.clone(), dz);
    let probz = normal::log_pdf_array2(&(zmod - znum), &repeat_scalar(0., zerr.len()), &zerr)
        .sum_axis(Axis(1));
    let probw = normal::log_pdf_array2(&(wmod - wnum), &repeat_scalar(0., werr.len()), &werr)
        .sum_axis(Axis(1));
    let res = probz + probw;
    res
}

pub fn log_prob1(
    theta: &Array2<f64>,
    zdata: (Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: (Array1<f64>, Array1<f64>, Array1<f64>),
    locs: &Array1<f64>,
    scales: &Array1<f64>,
    dz: Option<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let prior = log_prior1(theta, locs, scales);
    let likelihood = log_likelihood1(theta, &zdata, &wdata, dz.clone());
    (prior.clone(), prior + likelihood)
}

// ###################################################################################################
// ###################################################################################################
// ###################################################################################################
// two component model
// ###################################################################################################
// ###################################################################################################
// ###################################################################################################

pub fn generate_p0_2(nwalkers: usize, locs: Array1<f64>, scales: Array1<f64>) -> Array2<f64> {
    let rhob = RHOB_LOCS
        .iter()
        .zip(RHOB_SCALES.iter())
        .map(|(loc, scale)| normal::rvs(&nwalkers, loc, scale).to_vec())
        .flatten()
        .collect::<Vec<f64>>();
    let sigmaz = SIGMAZ_LOCS
        .iter()
        .zip(SIGMAZ_SCALES.iter())
        .map(|(loc, scale)| normal::rvs(&nwalkers, loc, scale).to_vec())
        .flatten()
        .collect::<Vec<f64>>();
    let skip = 24;
    let rho_dm = uniform::rvs(
        &nwalkers,
        &locs[RHO_DM_INDEX - skip],
        &scales[RHO_DM_INDEX - skip],
    )
    .to_vec();
    let log_nu0 = uniform::rvs(
        &nwalkers,
        &locs[LOG_NU0_INDEX - skip],
        &scales[LOG_NU0_INDEX - skip],
    )
    .to_vec();
    let r = normal::rvs(&nwalkers, &locs[R_INDEX - skip], &scales[R_INDEX - skip]).to_vec();
    let zsun = uniform::rvs(
        &nwalkers,
        &locs[ZSUN_INDEX - skip],
        &scales[ZSUN_INDEX - skip],
    )
    .to_vec();
    let w0 = uniform::rvs(&nwalkers, &locs[W0_INDEX - skip], &scales[W0_INDEX - skip]).to_vec();
    let log_sigmaw1 = uniform::rvs(
        &nwalkers,
        &locs[LOG_SIGMAW1_INDEX - skip],
        &scales[LOG_SIGMAW1_INDEX - skip],
    )
    .to_vec();
    let log_a1 = uniform::rvs(
        &nwalkers,
        &locs[LOG_A1_INDEX - skip],
        &scales[LOG_A1_INDEX - skip],
    )
    .to_vec();
    let log_sigmaw2 = uniform::rvs(
        &nwalkers,
        &locs[LOG_SIGMAW2_INDEX - skip],
        &scales[LOG_SIGMAW2_INDEX - skip],
    )
    .to_vec();
    let log_a2 = uniform::rvs(
        &nwalkers,
        &locs[LOG_A2_INDEX - skip],
        &scales[LOG_A2_INDEX - skip],
    )
    .to_vec();
    // concatenate all the Vecs into a single Vec
    let p0 = rhob
        .into_iter()
        .chain(sigmaz.into_iter())
        .chain(rho_dm.into_iter())
        .chain(log_nu0.into_iter())
        .chain(r.into_iter())
        .chain(zsun.into_iter())
        .chain(w0.into_iter())
        .chain(log_sigmaw1.into_iter())
        .chain(log_a1.into_iter())
        .chain(log_sigmaw2.into_iter())
        .chain(log_a2.into_iter())
        .collect::<Vec<f64>>();
    let res = Array2::from_shape_vec((33, nwalkers), p0)
        .unwrap()
        .t()
        .to_owned();
    res
}

pub fn log_prior2(theta: &Array2<f64>, locs: &Array1<f64>, scales: &Array1<f64>) -> Array1<f64> {
    let rhob: Array2<f64> = theta.slice(s![.., RHOB_INDEX]).to_owned();
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
    let skip = 24;
    let rho_dm = theta.slice(s![.., RHO_DM_INDEX]).to_owned();
    let rho_dm = uniform::log_pdf(
        &rho_dm,
        &locs[RHO_DM_INDEX - skip],
        &scales[RHO_DM_INDEX - skip],
    );
    let log_nu0 = theta.slice(s![.., LOG_NU0_INDEX]).to_owned();
    let log_nu0 = uniform::log_pdf(
        &log_nu0,
        &locs[LOG_NU0_INDEX - skip],
        &scales[LOG_NU0_INDEX - skip],
    );
    let r = theta.slice(s![.., R_INDEX]).to_owned();
    let r = normal::log_pdf(&r, &locs[R_INDEX - skip], &scales[R_INDEX - skip]);
    let zsun = theta.slice(s![.., ZSUN_INDEX]).to_owned();
    let zsun = uniform::log_pdf(&zsun, &locs[ZSUN_INDEX - skip], &scales[ZSUN_INDEX - skip]);
    let w0 = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0 = uniform::log_pdf(&w0, &locs[W0_INDEX - skip], &scales[W0_INDEX - skip]);
    let log_sigmaw1 = theta.slice(s![.., LOG_SIGMAW1_INDEX]).to_owned();
    let log_sigmaw1 = uniform::log_pdf(
        &log_sigmaw1,
        &locs[LOG_SIGMAW1_INDEX - skip],
        &scales[LOG_SIGMAW1_INDEX - skip],
    );
    let log_a1 = theta.slice(s![.., LOG_A1_INDEX]).to_owned();
    let log_a1 = uniform::log_pdf(
        &log_a1,
        &locs[LOG_A1_INDEX - skip],
        &scales[LOG_A1_INDEX - skip],
    );
    let log_sigmaw2 = theta.slice(s![.., LOG_SIGMAW2_INDEX]).to_owned();
    let log_sigmaw2 = uniform::log_pdf(
        &log_sigmaw2,
        &locs[LOG_SIGMAW2_INDEX - skip],
        &scales[LOG_SIGMAW2_INDEX - skip],
    );
    let log_a2 = theta.slice(s![.., LOG_A2_INDEX]).to_owned();
    let log_a2 = uniform::log_pdf(
        &log_a2,
        &locs[LOG_A2_INDEX - skip],
        &scales[LOG_A2_INDEX - skip],
    );
    let res = rhob.sum_axis(Axis(1))
        + sigmaz.sum_axis(Axis(1))
        + rho_dm
        + log_nu0
        + r
        + zsun
        + w0
        + log_sigmaw1
        + log_a1
        + log_sigmaw2
        + log_a2;
    res
}

fn log_likelihood2(
    theta: &Array2<f64>,
    zdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: &(Array1<f64>, Array1<f64>, Array1<f64>),
    dz: Option<f64>,
) -> Array1<f64> {
    let (zmid, znum, zerr) = zdata;
    let (wmid, wnum, werr) = wdata;
    let zmod = gravity::fz2(zmid.to_owned(), theta.clone(), dz);
    let wmod = gravity::fw2(wmid.to_owned(), theta.clone(), dz);
    let probz = normal::log_pdf_array2(&(zmod - znum), &repeat_scalar(0., zerr.len()), &zerr)
        .sum_axis(Axis(1));
    let probw = normal::log_pdf_array2(&(wmod - wnum), &repeat_scalar(0., werr.len()), &werr)
        .sum_axis(Axis(1));
    let res = probz + probw;
    res
}

pub fn log_prob2(
    theta: &Array2<f64>,
    zdata: (Array1<f64>, Array1<f64>, Array1<f64>),
    wdata: (Array1<f64>, Array1<f64>, Array1<f64>),
    locs: &Array1<f64>,
    scales: &Array1<f64>,
    dz: Option<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let prior = log_prior2(theta, locs, scales);
    let likelihood = log_likelihood2(theta, &zdata, &wdata, dz.clone());
    // nan?
    (prior.clone(), prior + likelihood)
}

//###############################################################################################
//###############################################################################################
//###############################################################################################
//###############################################################################################

pub fn sample(
    pos: &Array2<f64>,
    theta: &Array1<f64>,
    dz: Option<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let log_fzw = gravity::fzw(pos, &theta, dz).unwrap().ln();
    let len = log_fzw.len();
    let log_prior = Array1::<f64>::ones(len);
    (log_prior, log_fzw)
}
