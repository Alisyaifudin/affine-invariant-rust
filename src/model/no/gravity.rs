use crate::stats::normal;
use crate::utils;
use crate::utils::method::{Exponential, MinMax, Power, Sqrt, ToArray2};
use interp1d::Interp1d;
use ndarray::{s, Array1, Array2, Axis};
use ode_solvers::dopri5::*;
use ode_solvers::*;
use std::f64::consts::PI;

// use super::prob;
use super::prob::{
    LOG_A1_INDEX, LOG_A2_INDEX, LOG_NU0_INDEX, RHOB_INDEX, R_INDEX, SIGMAW1_INDEX,
    SIGMAW2_INDEX, SIGMAZ_INDEX, W0_INDEX, ZBOUND, ZSUN_INDEX,
};

pub type State = Vector2<f64>;
type Height = f64;

const G: f64 = 4.30091E-3; // pc/M_sun (km/s)^2

struct Gravity {
    rhob: SVector<f64, 12>,
    sigmaz: SVector<f64, 12>,
    rho_dm: f64,
    sigma_dd: f64,
    h_dd: f64,
    r: f64,
}

impl ode_solvers::System<State> for Gravity {
    // Equations of motion of the system
    fn system(&self, z: Height, y: &State, dy: &mut State) {
        let u = (y[0], y[1]);
        (dy[0], dy[1]) = dfz(
            &u,
            &z,
            &self.rhob,
            &self.sigmaz,
            &self.rho_dm,
            &self.sigma_dd,
            &self.h_dd,
            &self.r,
        );
    }
}

fn sech(x: &f64) -> f64 {
    let y = 1.0 / x.cosh();
    y
}

fn rho_dd_func(z: &f64, sigma_dd: &f64, h_dd: &f64) -> f64 {
    let y = sigma_dd / (4. * h_dd) * sech(&(z / (2. * h_dd))).powi(2);
    y
}

fn frho(phi: &f64, rhob: &f64, sigmaz: &f64) -> f64 {
    let y = rhob * (-phi / sigmaz.powi(2)).exp();
    y
}

fn rho_tot(
    z: &f64,
    phi: &f64,
    rhob: &SVector<f64, 12>,
    sigmaz: &SVector<f64, 12>,
    rho_dd: &f64,
    sigma_dd: &f64,
    h_dd: &f64,
    r: &f64,
) -> f64 {
    let rho: f64 = rhob
        .iter()
        .zip(sigmaz.iter())
        .map(|(rhob, sigmaz)| frho(phi, rhob, sigmaz))
        .sum::<f64>();
    let rhodd = rho_dd_func(z, sigma_dd, h_dd);
    let y = rho + rhodd + rho_dd - r;
    y
}

pub fn dfz(
    u: &(f64, f64),
    z: &f64,
    rhob: &SVector<f64, 12>,
    sigmaz: &SVector<f64, 12>,
    rho_dm: &f64,
    sigma_dd: &f64,
    h_dd: &f64,
    r: &f64,
) -> (f64, f64) {
    (
        u.1.clone(),
        4. * PI * G * rho_tot(z, &u.0, rhob, sigmaz, rho_dm, sigma_dd, h_dd, r),
    )
}

#[allow(dead_code)]
pub fn solve(
    theta: Array2<f64>,
    z_start: f64,
    z_end: f64,
    dz: f64,
) -> Vec<Option<(Vec<f64>, Vec<State>)>> {
    let rhob = theta.slice(s![.., RHOB_INDEX]).to_owned();
    let sigmaz = theta.slice(s![.., SIGMAZ_INDEX]).to_owned();
    let rho_dm = 0.;
    let sigma_dd = 0.;
    let h_dd = 1.;
    let r = theta.slice(s![.., R_INDEX]).to_owned();

    let res = rhob
        .axis_iter(Axis(0))
        .zip(sigmaz.axis_iter(Axis(0)))
        .zip(r.iter())
        .map(|((rhob, sigmaz), r)| {
            let system = Gravity {
                rhob: SVector::<f64, 12>::from_vec(rhob.to_vec()),
                sigmaz: SVector::<f64, 12>::from_vec(sigmaz.to_vec()),
                rho_dm,
                sigma_dd,
                h_dd,
                r: *r,
            };

            let y0: State = State::new(0., 0.);

            let mut stepper = Dopri5::new(system, z_start, z_end, dz, y0, 1.0e-10, 1.0e-10);
            let res = stepper.integrate();
            let res: Option<(Vec<f64>, Vec<State>)> = match res {
                Ok(_stats) => Some((stepper.x_out().to_owned(), stepper.y_out().to_owned())),
                Err(_) => None,
            };
            res
        })
        .collect::<Vec<Option<(Vec<f64>, Vec<State>)>>>();
    res
}

pub fn potential(z: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let dz = dz.unwrap_or(10.);
    let nwalkers = theta.raw_dim()[0];
    let z_len = z.len();
    let z = utils::repeat_1d(&z, nwalkers).t().to_owned();
    let z_sun = theta.slice(s![.., ZSUN_INDEX]).to_owned();
    let z_sun = utils::repeat_1d(&z_sun, z_len).to_owned();
    let z_rel = (z + z_sun).mapv_into(|z| z.abs());
    let z_max = z_rel.max();
    let res = solve(theta, 0., z_max, dz);
    let res = res
        .iter()
        .zip(z_rel.axis_iter(Axis(1)))
        .map(|(res, z_rel)| match res {
            Some((z, u)) => {
                let phi = u.iter().map(|u| u[0]).collect::<Vec<f64>>();
                let interpolator = Interp1d::new_sorted(z.clone(), phi).unwrap();
                let phi_rel = z_rel.to_owned().mapv_into(|z| interpolator.interpolate(z));
                phi_rel
            }
            None => z_rel.to_owned() * f64::NAN,
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

pub fn fz1(z: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let pot = potential(z, theta.clone(), dz);
    let sigmaw = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let nu0 = theta
        .slice(s![.., LOG_NU0_INDEX])
        .to_owned()
        .mapv_into(|v| v.exp());
    let res = pot
        .axis_iter(Axis(0))
        .zip(sigmaw.iter())
        .zip(nu0.iter())
        .map(|((pot, sigmaw), nu0)| {
            pot.to_owned()
                .mapv_into(|p| nu0 * (-p / sigmaw.powi(2)).exp())
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

pub fn fw1(w: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let nwalkers = theta.raw_dim()[0];
    let w_len = w.len();
    let w: Array2<f64> = utils::repeat_1d(&w, nwalkers).t().to_owned();
    let w0: Array1<f64> = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0: Array2<f64> = utils::repeat_1d(&w0, w_len).to_owned();
    let sigmaw: Array1<f64> = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let a: Array1<f64> = theta
        .slice(s![.., LOG_A1_INDEX])
        .to_owned()
        .mapv_into(|v| v.exp());
    let w_rel: Array2<f64> = w - w0;
    let pot_bound: Array2<f64> = potential(Array1::from_vec(vec![ZBOUND]), theta, dz);
    let res = w_rel
        .axis_iter(Axis(1))
        .zip(sigmaw.iter())
        .zip(a.iter())
        .zip(pot_bound.axis_iter(Axis(1)))
        .map(|(((w_rel, sigmaw), a), pot_bound)| {
            let w_bound = w_rel.to_owned().mapv_into(|w_rel| {
                let sign = w_rel.signum();
                let pot_b = pot_bound[0];
                sign * (w_rel.powi(2) + 2. * pot_b).sqrt()
            });
            let res =
                normal::pdf(w_rel.to_owned(), 0., *sigmaw) + normal::pdf(w_bound, 0., *sigmaw);
            res.mapv_into(|v| v * a)
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

pub fn fz2(z: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let pot = potential(z, theta.clone(), dz);
    let sigmaw1 = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let sigmaw2 = theta.slice(s![.., SIGMAW2_INDEX]).to_owned();
    let a1 = theta.slice(s![.., LOG_A1_INDEX]).to_owned().exp();
    let a2 = theta.slice(s![.., LOG_A2_INDEX]).to_owned().exp();
    let nu0 = theta
        .slice(s![.., LOG_NU0_INDEX])
        .to_owned()
        .mapv_into(|v| v.exp());
    let res = pot
        .axis_iter(Axis(0))
        .zip(sigmaw1.iter())
        .zip(sigmaw2.iter())
        .zip(a1.iter())
        .zip(a2.iter())
        .zip(nu0.iter())
        .map(|(((((pot, sigmaw1), sigmaw2), a1), a2), nu0)| {
            let atot = a1 + a2;
            pot.to_owned().mapv_into(|p| {
                nu0 * ((a1 / atot.clone()) * (-p / sigmaw1.powi(2)).exp()
                    + (a2 / atot) * (-p / sigmaw2.powi(2)).exp())
            })
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

pub fn fw2(w: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let nwalkers = theta.raw_dim()[0];
    let w_len = w.len();
    let w: Array2<f64> = utils::repeat_1d(&w, nwalkers).t().to_owned();
    let w0: Array1<f64> = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0: Array2<f64> = utils::repeat_1d(&w0, w_len).to_owned();
    let sigmaw1: Array1<f64> = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let sigmaw2: Array1<f64> = theta.slice(s![.., SIGMAW2_INDEX]).to_owned();
    let a1: Array1<f64> = theta.slice(s![.., LOG_A1_INDEX]).to_owned().exp();
    let a2: Array1<f64> = theta.slice(s![.., LOG_A2_INDEX]).to_owned().exp();
    let w_rel: Array2<f64> = w - w0;
    let pot_bound: Array2<f64> = potential(Array1::from_vec(vec![ZBOUND]), theta, dz);
    let res = w_rel
        .axis_iter(Axis(1))
        .zip(sigmaw1.iter())
        .zip(sigmaw2.iter())
        .zip(a1.iter())
        .zip(a2.iter())
        .zip(pot_bound.axis_iter(Axis(1)))
        .map(|(((((w_rel, sigmaw1), sigmaw2), a1), a2), pot_bound)| {
            let w_bound = w_rel.to_owned().mapv_into(|w_rel| {
                let sign = w_rel.signum();
                let pot_b = pot_bound[0];
                sign * (w_rel.powi(2) + 2. * pot_b).sqrt()
            });
            let res1 = normal::pdf(w_rel.to_owned(), 0., *sigmaw1)
                + normal::pdf(w_bound.clone(), 0., *sigmaw1);
            let res1 = res1.mapv_into(|v| v * a1);
            let res2 =
                normal::pdf(w_rel.to_owned(), 0., *sigmaw2) + normal::pdf(w_bound, 0., *sigmaw2);
            let res2 = res2.mapv_into(|v| v * a2);
            res1 + res2
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

pub fn fzw(
    pos: &Array2<f64>,
    theta: &Array1<f64>,
    dz: Option<f64>,
) -> Result<Array1<f64>, &'static str> {
    let z: Array1<f64> = pos.slice(s![.., 0]).to_owned();
    let w: Array1<f64> = pos.slice(s![.., 1]).to_owned();

    let ndim = theta.len();
    let w0 = theta[W0_INDEX];
    let w_rel = w - w0;
    let sigmaw1 = theta[SIGMAW1_INDEX];
    let a1 = theta[LOG_A1_INDEX].exp();
    let theta_thick = theta.clone().insert_axis(Axis(0));
    let potential = potential(z, theta_thick.clone(), dz).row(0).to_owned();
    if ndim == 31 {
        let sign = w_rel.clone().mapv_into(|w| w.signum());
        let w_val = sign * (w_rel.powi(2) + 2. * potential).sqrt();
        let p = a1 * normal::pdf(w_val, 0., sigmaw1);
        Ok(p)
    } else if ndim == 33 {
        let sigmaw2 = theta[SIGMAW2_INDEX];
        let a2 = theta[LOG_A2_INDEX].exp();

        let sign = w_rel.clone().mapv_into(|w| w.signum());
        let w_val = sign * (w_rel.powi(2) + 2. * potential).sqrt();
        let p = a1 * normal::pdf(w_val.clone(), 0., sigmaw1) + a2 * normal::pdf(w_val, 0., sigmaw2);
        Ok(p)
    } else {
        Err("Wrong number of parameters")
    }
}
