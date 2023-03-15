use crate::gravity::prob::dm::{
    LOG_A1_INDEX, LOG_NU0_INDEX, RHOB_INDEX, RHO_DM_INDEX, R_INDEX, SIGMAW1_INDEX, SIGMAZ_INDEX,
    W0_INDEX, ZSUN_INDEX,
};
use crate::stats::normal;
use crate::utils::{repeat_1d, MinMax, ToArray2};
use interp1d::Interp1d;
use ndarray::{arr1, s, Array1, Array2, Axis};
use ode_solvers::dopri5::*;
use ode_solvers::*;
use std::f64::consts::PI;
use std::sync::Arc;

pub type State = Vector2<f64>;
type Height = f64;

const G: f64 = 4.30091E-3; // pc/M_sun (km/s)^2

struct Gravity {
    rhob: SVector<f64, 12>,
    sigmaz: SVector<f64, 12>,
    rhoDM: f64,
    sigmaDD: f64,
    hDD: f64,
    R: f64,
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
            &self.rhoDM,
            &self.sigmaDD,
            &self.hDD,
            &self.R,
        );
    }
}

fn sech(x: &f64) -> f64 {
    let y = 1.0 / x.cosh();
    y
}

fn rhoDD(z: &f64, sigmaDD: &f64, hDD: &f64) -> f64 {
    let y = sigmaDD / (4. * hDD) * sech(&(z / (2. * hDD))).powi(2);
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
    rhoDM: &f64,
    sigmaDD: &f64,
    hDD: &f64,
    R: &f64,
) -> f64 {
    let rho: f64 = rhob
        .iter()
        .zip(sigmaz.iter())
        .map(|(rhob, sigmaz)| frho(phi, rhob, sigmaz))
        .sum::<f64>();
    let rhodd = rhoDD(z, sigmaDD, hDD);
    let y = rho + rhodd + rhoDM - R;
    y
}

pub fn dfz(
    u: &(f64, f64),
    z: &f64,
    rhob: &SVector<f64, 12>,
    sigmaz: &SVector<f64, 12>,
    rhoDM: &f64,
    sigmaDD: &f64,
    hDD: &f64,
    R: &f64,
) -> (f64, f64) {
    (
        u.1.clone(),
        4. * PI * G * rho_tot(z, &u.0, rhob, sigmaz, rhoDM, sigmaDD, hDD, R),
    )
}

#[allow(dead_code)]
pub fn solve(
    theta: Array2<f64>,
    z_start: f64,
    z_end: f64,
    dz: f64,
) -> Vec<Option<(Vec<f64>, Vec<State>)>> {
    let rhob: Array2<f64> = theta.slice(s![.., RHOB_INDEX]).to_owned();
    let sigmaz: Array2<f64> = theta.slice(s![.., SIGMAZ_INDEX]).to_owned();
    let rhoDM: Array1<f64> = theta.slice(s![.., RHO_DM_INDEX]).to_owned();
    let sigmaDD = 0.;
    let hDD = 1.;
    let R: Array1<f64> = theta.slice(s![.., R_INDEX]).to_owned();
    let res = rhob
        .axis_iter(Axis(1))
        .zip(sigmaz.axis_iter(Axis(1)))
        .zip(rhoDM.iter())
        .zip(R.iter())
        .map(|(((rhob, sigmaz), rhoDM), R)| {
            let rhob = rhob.to_owned();
            let sigmaz = sigmaz.to_owned();
            let rhoDM = rhoDM.to_owned();
            let R = R.to_owned();
            let system = Gravity {
                rhob: SVector::<f64, 12>::from_vec(rhob.to_vec()),
                sigmaz: SVector::<f64, 12>::from_vec(sigmaz.to_vec()),
                rhoDM,
                sigmaDD,
                hDD,
                R,
            };
            let y0 = State::new(0., 0.);
            let mut stepper = Dopri5::new(system, z_start, z_end, dz, y0, 1.0e-10, 1.0e-10);
            let res = stepper.integrate();

            // Handle result
            let val: Option<(Vec<f64>, Vec<State>)> = match res {
                Ok(stats) => Some((stepper.x_out().to_vec(), stepper.y_out().to_vec())),
                Err(_) => {
                    println!("rhoDM {}", rhoDM);
                    None
                }
            };
            val
        });
    let res = res.collect::<Vec<Option<(Vec<f64>, Vec<State>)>>>();
    res
}

pub fn potential(z: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Vec<Array1<f64>> {
    let dz = dz.unwrap_or(10.);
    let z_len = z.len();
    let z = repeat_1d(&z, theta.raw_dim()[0]).t().to_owned();
    let z_sun = theta.slice(s![.., ZSUN_INDEX]).to_owned();
    let z_sun = repeat_1d(&z_sun, z_len);
    let z_rel = (z + z_sun).mapv_into(|z| z.abs());
    let z_max = z_rel.max();
    let res = solve(theta, 0., z_max, dz);
    let phi_rel = res
        .iter()
        .zip(z_rel.axis_iter(Axis(0)))
        .map(|(res, z_rel)| match res {
            Some((z, u)) => {
                let phi = u.iter().map(|u| u[0]).collect::<Vec<f64>>();
                let interpolator = Interp1d::new_sorted(z.to_vec(), phi).unwrap();
                let phi_rel = z_rel.to_owned().mapv_into(|z| interpolator.interpolate(z));
                phi_rel
            }
            None => Array1::zeros(z_len),
        })
        .collect::<Vec<Array1<f64>>>();
    phi_rel
}

pub fn fz1(z: Array1<f64>, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let pot = potential(z.clone(), theta.clone(), dz);
    let nu0 = theta
        .slice(s![.., LOG_NU0_INDEX])
        .to_owned()
        .mapv_into(|v| v.exp());
    let sigmaw = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let fz = pot
        .iter()
        .zip(sigmaw.iter())
        .zip(nu0.iter())
        .map(|((pot, sigmaw), nu0)| pot.to_owned().mapv_into(|v| nu0 * (-v / sigmaw.powi(2)).exp()))
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    fz
}

// pub fn Nz1(z: Array1<f64>, dz: f64, Nz: f64, theta: Array2<f64>) -> Array2<f64> {
//     let fz_un = fz1(z.clone(), theta.clone());
//     let area = fz_un.trapz(&z);
//     let fz_un = fz_un / area;
//     let N = fz_un * dz * Nz;
//     N
// }

pub fn fw1(w: Array1<f64>, zbound: f64, theta: Array2<f64>, dz: Option<f64>) -> Array2<f64> {
    let sigmaw = theta.slice(s![.., SIGMAW1_INDEX]).to_owned();
    let w0 = theta.slice(s![.., W0_INDEX]).to_owned();
    let w0 = repeat_1d(&w0, w.len());
    let a = Arc::new(theta.slice(s![.., LOG_A1_INDEX]).to_owned());
    let potential = potential(arr1(&vec![zbound]), theta, dz);
    let res = potential
        .iter()
        .zip(sigmaw.iter())
        .zip(w0.axis_iter(Axis(1)))
        .zip(a.iter())
        .map(|(((pot, sigmaw), w0), a)| {
            let w_plane = w.to_owned() - w0;
            let w_bound = w_plane.clone().mapv_into(|w| {
                let sign = w.signum();
                a * sign * (w.powi(2) + 2. * pot[0]).sqrt()
            });
            let res = normal::pdf(&w_plane, &0., sigmaw) + normal::pdf(&w_bound, &0., sigmaw);
            res
        })
        .collect::<Vec<Array1<f64>>>()
        .to_array2();
    res
}

// pub fn Nw1(w: Array1<f64>, dw: f64, Nw: f64, zbound: f64, theta: Array1<f64>) -> Array1<f64> {
//     let fw_un = fw1(w.clone(), zbound, theta.clone());
//     let area = fw_un.trapz(&w);
//     let fw_un = fw_un / area;
//     let N = fw_un * dw * Nw;
//     N
// }
