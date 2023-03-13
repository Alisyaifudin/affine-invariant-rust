use ndarray::{Array1};
use std::f64::consts::PI;
use rand::Rng;

#[allow(dead_code)]
pub fn log_pdf(x: &Array1<f64>, loc: Option<f64>, scale: Option<f64>) -> Array1<f64> {
    let mu = loc.unwrap_or( 0.0);
    let sigma = scale.unwrap_or( 1.0);
    let exp = -0.5 * ((x.clone() - mu) / sigma).mapv_into(|v| v.powi(2));
    let prob = (1.0 / (sigma * 2.0 * PI).sqrt()).ln() + exp;
    prob
}

#[allow(dead_code)]
pub fn pdf(x: &Array1<f64>, loc: Option<f64>, scale: Option<f64>) -> Array1<f64> {
    let mu = loc.unwrap_or( 0.0);
    let sigma = scale.unwrap_or( 1.0);
    let exp = -0.5 * ((x.clone() - mu) / sigma).mapv_into(|v| v.powi(2));
    let prob = (1.0 / (sigma * 2.0 * PI).sqrt()) * exp.mapv_into(|v| v.exp());
    prob
}

#[allow(dead_code)]
pub fn rvs(size: &usize, mu: Option<f64>, sigma: Option<f64>) -> Array1<f64> {
    let mu = mu.unwrap_or( 0.0);
    let sigma = sigma.unwrap_or( 1.0);
    let mut rng = rand::thread_rng();
    let u1: Vec<f64> = (0..*size).map(|_| rng.gen()).collect();
    let u2: Vec<f64> = (0..*size).map(|_| rng.gen()).collect();
    // use Box-Muller transform
    let z = u1.iter().zip(u2.iter()).map(|(u1, u2)| {
        let z1 = (-2.0 * u1.ln()).sqrt();
        let z2 = (2.0 * PI * u2).cos();
        let z = z1 * z2;
        z * sigma + mu
    }).collect::<Vec<f64>>();  
    Array1::from(z)
}