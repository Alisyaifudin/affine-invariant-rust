use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use std::f64::consts::PI;

#[allow(dead_code)]
pub fn pdf(x: &Array1<f64>, loc: &f64, scale: &f64) -> Array1<f64> {
    let exp = -0.5 * ((x.clone() - *loc) / *scale).mapv_into(|v| v.powi(2));
    let prob = (1.0 / (scale * (2.0 * PI).sqrt())) * exp.mapv_into(|v| v.exp());
    prob
}

#[allow(dead_code)]
pub fn log_pdf(x: &Array1<f64>, loc: &f64, scale: &f64) -> Array1<f64> {
    let exp = -0.5 * ((x.clone() - *loc) / *scale).mapv_into(|v| v.powi(2));
    let prob = (1.0 / (scale * (2.0 * PI).sqrt())).ln() + exp;
    prob
}

#[allow(dead_code)]
pub fn rvs(size: &usize, mu: &f64, sigma: &f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let u1: Vec<f64> = (0..*size).map(|_| rng.gen()).collect();
    let u2: Vec<f64> = (0..*size).map(|_| rng.gen()).collect();
    // use Box-Muller transform
    let z = u1
        .iter()
        .zip(u2.iter())
        .map(|(u1, u2)| {
            let z1 = (-2.0 * u1.ln()).sqrt();
            let z2 = (2.0 * PI * u2).cos();
            let z = z1 * z2;
            z * sigma + mu
        })
        .collect::<Vec<f64>>();
    Array1::from(z)
}

// array form

#[allow(dead_code)]
pub fn pdf_array1(x: &Array1<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array1<f64> {
    let prob = x
        .iter()
        .zip(loc.iter())
        .zip(scale.iter())
        .map(|((x, loc), scale)| {
            (1. / (scale * (2. * PI).sqrt())) * (-0.5 * ((x - loc) / scale).powi(2)).exp()
        })
        .collect::<Array1<f64>>();
    prob
}

#[allow(dead_code)]
pub fn log_pdf_array1(x: &Array1<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array1<f64> {
    let prob = pdf_array1(x, loc, scale).mapv_into(|p| p.ln());
    prob
}

#[allow(dead_code)]
pub fn pdf_array2(x: &Array2<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let prob = x
        .axis_iter(Axis(0))
        .map(|xs| pdf_array1(&xs.to_owned(), loc, scale).to_vec())
        .collect::<Vec<Vec<f64>>>();
    Array2::from_shape_vec((x.shape()[0], loc.shape()[0]), prob.concat()).unwrap()
}

#[allow(dead_code)]
pub fn log_pdf_array2(x: &Array2<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let prob = pdf_array2(x, loc, scale).mapv_into(|p| p.ln());
    prob
}
