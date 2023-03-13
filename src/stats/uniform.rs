use ndarray::{Array1};
use rand::Rng;

#[allow(dead_code)]
pub fn pdf(x: &Array1<f64>, loc: Option<f64>, scale: Option<f64>) -> Array1<f64> {
    let loc = loc.unwrap_or( 0.0);
    let scale = scale.unwrap_or( 1.0);
    let prob = x.clone().mapv_into(|v| {
        if loc < v && v < loc + scale {
            1.0 / scale
        } else {
            0.0
        }
    });
    prob
}

#[allow(dead_code)]
pub fn log_pdf(x: &Array1<f64>, loc: Option<f64>, scale: Option<f64>) -> Array1<f64> {
    let loc = loc.unwrap_or( 0.0);
    let scale = scale.unwrap_or( 1.0);
    let prob = x.clone().mapv_into(|v| {
        if loc < v && v < loc + scale {
            (1.0 / scale).ln()
        } else {
            f64::NEG_INFINITY
        }
    });
    prob
}

#[allow(dead_code)]
pub fn rvs(size: &usize, loc: Option<f64>, scale: Option<f64>) -> Array1<f64> {
    let loc = loc.unwrap_or( 0.0);
    let scale = scale.unwrap_or( 1.0);
    let mut rng = rand::thread_rng();
    let u: Vec<f64> = (0..*size).map(|_| rng.gen::<f64>()*scale+loc).collect();
    Array1::from(u)
}