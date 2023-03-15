use ndarray::{Array1, Array2, Axis};
use rand::Rng;

#[allow(dead_code)]
pub fn pdf(x: &Array1<f64>, loc: &f64, scale: &f64) -> Array1<f64> {
    let prob = x.clone().mapv_into(|v| {
        if loc < &v && v < loc + scale {
            1.0 / scale
        } else {
            0.0
        }
    });
    prob
}

#[allow(dead_code)]
pub fn log_pdf(x: &Array1<f64>, loc: &f64, scale: &f64) -> Array1<f64> {
    let prob = x.clone().mapv_into(|v| {
        if loc < &v && v < loc + scale {
            (1.0 / scale).ln()
        } else {
            f64::NEG_INFINITY
        }
    });
    prob
}

#[allow(dead_code)]
pub fn rvs(size: &usize, loc: &f64, scale: &f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let u: Vec<f64> = (0..*size).map(|_| rng.gen::<f64>() * scale + loc).collect();
    Array1::from(u)
}

#[allow(dead_code)]
pub fn pdf_array1(x: &Array1<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array1<f64> {
    let prob = x
        .iter()
        .zip(loc.iter())
        .zip(scale.iter())
        .map(|((x, loc), scale)| {
            if loc < x && x < &(loc + scale) {
                1.0 / scale
            } else {
                0.0
            }
        })
        .collect::<Array1<f64>>();
    prob
}

#[allow(dead_code)]
pub fn log_pdf_array1(x: &Array1<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array1<f64> {
    let prob = x
        .iter()
        .zip(loc.iter())
        .zip(scale.iter())
        .map(|((x, loc), scale)| {
            if loc < x && x < &(loc + scale) {
                (1.0 / scale).ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect::<Array1<f64>>();
    prob
}

#[allow(dead_code)]
pub fn pdf_array2(x: &Array2<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let prob = x
        .axis_iter(Axis(0))
        .zip(loc.iter())
        .zip(scale.iter())
        .map(|((x, loc), scale)| {
            x.to_owned()
                .mapv_into(|xv| {
                    if loc < &xv && &xv < &(loc + scale) {
                        1.0 / scale
                    } else {
                        0.0
                    }
                })
                .to_vec()
        })
        .collect::<Vec<Vec<f64>>>();
    Array2::from_shape_vec((x.shape()[0], loc.shape()[0]), prob.concat()).unwrap()
}

#[allow(dead_code)]
pub fn log_pdf_array2(x: &Array2<f64>, loc: &Array1<f64>, scale: &Array1<f64>) -> Array2<f64> {
    let prob = x
        .axis_iter(Axis(0))
        .zip(loc.iter())
        .zip(scale.iter())
        .map(|((x, loc), scale)| {
            x.to_owned()
                .mapv_into(|xv| {
                    if loc < &xv && &xv < &(loc + scale) {
                        (1.0 / scale).ln()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .to_vec()
        })
        .collect::<Vec<Vec<f64>>>();
    Array2::from_shape_vec((x.shape()[0], loc.shape()[0]), prob.concat()).unwrap()
}