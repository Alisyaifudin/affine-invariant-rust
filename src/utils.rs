#![allow(non_snake_case)]
use ndarray::{Array, Array1, Array2, Axis};
use rand::Rng;

use crate::stats::{normal, uniform};

pub fn zu(u: Array1<f64>, a: f64) -> Array1<f64> {
    let z1 = 1.0 + (a - 1.0) * u.clone();
    let z2 = 1.0 + (a - 1.0) * u;
    let z = (1.0 / a) * z1 * z2;
    z
}

pub fn rand_int(n: &usize) -> usize {
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..*n);
    i
}

pub fn remove_rows<A: Clone>(matrix: &Array2<A>, to_remove: &[usize]) -> Array2<A> {
    let mut keep_row = vec![true; matrix.nrows()];
    to_remove.iter().for_each(|row| keep_row[*row] = false);

    let elements_iter = matrix
        .axis_iter(Axis(0))
        .zip(keep_row.iter())
        .filter(|(_row, keep)| **keep)
        .flat_map(|(row, _keep)| row.to_vec());

    let new_n_rows = matrix.nrows() - to_remove.len();
    Array::from_iter(elements_iter)
        .into_shape((new_n_rows, matrix.ncols()))
        .unwrap()
}
// given 1D array, repeat the array n times, forming a 2D array
pub fn repeat_1d(arr: &Array1<f64>, n: usize) -> Array2<f64> {
    let it = arr.iter().cloned().cycle().take(n * arr.len());
    let res = Array::from_iter(it).into_shape((n, arr.len())).unwrap();
    res.t().to_owned()
}

// pub const M_TRUE: f64 = -0.9594;
// pub const B_TRUE: f64 = 4.294;
// pub const F_TRUE: f64 = 0.534;

pub fn generate_data(N: &usize, m_true: &f64, b_true: &f64, f_true: &f64) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    // generate random data of type Vec
    let mut x = (0..*N)
        .map(|_| rng.gen_range(0.0..10.0))
        .collect::<Vec<f64>>();
    // sort the data
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // convert to ndarray
    let x = Array1::from(x);
    // generate yerr data, directly to ndarray
    let yerr = 0.1 + 2. * uniform::rvs(&*N, Some(0.0), Some(1.));
    // generate y data
    let y = m_true.clone() * x.clone() + b_true.clone();
    let y = y.clone()
        + (y * f_true.clone()).mapv_into(|v| v.abs()) * normal::rvs(&*N, Some(0.0), Some(1.0));
    let y = y + yerr.clone() * normal::rvs(&*N, Some(0.0), Some(1.0));
    let data = Array::from_iter(
        x.iter()
            .zip(y.iter())
            .zip(yerr.iter())
            .flat_map(|((x, y), yerr)| vec![*x, *y, *yerr]),
    )
    .into_shape((*N, 3))
    .unwrap()
    .t()
    .to_owned();
    data
}
