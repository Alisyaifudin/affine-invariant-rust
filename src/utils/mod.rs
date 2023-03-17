#![allow(non_snake_case)]
use ndarray::{Array, Array1, Array2};

pub mod method;

pub fn zu(u: Array1<f64>, a: f64) -> Array1<f64> {
    let z1 = 1.0 + (a - 1.0) * u.clone();
    let z2 = 1.0 + (a - 1.0) * u;
    let z = (1.0 / a) * z1 * z2;
    z
}

// pub fn rand_int(low: &usize, high: &usize, size: usize) -> Vec<usize> {
//     let mut rng = rand::thread_rng();
//     let rnd = (0..size)
//         .map(|_| rng.gen_range(low.clone()..high.clone()))
//         .collect::<Vec<usize>>();
//     rnd
// }

// pub fn remove_rows<A: Clone>(matrix: &Array2<A>, to_remove: &[usize]) -> Array2<A> {
//     let mut keep_row = vec![true; matrix.nrows()];
//     to_remove.iter().for_each(|row| keep_row[*row] = false);

//     let elements_iter = matrix
//         .axis_iter(Axis(0))
//         .zip(keep_row.iter())
//         .filter(|(_row, keep)| **keep)
//         .flat_map(|(row, _keep)| row.to_vec());

//     let new_n_rows = matrix.nrows() - to_remove.len();
//     Array::from_iter(elements_iter)
//         .into_shape((new_n_rows, matrix.ncols()))
//         .unwrap()
// }
// given 1D array, repeat the array n times, forming a 2D array
pub fn repeat_1d(arr: &Array1<f64>, n: usize) -> Array2<f64> {
    let it = arr.iter().cloned().cycle().take(n * arr.len());
    let res = Array::from_iter(it).into_shape((n, arr.len())).unwrap();
    res.to_owned()
}

// pub fn repeat_1d(arr: &Array1<f64>, n: usize) -> Array2<f64> {
//     let it = arr.iter().cloned().cycle().take(n * arr.len());
//     let res = Array::from_iter(it).into_shape((arr.len(), n)).unwrap();
//     res.t().to_owned()
// }

pub fn repeat_scalar(scalar: f64, n: usize) -> Array1<f64> {
    let it = std::iter::repeat(scalar).take(n);
    Array1::from_iter(it)
}
