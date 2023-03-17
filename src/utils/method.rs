use ndarray::{s, Array, Array1, Array2, Dimension};

pub trait MinMax {
    fn max(&self) -> f64;
    fn min(&self) -> f64;
}
impl<D: Dimension> MinMax for Array<f64, D> {
    fn max(&self) -> f64 {
        let mut sort_desc = self.iter().flat_map(|x| vec![*x]).collect::<Vec<f64>>();
        sort_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
        sort_desc[0]
    }
    fn min(&self) -> f64 {
        let mut sort_asc = self.iter().flat_map(|x| vec![*x]).collect::<Vec<f64>>();
        sort_asc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sort_asc[0]
    }
}

pub trait Integration {
    fn trapz(&self, x: &Array1<f64>) -> f64;
}

impl Integration for Array1<f64> {
    fn trapz(&self, x: &Array1<f64>) -> f64 {
        let x_left = x.slice(s![0_i32..-1_i32]);
        let x_right = x.slice(s![1_i32..]);
        let y_left = self.slice(s![0_i32..-1_i32]);
        let y_right = self.slice(s![1_i32..]);
        let sum = x_right
            .iter()
            .zip(x_left.iter())
            .zip(y_right.iter())
            .zip(y_left.iter())
            .map(|(((x_right, x_left), y_right), y_left)| {
                (x_right - x_left) * (y_right + y_left) / 2.0
            })
            .sum::<f64>();
        sum
    }
}

pub trait ToArray2<T>
where
    T: Clone,
{
    fn to_array2(&self) -> Array2<T>;
}
impl<T: Clone> ToArray2<T> for Vec<Array1<T>> {
    fn to_array2(&self) -> Array2<T> {
        let rows = self.len();
        let cols = self[0].len();
        let tmp = self
            .iter()
            .map(|x| x.to_vec())
            .flatten()
            .collect::<Vec<T>>();
        Array2::from_shape_vec((rows, cols), tmp).unwrap()
    }
}

pub trait Exponential<D> {
    fn exp(&self) -> Array<f64, D>;
    fn ln(&self) -> Array<f64, D>;
}
impl<D: Dimension> Exponential<D> for Array<f64, D> {
    fn exp(&self) -> Array<f64, D> {
        self.mapv(|x| x.exp())
    }
    fn ln(&self) -> Array<f64, D> {
        self.mapv(|x| x.ln())
    }
}
pub trait Power<D> {
    fn powf(&self, p: f64) -> Array<f64, D>;
    fn powi(&self, p: i32) -> Array<f64, D>;
}
impl<D: Dimension> Power<D> for Array<f64, D> {
    fn powf(&self, p: f64) -> Array<f64, D> {
        self.mapv(|x| x.powf(p))
    }
    fn powi(&self, p: i32) -> Array<f64, D> {
        self.mapv(|x| x.powi(p))
    }
}

pub trait Sqrt<D> {
    fn sqrt(&self) -> Array<f64, D>;
}

impl<D: Dimension> Sqrt<D> for Array<f64, D> {
    fn sqrt(&self) -> Array<f64, D> {
        self.mapv(|x| x.sqrt())
    }
}
