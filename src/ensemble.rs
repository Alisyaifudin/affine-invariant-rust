// use crate::prob::log_prob;
use crate::stats::uniform;
use crate::utils;
use crate::{gravity::prob::dm::log_prob1, NDIM};
use kdam::tqdm;
use ndarray::{s, Array1, Array2, Array3, Axis};

pub type Zdata = (Array1<f64>, Array1<f64>, Array1<f64>);
pub type Wdata = (Array1<f64>, Array1<f64>, Array1<f64>, f64);
#[allow(dead_code)]
pub struct EnsembleSampler1 {
    ndim: usize,
    nwalkers: usize,
    p0: Array2<f64>,
    parallel: bool,
    chain: Array3<f64>,
    acceptance: u32,
    zdata: Zdata,
    wdata: Wdata,
    dz: Option<f64>
}

#[allow(dead_code)]
type LogProb = fn(&Array2<f64>) -> Array1<f64>;

impl EnsembleSampler1 {
    #[allow(dead_code)]
    pub fn new(
        ndim: usize,
        nwalkers: usize,
        p0: Array2<f64>,
        parallel: bool,
        zdata: Zdata,
        wdata: Wdata,
        dz: Option<f64>
    ) -> Self {
        let mut chain = Array3::zeros((1, nwalkers, ndim));
        chain.slice_mut(s![0, .., ..]).assign(&p0);
        EnsembleSampler1 {
            ndim,
            nwalkers,
            p0,
            parallel,
            chain,
            acceptance: 0,
            zdata,
            wdata,
            dz
        }
    }

    #[allow(dead_code)]
    pub fn get_data(&self) -> (Zdata, Wdata) {
        (self.zdata.clone(), self.wdata.clone())
    }
    #[allow(dead_code)]
    pub fn get_chain(&self) -> Array3<f64> {
        self.chain.clone()
    }
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.chain = self.chain.slice(s![-1..-2, .., ..]).to_owned();
        self.acceptance = 0;
    }
    #[allow(dead_code)]
    fn moves(&mut self, a: f64) {
        let u = uniform::rvs(&self.nwalkers, &0., &1.);
        let r = uniform::rvs(&self.nwalkers, &0., &1.);
        let z = utils::zu(u, a);
        let mut p_next = self.chain.slice(s![-1, .., ..]).to_owned();
        for i in 0..self.nwalkers {
            let p_rest = utils::remove_rows(&p_next, &[i]);
            let idx = utils::rand_int(&(self.nwalkers - 1));
            let Xk = p_next.slice(s![i..i + 1, ..]).to_owned();
            let Xj = p_rest.slice(s![idx..idx + 1, ..]).to_owned();
            let Y = Xj.clone() + z[i] * (Xk.clone() - Xj);
            let q = (self.ndim as f64 - 1.) * z[i].ln() + log_prob1(&Y, &self.zdata, &self.wdata, &self.dz)
                - log_prob1(&Xk, &self.zdata, &self.wdata, &self.dz);
            if q[0] > r[i].ln() {
                p_next.slice_mut(s![i..i + 1, ..]).assign(&Y);
                self.acceptance += 1;
            } else {
                p_next.slice_mut(s![i..i + 1, ..]).assign(&Xk);
            }
        }
        let a = p_next.insert_axis(Axis(0));
        self.chain.append(Axis(0), a.view()).unwrap();
    }
    fn moves_parallel(&mut self, a: f64) {
        let mut p_next: Array2<f64> = self.chain.slice(s![-1, .., ..]).to_owned();
        for i in 0..2 {
            let Xk_1: Array2<f64> = p_next.slice(s![..self.nwalkers / 2, ..]).to_owned();
            let Xk_2: Array2<f64> = p_next.slice(s![self.nwalkers / 2.., ..]).to_owned();
            let u: Array1<f64> = uniform::rvs(&(self.nwalkers / 2), &0., &1.);
            let z: Array1<f64> = utils::zu(u, a);
            let idx = (0..(self.nwalkers / 2))
                .map(|_| utils::rand_int(&(self.nwalkers / 2)))
                .collect::<Vec<usize>>();
            let Xk = if i == 0 { Xk_1.clone() } else { Xk_2.clone() };
            let Xk_other = if i == 0 { Xk_2 } else { Xk_1 };
            let Xj = idx
                .iter()
                .map(|&x| Xk_other.slice(s![x, ..]).to_owned())
                .collect::<Vec<Array1<f64>>>();
            let v = Xj.iter().flatten().cloned().collect::<Vec<f64>>();
            let Xj = Array2::from_shape_vec((self.nwalkers / 2, self.ndim), v).unwrap();
            let zz = utils::repeat_1d(&z, NDIM).t().to_owned();
            let Y = Xj.clone() + zz * (Xk.clone() - Xj);
            let q = (self.ndim as f64 - 1.) * z.mapv_into(|v| v.ln())
                + log_prob1(&Y, &self.zdata, &self.wdata, &self.dz)
                - log_prob1(&Xk, &self.zdata, &self.wdata, &self.dz);
            let r: Array1<f64> = uniform::rvs(&(self.nwalkers / 2), &0., &1.);
            let mask = r
                .iter()
                .zip(q.iter())
                .map(|(r, q)| r.ln() <= *q)
                .collect::<Vec<bool>>();
            self.acceptance += mask.iter().filter(|&&x| x).count() as u32;
            let XY = Xk.axis_iter(Axis(0)).zip(Y.axis_iter(Axis(0)));
            mask.iter()
                .enumerate()
                .zip(XY)
                .for_each(|((k, &m), (x, y))| {
                    if m {
                        p_next
                            .slice_mut(s![k + i * (self.nwalkers / 2), ..])
                            .assign(&y);
                    } else {
                        p_next
                            .slice_mut(s![k + i * (self.nwalkers / 2), ..])
                            .assign(&x);
                    }
                });
        }
        let a = p_next.insert_axis(Axis(0));
        self.chain.append(Axis(0), a.view()).unwrap();
    }
    #[allow(dead_code)]
    pub fn run_mcmc(
        &mut self,
        nsteps: usize,
        parallel: bool,
        verbose: Option<bool>,
        a: Option<f64>,
    ) {
        let a = a.unwrap_or(2.);
        let verbose = verbose.unwrap_or(false);
        for _ in tqdm!(0..nsteps) {
            if parallel {
                self.moves_parallel(a);
            } else {
                self.moves(a);
            }
        }
        if verbose {
            println!(
                "Acceptance rate: {}",
                self.acceptance as f64 / ((nsteps * self.nwalkers) as f64)
            );
        }
    }
}
