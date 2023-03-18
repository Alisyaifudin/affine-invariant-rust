// use crate::prob::log_prob;
use crate::stats::uniform;
use crate::utils;
use crate::utils::method::{Exponential, ToArray2};
use kdam::tqdm;
use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::seq::index::sample;
use rand::seq::SliceRandom;

type LogProb = Box<dyn Fn(&Array2<f64>) -> (Array1<f64>, Array1<f64>)>;

#[allow(dead_code)]
pub struct EnsembleSampler {
    ndim: usize,
    nwalkers: usize,
    p0: Array2<f64>,
    parallel: bool,
    chain: Array3<f64>,
    probs: Array3<f64>,
    acceptance: u32,
    log_prob: LogProb,
}

#[allow(dead_code)]
impl EnsembleSampler {
    fn log_prob(&self, x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        (self.log_prob)(x)
    }
    #[allow(dead_code)]
    pub fn new(
        ndim: usize,
        nwalkers: usize,
        p0: Array2<f64>,
        parallel: bool,
        log_prob: LogProb,
    ) -> Self {
        let mut chain = Array3::zeros((1, nwalkers, ndim));
        chain.slice_mut(s![0, .., ..]).assign(&p0);
        let probs = Array3::zeros((0, nwalkers, 3));
        EnsembleSampler {
            ndim,
            nwalkers,
            p0,
            parallel,
            chain,
            probs,
            acceptance: 0,
            log_prob,
        }
    }
    #[allow(dead_code)]
    pub fn get_chain(&self) -> Array3<f64> {
        self.chain.clone()
    }
    pub fn get_probs(&self) -> Array3<f64> {
        self.probs.clone()
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
        let mut probs_next = self.probs.slice(s![-1, .., ..]).to_owned();
        (0..self.nwalkers).for_each(|i| loop {
            let xk = p_next.slice_axis_mut(Axis(0), (i..i + 1).into()).to_owned();
            let indices = sample(&mut rand::thread_rng(), p_next.nrows(), 1);
            let xj = indices
                .into_iter()
                .map(|i| p_next.row(i).to_owned())
                .collect::<Vec<Array1<f64>>>()
                .to_array2();
            let y = xj.clone() + z[i] * (xk.clone() - xj);
            let test = y.sum();
            if test == f64::NAN {
                continue;
            }
            let prob_xk = self.log_prob(&xk);
            let prob_y = self.log_prob(&y);
            let q = (self.ndim as f64 - 1.) * z[i].ln() + prob_y.1 - prob_xk.1;
            if q[0] > r[i].ln() {
                p_next.slice_mut(s![i..i + 1, ..]).assign(&y);
                // append y to posterior index i
                let posterior = q[0];
                let prior = prob_y.0[0];
                let likelihood = posterior - prior;
                let probs = ndarray::arr1(&[posterior, prior, likelihood]);
                let probs = probs.insert_axis(Axis(0));
                probs_next.slice_mut(s![i..i + 1, ..]).assign(&probs.view());
                self.acceptance += 1;
            } else {
                p_next.slice_mut(s![i..i + 1, ..]).assign(&xk);
            }
            break;
        });
        let a = p_next.insert_axis(Axis(0));
        self.chain.append(Axis(0), a.view()).unwrap();
        let a = probs_next.insert_axis(Axis(0));
        self.probs.append(Axis(0), a.view()).unwrap();
    }
    fn moves_parallel(&mut self, batch: usize, a: f64) {
        let mut p_next: Array2<f64> = self.chain.slice(s![-1, .., ..]).to_owned();
        let mut probs_next = self.probs.slice(s![-1, .., ..]).to_owned();
        let sub_walkers = self.nwalkers / batch;
        (0..batch).for_each(|i| {
            loop {
                let start = i * sub_walkers;
                let end = (i + 1) * sub_walkers;
                let end = if end > self.nwalkers {
                    self.nwalkers
                } else {
                    end
                };
                let sub_walkers = end - start;
                // pop p_next by first axis from start..end
                let x_left = p_next.slice(s![..start, ..]).to_owned();
                let xk = p_next.slice(s![start..end, ..]).to_owned();
                let x_right = p_next.slice(s![end.., ..]).to_owned();
                let x_rest =
                    ndarray::concatenate(Axis(0), &[x_left.view(), x_right.view()]).unwrap();
                let u: Array1<f64> = uniform::rvs(&sub_walkers, &0., &1.);
                let z: Array1<f64> = utils::zu(u, a);
                let indices = sample(&mut rand::thread_rng(), x_rest.nrows(), sub_walkers);
                // println!("Indices {:?}", indices);
                let xj = indices
                    .into_iter()
                    .map(|i| x_rest.row(i).to_owned())
                    .collect::<Vec<Array1<f64>>>()
                    .to_array2();
                // println!("Parallel moves {}", 8);
                // println!("Xj {:?}", Xj.raw_dim());
                let zz = utils::repeat_1d(&z, xj.raw_dim()[1]).t().to_owned();
                // println!("Parallel moves {}", 9);
                // println!("zz {:?}", zz);
                // println!("zz dim {:?}", zz.raw_dim());
                let y = xj.clone() + zz * (xk.clone() - xj);
                let test = y.sum();
                if test == f64::NAN {
                    continue;
                }
                // println!("Parallel moves {}", 10);

                // println!("Parallel moves {}", 11);
                // println!("q1 {:?}", q1.raw_dim());
                let prob_xk = self.log_prob(&xk);
                let prob_y = self.log_prob(&y);
                // println!("q2 {:?}", q2.raw_dim());
                // println!("Parallel moves {}", 11);
                let q = (self.ndim as f64 - 1.) * z.ln() + prob_y.clone().1 - prob_xk.clone().1;
                // println!("Parallel moves {}", 12);
                let r: Array1<f64> = uniform::rvs(&sub_walkers, &0., &1.).ln();
                // println!("Parallel moves {}", 13);
                let mask = r
                    .iter()
                    .zip(q.iter())
                    .map(|(r, q)| r <= q)
                    .collect::<Vec<bool>>();
                // println!("Parallel moves {}", 14);
                self.acceptance += mask.iter().filter(|&&x| x).count() as u32;
                // println!("Parallel moves {}", 15);
                let xy = xk.axis_iter(Axis(0)).zip(y.axis_iter(Axis(0)));
                // println!("Parallel moves {}", 16);
                mask.iter()
                    .enumerate()
                    .zip(xy)
                    .for_each(|((k, &m), (x, y))| {
                        if m {
                            // println!("Parallel moves {}", 17);
                            p_next.slice_mut(s![k + i * sub_walkers, ..]).assign(&y);
                            let prior = prob_y.0[k];
                            let posterior = prob_y.1[k];
                            let likelihood = posterior - prior;
                            let probs = ndarray::arr1(&[posterior, prior, likelihood]);
                            let probs = probs.insert_axis(Axis(0));
                            probs_next
                                .slice_mut(s![k + i * sub_walkers, ..])
                                .assign(&probs.view());
                            // println!("Parallel moves {}", 18);
                        } else {
                            // println!("Parallel moves {}", 19);
                            p_next.slice_mut(s![k + i * sub_walkers, ..]).assign(&x);
                            let prior = prob_xk.0[k];
                            let posterior = prob_xk.1[k];
                            let likelihood = posterior - prior;
                            let probs = ndarray::arr1(&[posterior, prior, likelihood]);
                            let probs = probs.insert_axis(Axis(0));
                            probs_next
                                .slice_mut(s![k + i * sub_walkers, ..])
                                .assign(&probs.view());
                            // println!("Parallel moves {}", 20);
                        }
                    });
                break;
            }
        });
        // println!("Parallel moves {}", 21);
        let a = p_next.insert_axis(Axis(0));
        // println!("Parallel moves {}", 22);
        self.chain.append(Axis(0), a.view()).unwrap();
        // println!("Parallel moves {}", 23);
        self.chain.shuffle_axis(1);
        // println!("Parallel moves {}", 24);
    }
    #[allow(dead_code)]
    pub fn run_mcmc(&mut self, nsteps: usize, parallel: bool, batch: usize, verbose: bool, a: f64) {
        for _ in tqdm!(0..nsteps) {
            if parallel {
                self.moves_parallel(batch, a);
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

trait ShuffleArray3 {
    fn shuffle_axis(&mut self, axis: usize);
}

impl ShuffleArray3 for Array3<f64> {
    fn shuffle_axis(&mut self, axis: usize) {
        let axis = Axis(axis);

        let num_slices = self.len_of(axis);

        self.axis_iter_mut(axis)
            .into_iter()
            .skip(1)
            .take(num_slices - 2)
            .collect::<Vec<_>>()
            .as_mut_slice()
            .shuffle(&mut rand::thread_rng());
    }
}
