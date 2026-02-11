//! Sampler
//!
//! Strategies for sampling data before fitting new trees, allowing for stochastic
//! gradient boosting and better regularization.
use rand::rngs::StdRng;
use rand::RngExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum SampleMethod {
    None,
    Random,
}

// A sampler can be used to subset the data prior to fitting a new tree.
pub trait Sampler {
    /// Sample the data, returning a tuple, where the first item is the samples
    /// chosen for training, and the second are the samples excluded.
    fn sample(&mut self, rng: &mut StdRng, index: &[usize]) -> (Vec<usize>, Vec<usize>);
}

pub struct RandomSampler {
    subsample: f32,
}

impl RandomSampler {
    #[allow(dead_code)]
    pub fn new(subsample: f32) -> Self {
        RandomSampler { subsample }
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self, rng: &mut StdRng, index: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let subsample = self.subsample;
        let mut chosen = Vec::new();
        let mut excluded = Vec::new();
        for i in index {
            if rng.random::<f32>() < subsample {
                chosen.push(*i);
            } else {
                excluded.push(*i)
            }
        }
        (chosen, excluded)
    }
}
