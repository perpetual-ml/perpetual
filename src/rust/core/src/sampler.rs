//! Sampler
//!
//! Strategies for sampling data before fitting new trees, allowing for stochastic
//! gradient boosting and better regularization.
use rand::RngExt;
use rand::rngs::StdRng;
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_sampler() {
        let mut rng = StdRng::seed_from_u64(42);
        let index = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut sampler = RandomSampler::new(0.5);
        let (chosen, excluded) = sampler.sample(&mut rng, &index);

        // With seed 42 and 0.5 subsample, we should get some split.
        assert!(!chosen.is_empty());
        assert!(!excluded.is_empty());
        assert_eq!(chosen.len() + excluded.len(), index.len());

        // Test with subsample 1.0 (all should be chosen)
        let mut sampler_all = RandomSampler::new(1.0);
        let (chosen_all, excluded_all) = sampler_all.sample(&mut rng, &index);
        assert_eq!(chosen_all.len(), index.len());
        assert!(excluded_all.is_empty());

        // Test with subsample 0.0 (none should be chosen)
        let mut sampler_none = RandomSampler::new(0.0);
        let (chosen_none, excluded_none) = sampler_none.sample(&mut rng, &index);
        assert!(chosen_none.is_empty());
        assert_eq!(excluded_none.len(), index.len());
    }
}
