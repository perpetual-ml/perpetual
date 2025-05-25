#![allow(dead_code)]
use rand::distr::{Uniform};
use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub(crate) fn create_data(n_samples: usize, n_features: usize) -> (Vec<f64>, Vec<f64>) {
    
    // reproducible seed
    let mut rng = StdRng::seed_from_u64(1903);

    // feature distributions
    let feature_distribution = Uniform::new(0.0, 1.0);
    let noise_distribution = Uniform::new(-1.0, 1.0);
    let weight_distrubtion = Uniform::new(-1.0, 1.0);

    // construct matrix-like
    // objects
    let mut feature_space: Vec<Vec<f64>> = vec![Vec::with_capacity(n_samples); n_features];
    let mut target_variable: Vec<f64> = Vec::with_capacity(n_samples);

    // generate random weights for the linear model
    let weights: Vec<f64> = (0..n_features)
        .map(|_| rng.sample(weight_distrubtion.unwrap()))
        .collect();


    for _ in 0..n_samples {

        let mut x_sample = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let v = rng.sample(feature_distribution.unwrap());
            feature_space[j].push(v);
            x_sample.push(v);
        }

        // linear model + uniform noise
        let linear: f64 = x_sample
            .iter()
            .zip(weights.iter())
            .map(|(x, w)| x * w)
            .sum();
        let y = linear + rng.sample(noise_distribution.unwrap());
        target_variable.push(y);
    }

    // flatten into column-major
    let mut data = Vec::with_capacity(n_samples * n_features);
    for col in feature_space {
        data.extend(col);
    }

    (data, target_variable)

}

// prediction_pair
//
// Generates (y, y_hat, sample_weights)-tuple
pub(crate) fn prediction_pair(n_samples: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {

    // reproducible seed
    let mut rng = StdRng::seed_from_u64(1903);
    
    // generate place holders
    let mut y: Vec<f64>     = Vec::with_capacity(n_samples);
    let mut y_hat: Vec<f64> = Vec::with_capacity(n_samples);
    let mut sample_weights: Vec<f64> = Vec::with_capacity(n_samples);

    // generate distributions
    // NOTE: we exclude negative values
    // to avoid clashes with invalid operations
    // in for example 
    let y_distribution = Uniform::new(0.0, 1.0);
    let noise_distribution = Uniform::new(0.0, 1.0);
    let weight_distribution = Uniform::new(0.0, 1.0);

    // populate vectors
    for _ in 0..n_samples {
        // extract values
        let value = rng.sample(y_distribution.unwrap());
        let noise = rng.sample(noise_distribution.unwrap());
        let weight: f64 = rng.sample(weight_distribution.unwrap());

        // add values
        y.push(value);
        y_hat.push(value + noise);
        sample_weights.push(weight);
    }

    // return tuple
    (y, y_hat, sample_weights)

}