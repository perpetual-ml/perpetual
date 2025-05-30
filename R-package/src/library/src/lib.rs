use rand::distr::{Uniform};
use rand::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use perpetual::Matrix;
use perpetual::UnivariateBooster;
use perpetual::objective_functions::*;

pub fn wrapper_dgp(n_samples: usize, n_features: usize) -> (Vec<f64>, Vec<f64>) {
    
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


#[unsafe(no_mangle)]
pub extern "C" fn wrapper_univariate(n_samples: usize, n_features: usize) {
    // prepare data
    let (data, y) = wrapper_dgp(n_samples, n_features);
    let matrix = Matrix::new(&data, n_samples, n_features);

    let mut model = UnivariateBooster::default()
        .set_objective(Objective::LogLoss)
        .set_budget(0.5);

    let _ = model.fit(&matrix, &y, None);

    println!("Model prediction: {:?} ...", &model.predict(&matrix, true)[0..10]);


}
