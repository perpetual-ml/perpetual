use std::collections::HashSet;

use crate::data::{FloatData, JaggedMatrix, Matrix};
use crate::errors::PerpetualError;
use crate::utils::{is_missing, map_bin, percentiles};

/// If there are fewer unique values than their are
/// percentiles, just return the unique values of the
/// vectors.
///
/// * `v` - A numeric slice to calculate percentiles for.
/// * `sample_weight` - Instance weights for each row in the data.
fn percentiles_or_value<T>(v: &[T], sample_weight: &[T], pcts: &[T]) -> Vec<T>
where
    T: FloatData<T>,
{
    let mut v_u = v.to_owned();
    v_u.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    v_u.dedup();
    if v_u.len() <= pcts.len() + 1 {
        v_u
    } else {
        percentiles(v, sample_weight, pcts)
    }
}

// We want to be able to bin our dataset into discrete buckets.
// First we will calculate percentiles and the number of unique values
// for each feature.
// Then we will bucket them into bins from 0 to N + 1 where N is the number
// of unique bin values created from the percentiles, and the very last
// bin is missing values.
// For now, we will just use usize, although, it would be good to see if
// we can use something smaller, u8 for instance.
// If we generated these cuts:
// [0.0, 7.8958, 14.4542, 31.0, 512.3292, inf]
// We would have a number with bins 0 (missing), 1 [MIN, 0.0), 2 (0.0, 7], 3 [], 4, 5
// a split that is [feature < 5] would translate to [feature < 31.0 ]
#[derive(Debug)]
pub struct BinnedData<T> {
    pub binned_data: Vec<u16>,
    pub cuts: JaggedMatrix<T>,
    pub nunique: Vec<usize>,
}

/// Convert a matrix of data, into a binned matrix.
///
/// * `data` - Numeric data to be binned.
/// * `cuts` - A slice of Vectors, where the vectors are the corresponding
///     cut values for each of the columns.
fn bin_matrix_from_cuts<T: FloatData<T>>(data: &Matrix<T>, cuts: &JaggedMatrix<T>, missing: &T) -> Vec<u16> {
    // loop through the matrix, binning the data.
    // We will determine the column we are in, by
    // using the modulo operator, on the record value.
    data.data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let col = i / data.rows;
            // This will always be smaller than u16::MAX so we
            // are good to just unwrap here.
            map_bin(cuts.get_col(col), v, missing).unwrap()
        })
        .collect()
}

/// Bin a numeric matrix.
///
/// * `data` - A numeric matrix, of data to be binned.
/// * `sample_weight` - Instance weights for each row of the data.
/// * `nbins` - The number of bins each column should be binned into.
/// * `missing` - Float value to consider as missing.
pub fn bin_matrix(
    data: &Matrix<f64>,
    sample_weight: Option<&[f64]>,
    nbins: u16,
    missing: f64,
    cat_index: Option<&HashSet<usize>>,
) -> Result<BinnedData<f64>, PerpetualError> {
    let mut pcts = Vec::new();
    let nbins_ = f64::from_u16(nbins);
    for i in 0..nbins {
        let v = f64::from_u16(i) / nbins_;
        pcts.push(v);
    }

    let s_w = vec![1.0; data.rows];
    let weight = match sample_weight {
        Some(sample_weight) => sample_weight,
        None => &s_w,
    };

    let to_remove = match cat_index {
        Some(cat_index) => HashSet::from_iter(cat_index),
        None => HashSet::new(),
    };
    let mut num_index: Vec<usize> = (0..data.cols).collect();
    num_index.retain(|e| !to_remove.contains(&(*e)));
    let num_index_set: HashSet<usize> = HashSet::from_iter(num_index);

    // First we need to generate the bins for each of the columns.
    // We will loop through all of the columns, and generate the cuts.
    let mut cuts = JaggedMatrix::new();
    let mut nunique = Vec::new();
    for i in 0..data.cols {
        let (no_miss, w): (Vec<f64>, Vec<f64>) = data
            .get_col(i)
            .iter()
            .zip(weight.iter())
            // It is unrecoverable if they have provided missing values in
            // the data other than the specificized missing.
            .filter(|(v, _)| !is_missing(v, &missing))
            .unzip();
        assert_eq!(no_miss.len(), w.len());

        if num_index_set.contains(&i) {
            let mut col_cuts = percentiles_or_value(&no_miss, &w, &pcts);
            col_cuts.push(f64::MAX);
            col_cuts.dedup();
            // if col_cuts.len() < 2 {
            //     return Err(PerpetualError::NoVariance(i));
            // }
            // There will be one less bins, then there are cuts.
            // The first value will be for missing.
            nunique.push(col_cuts.len());
            let l = col_cuts.len();
            cuts.data.extend(col_cuts);
            let e = match cuts.ends.last() {
                Some(v) => v + l,
                None => l,
            };
            cuts.ends.push(e);
        } else {
            // There will be number of bins as many as number of categories. Number of bins for categorical features is not limited currently.
            let col_categories: HashSet<u16> = HashSet::from_iter(no_miss.iter().map(|&e| e as u16));
            let mut col_cuts: Vec<f64> = col_categories.iter().map(|&e| e as f64).collect();
            col_cuts.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            col_cuts.push(f64::MAX);
            nunique.push(col_cuts.len());
            let l = col_cuts.len();
            cuts.data.extend(col_cuts);
            let e = match cuts.ends.last() {
                Some(v) => v + l,
                None => l,
            };
            cuts.ends.push(e);
        }
    }

    cuts.cols = cuts.ends.len();
    cuts.n_records = cuts.ends.iter().sum();

    let binned_data = bin_matrix_from_cuts(data, &cuts, &missing);

    Ok(BinnedData {
        binned_data,
        cuts,
        nunique,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn test_bin_data() {
        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let b = bin_matrix(&data, None, 10, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        for column in 0..data.cols {
            let mut b_compare = 1;
            for cuts in b.cuts.get_col(column).windows(2) {
                let c1 = cuts[0];
                let c2 = cuts[1];
                let mut n_v = 0;
                let mut n_b = 0;
                for (bin, value) in bdata.get_col(column).iter().zip(data.get_col(column)) {
                    if *bin == b_compare {
                        n_b += 1;
                    }
                    if (c1 <= *value) && (*value < c2) {
                        n_v += 1;
                    }
                }
                assert_eq!(n_v, n_b);
                b_compare += 1;
            }
        }
        println!("{:?}", b);
    }

    #[test]
    fn test_bin_data_categorical() {
        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let n_rows = 712;
        let n_columns = 13;
        let n_lines = n_columns * n_rows;
        let data_vec: Vec<f64> = file
            .lines()
            .take(n_lines)
            .map(|x| x.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);
        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let b = bin_matrix(&data, None, 256, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        println!("{:?}", b.cuts);
        println!("{:?}", b.nunique);

        for column in 0..data.cols {
            let mut b_compare = 1;
            for cuts in b.cuts.get_col(column).windows(2) {
                let c1 = cuts[0];
                let c2 = cuts[1];
                let mut n_v = 0;
                let mut n_b = 0;
                for (bin, value) in bdata.get_col(column).iter().zip(data.get_col(column)) {
                    if *bin == b_compare {
                        n_b += 1;
                    }
                    if (c1 <= *value) && (*value < c2) {
                        n_v += 1;
                    }
                }
                assert_eq!(n_v, n_b);
                b_compare += 1;
            }
        }
    }
}
