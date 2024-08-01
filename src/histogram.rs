use std::cmp::Ordering;

use crate::data::{FloatData, JaggedMatrix, Matrix};
use hashbrown::HashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Bin {
    pub g_folded: [f32; 5],
    pub h_folded: Option<[f32; 5]>,
    pub cut_value: f64,
    pub counts: [usize; 5],
    pub num: u16,
    pub col: usize,
}

impl Bin {
    pub fn new(h_folded: Option<[f32; 5]>, cut_value: f64, num: u16, col: usize) -> Self {
        Bin {
            g_folded: [f32::ZERO; 5],
            h_folded,
            cut_value,
            counts: [0; 5],
            num,
            col,
        }
    }

    pub fn from_parent_child(root_bin: &Bin, child_bin: &Bin, update_bin: &mut Bin) {
        for ((z, a), b) in update_bin
            .g_folded
            .iter_mut()
            .zip(&root_bin.g_folded)
            .zip(&child_bin.g_folded)
        {
            *z = a - b;
        }
        for ((z, a), b) in update_bin
            .counts
            .iter_mut()
            .zip(&root_bin.counts)
            .zip(&child_bin.counts)
        {
            *z = a - b;
        }

        match root_bin.h_folded {
            Some(_h_folded) => {
                let h_f_iter = update_bin.h_folded.as_mut().unwrap().iter_mut();
                for ((zval, aval), bval) in h_f_iter
                    .zip(&root_bin.h_folded.unwrap())
                    .zip(&child_bin.h_folded.unwrap())
                {
                    *zval = aval - bval;
                }
            }
            None => {
                update_bin.h_folded = None;
            }
        };
    }

    pub fn from_parent_two_children(root_bin: &Bin, first_bin: &Bin, second_bin: &Bin, update_bin: &mut Bin) {
        for (((z, a), b), c) in update_bin
            .g_folded
            .iter_mut()
            .zip(&root_bin.g_folded)
            .zip(&first_bin.g_folded)
            .zip(&second_bin.g_folded)
        {
            *z = a - b - c;
        }
        for (((z, a), b), c) in update_bin
            .counts
            .iter_mut()
            .zip(&root_bin.counts)
            .zip(&first_bin.counts)
            .zip(&second_bin.counts)
        {
            *z = a - b - c;
        }

        match root_bin.h_folded {
            Some(_h_folded) => {
                let h_f_iter = update_bin.h_folded.as_mut().unwrap().iter_mut();
                for (((z, a), b), c) in h_f_iter
                    .zip(&root_bin.h_folded.unwrap())
                    .zip(&first_bin.h_folded.unwrap())
                    .zip(&second_bin.h_folded.unwrap())
                {
                    *z = a - b - c;
                }
            }
            None => {
                update_bin.h_folded = None;
            }
        };
    }
}

pub fn create_empty_histogram(cuts: &[f64], is_const_hess: bool, col: usize) -> Vec<Bin> {
    let mut histogram: Vec<Bin> = Vec::with_capacity(cuts.len());
    if is_const_hess {
        histogram.push(Bin::new(None, f64::NAN, 0, col));
        histogram.extend(
            cuts[..(cuts.len() - 1)]
                .iter()
                .enumerate()
                .map(|(it, c)| Bin::new(None, *c, it as u16 + 1, col)),
        );
    } else {
        histogram.push(Bin::new(Some([f32::ZERO; 5]), f64::NAN, 0, col));
        histogram.extend(
            cuts[..(cuts.len() - 1)]
                .iter()
                .enumerate()
                .map(|(it, c)| Bin::new(Some([f32::ZERO; 5]), *c, it as u16 + 1, col)),
        );
    }
    histogram
}

pub fn update_feature_histogram(
    histogram: &mut [Bin],
    feature: &[u16], // an array which shows the bin index for each element of the feature, length = whole length of data
    sorted_grad: &[f32], // grad with length of data that falls into this node
    sorted_hess: Option<&[f32]>, // hess with length of data that falls into this split
    index: &[usize], // indices with length of data that falls into this node
) {
    match sorted_hess {
        Some(sorted_hess) => {
            histogram.iter_mut().for_each(|hist| {
                hist.g_folded = [f32::ZERO; 5];
                hist.h_folded = Some([f32::ZERO; 5]);
                hist.counts = [0; 5];
            });
            index.iter().zip(sorted_grad).zip(sorted_hess).for_each(|((i, g), h)| {
                if let Some(v) = histogram.get_mut(feature[*i] as usize) {
                    let fold = i % 5;
                    v.g_folded[fold] += *g;
                    let k = v.h_folded.as_mut().unwrap();
                    k[fold] += *h;
                    v.counts[fold] += 1;
                }
            });
        }
        None => {
            histogram.iter_mut().for_each(|hist| {
                hist.g_folded = [f32::ZERO; 5];
                hist.counts = [0; 5];
            });
            index.iter().zip(sorted_grad).for_each(|(i, g)| {
                if let Some(v) = histogram.get_mut(feature[*i] as usize) {
                    let fold = i % 5;
                    v.g_folded[fold] += *g;
                    v.counts[fold] += 1;
                }
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_histogram(
    hist: &mut HistogramMatrix,
    start: usize,
    stop: usize,
    data: &Matrix<u16>,
    grad: &[f32],
    hess: Option<&[f32]>,
    index: &[usize],
    col_index: &[usize],
    parallel: bool,
    sort: bool,
) {
    let (sorted_grad, sorted_hess) = match hess {
        Some(hess) => {
            if !sort {
                (grad, Some(hess))
            } else {
                let g = &grad[start..stop];
                let h = &hess[start..stop];
                (g, Some(h))
            }
        }
        None => {
            if !sort {
                (grad, None)
            } else {
                let g = &grad[start..stop];
                (g, None)
            }
        }
    };

    if parallel {
        let feature_histograms = hist.0.data.par_chunk_by_mut(|a, b| a.num < b.num);

        feature_histograms.for_each(|h| {
            let feature_col = data.get_col(h[0].col);
            update_feature_histogram(
                h,
                feature_col,
                &sorted_grad,
                sorted_hess.as_deref(),
                &index[start..stop],
            );
        });
    } else {
        col_index.iter().for_each(|col| {
            update_feature_histogram(
                hist.0.get_col_mut(*col),
                data.get_col(*col),
                &sorted_grad,
                sorted_hess.as_deref(),
                &index[start..stop],
            );
        });
    }
}

/// Histograms implemented as as jagged matrix.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HistogramMatrix(pub JaggedMatrix<Bin>);

impl HistogramMatrix {
    /// Create an empty histogram matrix.
    pub fn empty(
        data: &Matrix<u16>,
        cuts: &JaggedMatrix<f64>,
        col_index: &[usize],
        is_const_hess: bool,
        parallel: bool,
    ) -> Self {
        // If we have sampled down the columns, we need to recalculate the ends.
        // we can do this by iterating over the cut's, as this will be the size
        // of the histograms.
        let ends: Vec<usize> = if col_index.len() == data.cols {
            cuts.ends.to_owned()
        } else {
            col_index
                .iter()
                .scan(0_usize, |state, i| {
                    *state += cuts.get_col(*i).len();
                    Some(*state)
                })
                .collect()
        };

        let n_records = if col_index.len() == data.cols {
            cuts.n_records
        } else {
            ends.iter().sum()
        };

        let histograms = if parallel {
            col_index
                .par_iter()
                .flat_map(|col| create_empty_histogram(cuts.get_col(*col), is_const_hess, *col))
                .collect::<Vec<Bin>>()
        } else {
            col_index
                .iter()
                .flat_map(|col| create_empty_histogram(cuts.get_col(*col), is_const_hess, *col))
                .collect::<Vec<Bin>>()
        };

        HistogramMatrix(JaggedMatrix {
            data: histograms,
            ends,
            cols: col_index.len(),
            n_records,
        })
    }

    /// Calculate the histogram matrix, for a child, given the parent histogram
    /// matrix, and the other child histogram matrix. This should be used
    /// when the node has only two possible splits, left and right.
    pub fn from_parent_child(
        hist_tree: &mut HashMap<usize, HistogramMatrix>,
        root_num: usize,
        child_num: usize,
        update_num: usize,
    ) {
        unsafe {
            let mut histograms = hist_tree
                .get_many_unchecked_mut([&root_num, &child_num, &update_num])
                .unwrap();

            let (last, rest) = histograms.split_last_mut().unwrap();
            let (child, root) = rest.split_last_mut().unwrap();

            let root_hist = &mut root[0].0;
            let child_hist = &mut child.0;
            let update_hist = &mut last.0;

            root_hist
                .data
                .iter()
                .zip(child_hist.data.iter_mut())
                .zip(update_hist.data.iter_mut())
                .for_each(|((root_bin, child_bin), update_bin)| {
                    Bin::from_parent_child(root_bin, child_bin, update_bin)
                });
        }
    }

    /// Calculate the histogram matrix for a child, given the parent histogram
    /// and two other child histograms. This should be used with the node has
    /// three possible split paths, right, left, and missing.
    pub fn from_parent_two_children(
        hist_tree: &mut HashMap<usize, HistogramMatrix>,
        root_num: usize,
        first_num: usize,
        second_num: usize,
        update_num: usize,
    ) {
        unsafe {
            let mut histograms = hist_tree
                .get_many_unchecked_mut([&root_num, &first_num, &second_num, &update_num])
                .unwrap();

            // Switch to get_many_unchecked_mut when it is stable
            // https://doc.rust-lang.org/std/primitive.slice.html#method.get_many_unchecked_mut
            let (last, rest) = histograms.split_last_mut().unwrap();
            let root_hist = &rest[0].0.data;
            let first_hist = &rest[1].0.data;
            let second_hist = &rest[2].0.data;
            let update_hist = &mut last.0.data;

            root_hist
                .iter()
                .zip(first_hist.iter())
                .zip(second_hist.iter())
                .zip(update_hist.iter_mut())
                .for_each(|(((root_bin, first_bin), second_bin), update_bin)| {
                    Bin::from_parent_two_children(root_bin, first_bin, second_bin, update_bin)
                });
        }
    }
}

pub fn sort_cat_bins(mut histograms: [&mut HistogramMatrix; 3], cat_index: &[u64]) {
    for hist in histograms.iter_mut() {
        for c in cat_index {
            let feature_hist = hist.0.get_col_mut((*c) as usize);
            feature_hist.sort_unstable_by_key(|bin| bin.num);
        }
    }
}

pub fn reorder_cat_bins(mut histograms: [&mut HistogramMatrix; 3], cat_index: &[u64], is_const_hess: bool) {
    cat_index.iter().for_each(|col| {
        for hist in histograms.iter_mut() {
            if is_const_hess {
                hist.0.get_col_mut((*col) as usize).sort_unstable_by(|bin1, bin2| {
                    if bin1.num == 0 {
                        return Ordering::Less;
                    } else if bin2.num == 0 {
                        return Ordering::Greater;
                    }
                    let div1: f32 = bin1.g_folded.iter().sum::<f32>() / bin1.h_folded.unwrap().iter().sum::<f32>();
                    let div2: f32 = bin2.g_folded.iter().sum::<f32>() / bin2.h_folded.unwrap().iter().sum::<f32>();
                    div2.partial_cmp(&div1).unwrap_or(Ordering::Less)
                });
            } else {
                hist.0.get_col_mut((*col) as usize).sort_unstable_by(|bin1, bin2| {
                    if bin1.num == 0 {
                        return Ordering::Less;
                    } else if bin2.num == 0 {
                        return Ordering::Greater;
                    }
                    let div1: f32 = bin1.g_folded.iter().sum::<f32>() / bin1.counts.iter().sum::<usize>() as f32;
                    let div2: f32 = bin2.g_folded.iter().sum::<f32>() / bin2.counts.iter().sum::<usize>() as f32;
                    div2.partial_cmp(&div1).unwrap_or(Ordering::Less)
                });
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_matrix;
    use crate::objective::{LogLoss, ObjectiveFunction};
    use std::fs;
    #[test]
    fn test_single_histogram() {
        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let b = bin_matrix(&data, None, 10, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut hist_init = HistogramMatrix::empty(&bdata, &b.cuts, &col_index, true, false);
        update_feature_histogram(
            &mut hist_init.0.get_col_mut(1),
            &bdata.get_col(1),
            &g,
            h.as_deref(),
            &bdata.index,
        );

        let mut f = bdata.get_col(1).to_owned();

        println!("histogram:");
        println!("{:?}", hist_init.0.get_col(1));
        println!("feature_data:");
        println!("{:?}", &data.get_col(1));
        println!("feature_data_bin_indices:");
        println!("{:?}", &bdata.get_col(1));
        println!("data_indices:");
        println!("{:?}", &bdata.index);
        println!("cuts:");
        println!("{:?}", &b.cuts.get_col(1));
        f.sort();
        f.dedup();
        println!("f:");
        println!("{:?}", &f);
        println!("{:?}", &f.len());
        println!("{:?}", &hist_init.0.get_col(1).len());
        assert_eq!(f.len() + 1, hist_init.0.get_col(1).len());
    }

    #[test]
    fn test_single_histogram_categorical() {
        let file = fs::read_to_string("resources/adult_train_flat.csv").expect("Something went wrong reading the file");
        let n_rows = 39073;
        let n_columns = 14;
        let n_lines = n_columns * 39073;
        let data_vec: Vec<f64> = file
            .lines()
            .take(n_lines)
            .map(|x| x.trim().parse::<f64>().unwrap_or(f64::NAN))
            .collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);
        let b = bin_matrix(&data, None, 256, f64::NAN, Some(&vec![1])).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let yhat = vec![0.5; y.len()];
        let (g, h) = LogLoss::calc_grad_hess(&y, &yhat, None, None);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut hist_init = HistogramMatrix::empty(&bdata, &b.cuts, &col_index, false, false);

        let hist_col = 1;
        update_histogram(
            &mut hist_init,
            0,
            bdata.index.len(),
            &bdata,
            &g,
            h.as_deref(),
            &bdata.index,
            &col_index,
            true,
            false,
        );

        let mut f = bdata.get_col(hist_col).to_owned();

        println!("histogram:");
        println!("{:?}", hist_init.0.get_col(hist_col));
        println!("cuts:");
        println!("{:?}", &b.cuts.get_col(hist_col));
        f.sort();
        f.dedup();
        println!("f:");
        println!("{:?}", &f);
        println!("{:?}", &f.len());
        println!("{:?}", &hist_init.0.get_col(hist_col).len());
        assert_eq!(f.len(), hist_init.0.get_col(hist_col).len());
    }
}
