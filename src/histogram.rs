use crate::bin::Bin;
use crate::data::{FloatData, JaggedMatrix};
use crate::Matrix;
use rayon::{prelude::*, ThreadPool};
use std::cell::UnsafeCell;

#[derive(Debug)]
pub struct FeatureHistogramOwned {
    pub data: Vec<Bin>,
}

impl FeatureHistogramOwned {
    pub fn empty_from_cuts(cuts: &[f64], is_const_hess: bool) -> Self {
        let mut histogram: Vec<Bin> = Vec::with_capacity(cuts.len());
        if is_const_hess {
            histogram.push(Bin::empty_const_hess(0, f64::NAN));
            histogram.extend(
                cuts[..(cuts.len() - 1)]
                    .iter()
                    .enumerate()
                    .map(|(it, c)| Bin::empty_const_hess(it as u16 + 1, *c)),
            );
        } else {
            histogram.push(Bin::empty(0, f64::NAN));
            histogram.extend(
                cuts[..(cuts.len() - 1)]
                    .iter()
                    .enumerate()
                    .map(|(it, c)| Bin::empty(it as u16 + 1, *c)),
            );
        }
        FeatureHistogramOwned { data: histogram }
    }

    pub fn empty(max_bin: u16, is_const_hess: bool) -> Self {
        let mut histogram: Vec<Bin> = Vec::with_capacity(max_bin.into());
        if is_const_hess {
            histogram.push(Bin::empty_const_hess(0, f64::NAN));
            histogram.extend((0..(max_bin + 1)).map(|i| Bin::empty_const_hess(i + 1, f64::NAN)));
        } else {
            histogram.push(Bin::empty(0, f64::NAN));
            histogram.extend((0..(max_bin + 1)).map(|i| Bin::empty(i + 1, f64::NAN)));
        }
        FeatureHistogramOwned { data: histogram }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FeatureHistogram<'a> {
    pub data: &'a [UnsafeCell<Bin>],
}

unsafe impl<'a> Send for FeatureHistogram<'a> {}
unsafe impl<'a> Sync for FeatureHistogram<'a> {}

impl<'a> FeatureHistogram<'a> {
    pub fn new(hist: &'a mut [Bin]) -> Self {
        let ptr = hist as *mut [Bin] as *const [UnsafeCell<Bin>];
        Self { data: unsafe { &*ptr } }
    }

    pub unsafe fn update(
        &self,
        feature: &[u16], // an array which shows the bin index for each element of the feature, length = whole length of data
        sorted_grad: &[f32], // grad with length of data that falls into this node
        sorted_hess: Option<&[f32]>, // hess with length of data that falls into this split
        index: &[usize], // indices with length of data that falls into this node
    ) {
        match sorted_hess {
            Some(sorted_hess) => {
                self.data.iter().for_each(|b| {
                    let bin = b.get().as_mut().unwrap();
                    bin.g_folded = [f32::ZERO; 5];
                    bin.h_folded = Some([f32::ZERO; 5]);
                    bin.counts = [0; 5];
                });
                index.iter().zip(sorted_grad).zip(sorted_hess).for_each(|((i, g), h)| {
                    let b = self.data.get_unchecked(feature[*i] as usize).get();
                    let bin = b.as_mut().unwrap_unchecked();
                    let fold = i % 5;
                    bin.g_folded[fold] += *g;
                    let k = bin.h_folded.as_mut().unwrap();
                    k[fold] += *h;
                    bin.counts[fold] += 1;
                });
            }
            None => {
                self.data.iter().for_each(|b| {
                    let bin = b.get().as_mut().unwrap();
                    bin.g_folded = [f32::ZERO; 5];
                    bin.counts = [0; 5];
                });
                index.iter().zip(sorted_grad).for_each(|(i, g)| {
                    let b = self.data.get_unchecked(feature[*i] as usize).get();
                    let bin = b.as_mut().unwrap_unchecked();
                    let fold = i % 5;
                    bin.g_folded[fold] += *g;
                    bin.counts[fold] += 1;
                });
            }
        }
    }

    pub unsafe fn update_cuts(&self, cuts: &[f64]) {
        let cuts_mod = &cuts[..(cuts.len() - 1)];
        self.data.iter().enumerate().for_each(|(i, b)| {
            let bin = b.get().as_mut().unwrap();
            if i == 0 {
                bin.cut_value = f64::NAN;
            } else {
                bin.cut_value = *cuts_mod.get(i - 1).unwrap_or(&f64::NAN);
            }
        });
    }
}

#[derive(Debug)]
pub struct NodeHistogramOwned {
    pub data: Vec<FeatureHistogramOwned>,
}

impl NodeHistogramOwned {
    /// Create an empty histogram matrix.
    pub fn empty_from_cuts(cuts: &JaggedMatrix<f64>, col_index: &[usize], is_const_hess: bool, parallel: bool) -> Self {
        let histograms: Vec<FeatureHistogramOwned> = if parallel {
            col_index
                .par_iter()
                .map(|col| FeatureHistogramOwned::empty_from_cuts(cuts.get_col(*col), is_const_hess))
                .collect()
        } else {
            col_index
                .iter()
                .map(|col| FeatureHistogramOwned::empty_from_cuts(cuts.get_col(*col), is_const_hess))
                .collect()
        };
        NodeHistogramOwned { data: histograms }
    }

    /// Create an empty histogram matrix.
    pub fn empty(max_bin: u16, col_amount: usize, is_const_hess: bool, parallel: bool) -> Self {
        let histograms: Vec<FeatureHistogramOwned> = if parallel {
            (0..col_amount)
                .collect::<Vec<_>>()
                .par_iter()
                .map(|_col| FeatureHistogramOwned::empty(max_bin, is_const_hess))
                .collect()
        } else {
            (0..col_amount)
                .map(|_col| FeatureHistogramOwned::empty(max_bin, is_const_hess))
                .collect()
        };
        NodeHistogramOwned { data: histograms }
    }
}

#[derive(Debug)]
pub struct NodeHistogram<'a> {
    pub data: Vec<FeatureHistogram<'a>>,
}

impl NodeHistogram<'_> {
    pub fn from_owned(hist: &mut NodeHistogramOwned) -> NodeHistogram {
        let histograms = hist
            .data
            .iter_mut()
            .map(|f| FeatureHistogram::new(&mut f.data))
            .collect();
        NodeHistogram { data: histograms }
    }

    /// Calculate the histogram matrix, for a child, given the parent histogram
    /// matrix, and the other child histogram matrix. This should be used
    /// when the node has only two possible splits, left and right.
    pub fn from_parent_child(hist_tree: &[NodeHistogram], root_num: usize, child_num: usize, update_num: usize) {
        unsafe {
            let root_hist = &hist_tree.get_unchecked(root_num).data;
            let child_hist = &hist_tree.get_unchecked(child_num).data;
            let update_hist = &hist_tree.get_unchecked(update_num).data;

            root_hist
                .iter()
                .zip(child_hist.iter())
                .zip(update_hist.iter())
                .for_each(|((root_feat_hist, child_feat_hist), update_feat_hist)| {
                    root_feat_hist
                        .data
                        .iter()
                        .zip(child_feat_hist.data.iter())
                        .zip(update_feat_hist.data.iter())
                        .for_each(|((root_bin, child_bin), update_bin)| {
                            Bin::from_parent_child(root_bin.get(), child_bin.get(), update_bin.get())
                        })
                });
        }
    }

    /// Calculate the histogram matrix for a child, given the parent histogram
    /// and two other child histograms. This should be used with the node has
    /// three possible split paths, right, left, and missing.
    pub fn from_parent_two_children(
        hist_tree: &[NodeHistogram],
        root_num: usize,
        first_num: usize,
        second_num: usize,
        update_num: usize,
    ) {
        unsafe {
            let root_hist = &hist_tree.get_unchecked(root_num).data;
            let first_hist = &hist_tree.get_unchecked(first_num).data;
            let second_hist = &hist_tree.get_unchecked(second_num).data;
            let update_hist = &hist_tree.get_unchecked(update_num).data;

            root_hist
                .iter()
                .zip(first_hist.iter())
                .zip(second_hist.iter())
                .zip(update_hist.iter())
                .for_each(
                    |(((root_feat_hist, first_feat_hist), second_feat_hist), update_feat_hist)| {
                        root_feat_hist
                            .data
                            .iter()
                            .zip(first_feat_hist.data.iter())
                            .zip(second_feat_hist.data.iter())
                            .zip(update_feat_hist.data.iter())
                            .for_each(|(((root_bin, first_bin), second_bin), update_bin)| {
                                Bin::from_parent_two_children(
                                    root_bin.get(),
                                    first_bin.get(),
                                    second_bin.get(),
                                    update_bin.get(),
                                )
                            })
                    },
                );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_cuts(hist: &NodeHistogram, col_index: &[usize], cuts: &JaggedMatrix<f64>, parallel: bool) {
    if parallel {
        hist.data
            .par_iter()
            .zip(col_index.par_iter())
            .for_each(|(h, i)| unsafe { h.update_cuts(cuts.get_col(*i)) })
    } else {
        hist.data
            .iter()
            .zip(col_index.iter())
            .for_each(|(h, i)| unsafe { h.update_cuts(cuts.get_col(*i)) })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_histogram(
    hist: &NodeHistogram,
    start: usize,
    stop: usize,
    data: &Matrix<u16>,
    grad: &[f32],
    hess: Option<&[f32]>,
    index: &[usize],
    col_index: &[usize],
    pool: &ThreadPool,
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

    unsafe {
        if pool.current_num_threads() > 1 {
            pool.scope(|s| {
                for i in 0..hist.data.len() {
                    let h = hist.data.get_unchecked(i);
                    let feature = data.get_col(col_index[i]);
                    s.spawn(|_| {
                        h.update(feature, &sorted_grad, sorted_hess.as_deref(), &index[start..stop]);
                    });
                }
            });
        } else {
            col_index.iter().enumerate().for_each(|(i, col)| {
                println!("i: {:?}, col: {:?}", i, col);
                hist.data.get_unchecked(i).update(
                    data.get_col(*col),
                    &sorted_grad,
                    sorted_hess.as_deref(),
                    &index[start..stop],
                );
            });
        }
    }
}
