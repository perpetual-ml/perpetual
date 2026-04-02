//! Bin
//!
//! This module defines the `Bin` struct and related utilities for handling binned data.
//! Each bin stores statistics (gradient and hessian) for a specific feature split point.
use std::{cell::UnsafeCell, cmp::Ordering};

use crate::data::FloatData;

use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub struct Bin {
    /// The bin number or index.
    pub num: u16,
    /// The split value for this bin.
    pub cut_value: f64,
    /// The folded gradient statistics.
    pub g_folded: [f32; 5],
    /// The folded hessian statistics.
    pub h_folded: [f32; 5],
    /// The folded count statistics.
    pub counts: [u32; 5],
}

impl Bin {
    /// Create an empty bin with constant hessian.
    pub fn empty_const_hess(num: u16, cut_value: f64) -> Self {
        Bin {
            num,
            cut_value,
            g_folded: [f32::ZERO; 5],
            h_folded: [f32::ZERO; 5],
            counts: [0; 5],
        }
    }
    /// Create an empty bin.
    pub fn empty(num: u16, cut_value: f64) -> Self {
        Bin {
            num,
            cut_value,
            g_folded: [f32::ZERO; 5],
            h_folded: [f32::ZERO; 5],
            counts: [0; 5],
        }
    }

    /// # Safety
    /// Updates a `Bin` by subtracting the values of another `Bin` from a parent `Bin`.
    pub unsafe fn from_parent_child(root_bin: *mut Bin, child_bin: *mut Bin, update_bin: *mut Bin) {
        unsafe {
            let rb = root_bin.as_ref().unwrap_unchecked();
            let cb = child_bin.as_ref().unwrap_unchecked();
            let ub = update_bin.as_mut().unwrap_unchecked();
            // Fused loop: update g_folded, h_folded, counts in a single pass per fold
            for j in 0..5 {
                *ub.g_folded.get_unchecked_mut(j) = rb.g_folded.get_unchecked(j) - cb.g_folded.get_unchecked(j);
                *ub.h_folded.get_unchecked_mut(j) = rb.h_folded.get_unchecked(j) - cb.h_folded.get_unchecked(j);
                *ub.counts.get_unchecked_mut(j) = rb.counts.get_unchecked(j) - cb.counts.get_unchecked(j);
            }
        }
    }

    /// Updates a `Bin` by subtracting the values of two other `Bin`s from a parent `Bin`.
    /// This operation is performed on the `g_folded`, `counts`, and `h_folded` fields.
    ///
    /// # Arguments
    ///
    /// * `root_bin`: A mutable raw pointer to the parent `Bin`.
    /// * `first_bin`: A mutable raw pointer to the first child `Bin`.
    /// * `second_bin`: A mutable raw pointer to the second child `Bin`.
    /// * `update_bin`: A mutable raw pointer to the `Bin` that will be updated.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers (`*mut Bin`).
    /// The caller must ensure the following conditions are met to avoid undefined behavior:
    ///
    /// * All pointers (`root_bin`, `first_bin`, `second_bin`, `update_bin`) must be
    ///   **valid** and **non-null**.
    /// * The memory pointed to by each of these pointers must be **valid** for reads
    ///   and writes (for `update_bin`).
    /// * The data structures (`g_folded`, `counts`, `h_folded`) within the `Bin`s
    ///   must be in a valid state for the operations being performed.
    pub unsafe fn from_parent_two_children(
        root_bin: *mut Bin,
        first_bin: *mut Bin,
        second_bin: *mut Bin,
        update_bin: *mut Bin,
    ) {
        unsafe {
            let rb = root_bin.as_ref().unwrap_unchecked();
            let fb = first_bin.as_ref().unwrap_unchecked();
            let sb = second_bin.as_ref().unwrap_unchecked();
            let ub = update_bin.as_mut().unwrap_unchecked();
            for j in 0..5 {
                *ub.g_folded.get_unchecked_mut(j) =
                    rb.g_folded.get_unchecked(j) - fb.g_folded.get_unchecked(j) - sb.g_folded.get_unchecked(j);
                *ub.h_folded.get_unchecked_mut(j) =
                    rb.h_folded.get_unchecked(j) - fb.h_folded.get_unchecked(j) - sb.h_folded.get_unchecked(j);
                *ub.counts.get_unchecked_mut(j) =
                    rb.counts.get_unchecked(j) - fb.counts.get_unchecked(j) - sb.counts.get_unchecked(j);
            }
        }
    }
}

/// Sort categorical bins by their bin number.
pub fn sort_cat_bins_by_num(histogram: &mut [&UnsafeCell<Bin>]) {
    unsafe {
        histogram.sort_unstable_by_key(|bin| bin.get().as_ref().unwrap().num);
    }
}

#[derive(Copy, Clone)]
enum CatBinSortMode {
    OrderedStat,
    PosteriorMean,
    SupportLift,
}

#[inline]
fn cat_sort_priors(histogram: &[&UnsafeCell<Bin>], is_const_hess: bool) -> (f32, f32, f32) {
    unsafe {
        let mut prior_strength = 1.0_f32;
        let mut prior_numerator = 0.0_f32;
        let mut prior_denom = 0.0_f32;
        let mut used_bins = 0_u32;
        for bin in histogram.iter() {
            let current = bin.get().as_ref().unwrap();
            if current.num == 0 {
                continue;
            }
            let denom = if is_const_hess {
                current.counts.iter().sum::<u32>() as f32
            } else {
                current.h_folded.iter().sum::<f32>()
            };
            if denom > 0.0 {
                prior_strength += denom;
                prior_numerator += current.g_folded.iter().sum::<f32>();
                prior_denom += denom;
                used_bins += 1;
            }
        }
        if used_bins > 0 {
            prior_strength /= used_bins as f32;
        }
        (prior_numerator, prior_denom, prior_strength)
    }
}

#[inline]
fn cat_bin_score(
    bin: &Bin,
    is_const_hess: bool,
    prior_numerator: f32,
    prior_denom: f32,
    prior_strength: f32,
    l2_regularization: f32,
) -> f32 {
    let denominators = if is_const_hess {
        bin.counts.map(|count| count as f32)
    } else {
        bin.h_folded
    };
    let total_numerator = bin.g_folded.iter().sum::<f32>();
    let total_denom = denominators.iter().sum::<f32>();
    if total_denom <= 0.0 {
        return 0.0;
    }

    let prior_mean = if prior_denom > 0.0 {
        prior_numerator / (prior_denom + l2_regularization)
    } else {
        0.0
    };

    let mut ordered_scores = [0.0_f32; 5];
    let mut holdout_alignment = [0.0_f32; 5];
    let mut used_folds = 0_usize;

    for (&fold_denom, &fold_gradient) in denominators.iter().zip(bin.g_folded.iter()) {
        let train_denom = total_denom - fold_denom;
        if train_denom <= 0.0 {
            continue;
        }

        let train_numerator = total_numerator - fold_gradient;
        let ordered_score =
            (train_numerator + prior_strength * prior_mean) / (train_denom + prior_strength + l2_regularization);
        let holdout_score = if fold_denom > 0.0 {
            fold_gradient / (fold_denom + l2_regularization)
        } else {
            prior_mean
        };
        let alignment = 1.0 / (1.0 + (holdout_score - ordered_score).abs() / (ordered_score.abs() + 1e-3));

        ordered_scores[used_folds] = ordered_score;
        holdout_alignment[used_folds] = alignment;
        used_folds += 1;
    }

    if used_folds == 0 {
        return 0.0;
    }

    let mean_score = ordered_scores[..used_folds].iter().sum::<f32>() / used_folds as f32;
    let variance = ordered_scores[..used_folds]
        .iter()
        .map(|score| {
            let diff = *score - mean_score;
            diff * diff
        })
        .sum::<f32>()
        / used_folds as f32;
    let stability = 1.0 / (1.0 + variance.sqrt() / (mean_score.abs() + 1e-6));
    let alignment = holdout_alignment[..used_folds].iter().sum::<f32>() / used_folds as f32;
    let support_shrinkage = total_denom / (total_denom + 0.35 * prior_strength.max(1.0));

    mean_score * (0.55 * stability + 0.45 * alignment) * support_shrinkage
}

#[inline]
fn cat_bin_posterior_mean(
    bin: &Bin,
    is_const_hess: bool,
    prior_numerator: f32,
    prior_denom: f32,
    prior_strength: f32,
    l2_regularization: f32,
) -> f32 {
    let total_numerator = bin.g_folded.iter().sum::<f32>();
    let total_denom = if is_const_hess {
        bin.counts.iter().sum::<u32>() as f32
    } else {
        bin.h_folded.iter().sum::<f32>()
    };
    if total_denom <= 0.0 {
        return 0.0;
    }

    let prior_mean = if prior_denom > 0.0 {
        prior_numerator / (prior_denom + l2_regularization)
    } else {
        0.0
    };
    (total_numerator + prior_strength * prior_mean) / (total_denom + prior_strength + l2_regularization)
}

#[inline]
fn cat_bin_support_lift(
    bin: &Bin,
    is_const_hess: bool,
    prior_numerator: f32,
    prior_denom: f32,
    prior_strength: f32,
    l2_regularization: f32,
) -> f32 {
    let posterior_mean = cat_bin_posterior_mean(
        bin,
        is_const_hess,
        prior_numerator,
        prior_denom,
        prior_strength,
        l2_regularization,
    );
    let total_denom = if is_const_hess {
        bin.counts.iter().sum::<u32>() as f32
    } else {
        bin.h_folded.iter().sum::<f32>()
    };
    if total_denom <= 0.0 {
        return 0.0;
    }

    let prior_mean = if prior_denom > 0.0 {
        prior_numerator / (prior_denom + l2_regularization)
    } else {
        0.0
    };
    let support = total_denom.sqrt() / (total_denom.sqrt() + prior_strength.max(1.0).sqrt());
    (posterior_mean - prior_mean) * support
}

#[inline]
fn cat_bin_sort_score(
    bin: &Bin,
    is_const_hess: bool,
    prior_numerator: f32,
    prior_denom: f32,
    prior_strength: f32,
    l2_regularization: f32,
    mode: CatBinSortMode,
) -> f32 {
    match mode {
        CatBinSortMode::OrderedStat => cat_bin_score(
            bin,
            is_const_hess,
            prior_numerator,
            prior_denom,
            prior_strength,
            l2_regularization,
        ),
        CatBinSortMode::PosteriorMean => cat_bin_posterior_mean(
            bin,
            is_const_hess,
            prior_numerator,
            prior_denom,
            prior_strength,
            l2_regularization,
        ),
        CatBinSortMode::SupportLift => cat_bin_support_lift(
            bin,
            is_const_hess,
            prior_numerator,
            prior_denom,
            prior_strength,
            l2_regularization,
        ),
    }
}

#[inline]
fn preferred_single_categorical_sort_mode(
    histogram: &[&UnsafeCell<Bin>],
    is_const_hess: bool,
    prior_strength: f32,
) -> CatBinSortMode {
    let mut active_bins = 0_usize;
    let mut low_support_bins = 0_usize;
    let low_support_cutoff = 1.5 * prior_strength.max(1.0);

    unsafe {
        for bin in histogram.iter() {
            let current = bin.get().as_ref().unwrap();
            if current.num == 0 {
                continue;
            }
            let denom = if is_const_hess {
                current.counts.iter().sum::<u32>() as f32
            } else {
                current.h_folded.iter().sum::<f32>()
            };
            if denom <= 0.0 {
                continue;
            }
            active_bins += 1;
            if denom <= low_support_cutoff {
                low_support_bins += 1;
            }
        }
    }

    if active_bins < 48 {
        return CatBinSortMode::OrderedStat;
    }

    let low_support_share = low_support_bins as f32 / active_bins as f32;
    if low_support_share >= 0.7 {
        CatBinSortMode::SupportLift
    } else if low_support_share >= 0.45 {
        CatBinSortMode::PosteriorMean
    } else {
        CatBinSortMode::OrderedStat
    }
}

pub fn categorical_histogram_orders<'a>(
    histogram: &[&'a UnsafeCell<Bin>],
    is_const_hess: bool,
    l2_regularization: f32,
) -> Vec<Vec<&'a UnsafeCell<Bin>>> {
    let (prior_numerator, prior_denom, prior_strength) = cat_sort_priors(histogram, is_const_hess);
    let mut orders: Vec<Vec<&'a UnsafeCell<Bin>>> = Vec::new();

    for mode in [
        CatBinSortMode::OrderedStat,
        CatBinSortMode::PosteriorMean,
        CatBinSortMode::SupportLift,
    ] {
        let mut order = histogram.to_vec();
        unsafe {
            order.sort_unstable_by(|bin1, bin2| {
                let b1 = bin1.get().as_ref().unwrap();
                let b2 = bin2.get().as_ref().unwrap();
                if b1.num == 0 {
                    return Ordering::Less;
                } else if b2.num == 0 {
                    return Ordering::Greater;
                }

                let score1 = cat_bin_sort_score(
                    b1,
                    is_const_hess,
                    prior_numerator,
                    prior_denom,
                    prior_strength,
                    l2_regularization,
                    mode,
                );
                let score2 = cat_bin_sort_score(
                    b2,
                    is_const_hess,
                    prior_numerator,
                    prior_denom,
                    prior_strength,
                    l2_regularization,
                    mode,
                );
                score2.total_cmp(&score1)
            });
        }
        let is_duplicate = orders.iter().any(|existing: &Vec<&'a UnsafeCell<Bin>>| {
            existing
                .iter()
                .map(|bin| unsafe { bin.get().as_ref().unwrap().num })
                .eq(order.iter().map(|bin| unsafe { bin.get().as_ref().unwrap().num }))
        });
        if !is_duplicate {
            orders.push(order);
        }
    }

    orders
}

/// Sort categorical bins by their statistics (gradient/hessian or gradient/count).
pub fn sort_cat_bins_by_stat(histogram: &mut [&UnsafeCell<Bin>], is_const_hess: bool, l2_regularization: f32) {
    let (prior_numerator, prior_denom, prior_strength) = cat_sort_priors(histogram, is_const_hess);
    let sort_mode = preferred_single_categorical_sort_mode(histogram, is_const_hess, prior_strength);
    unsafe {
        histogram.sort_unstable_by(|bin1, bin2| {
            let b1 = bin1.get().as_ref().unwrap();
            let b2 = bin2.get().as_ref().unwrap();
            if b1.num == 0 {
                return Ordering::Less;
            } else if b2.num == 0 {
                return Ordering::Greater;
            }

            let score1 = cat_bin_sort_score(
                b1,
                is_const_hess,
                prior_numerator,
                prior_denom,
                prior_strength,
                l2_regularization,
                sort_mode,
            );
            let score2 = cat_bin_sort_score(
                b2,
                is_const_hess,
                prior_numerator,
                prior_denom,
                prior_strength,
                l2_regularization,
                sort_mode,
            );
            score2.total_cmp(&score1)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bin() {
        let mut root_bin = Bin::empty_const_hess(0, 0.0);
        root_bin.counts = [10, 10, 10, 10, 10];
        let mut child_bin = Bin::empty_const_hess(1, 0.0);
        child_bin.counts = [9, 8, 7, 6, 5];
        let mut update_bin = Bin::empty_const_hess(2, 0.0);
        unsafe {
            Bin::from_parent_child(
                &mut root_bin as *mut Bin,
                &mut child_bin as *mut Bin,
                &mut update_bin as *mut Bin,
            )
        };
        assert!(update_bin.counts == [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_from_parent_two_children() {
        let mut root = Bin::empty(0, 0.0);
        root.g_folded = [10.0, 20.0, 30.0, 40.0, 50.0];
        root.h_folded = [5.0, 5.0, 5.0, 5.0, 5.0];
        root.counts = [100, 100, 100, 100, 100];
        let mut c1 = Bin::empty(1, 0.0);
        c1.g_folded = [3.0, 6.0, 9.0, 12.0, 15.0];
        c1.h_folded = [1.0, 1.0, 1.0, 1.0, 1.0];
        c1.counts = [30, 30, 30, 30, 30];
        let mut c2 = Bin::empty(2, 0.0);
        c2.g_folded = [2.0, 4.0, 6.0, 8.0, 10.0];
        c2.h_folded = [1.0, 2.0, 1.0, 2.0, 1.0];
        c2.counts = [20, 20, 20, 20, 20];
        let mut update = Bin::empty(3, 0.0);
        unsafe {
            Bin::from_parent_two_children(
                &mut root as *mut Bin,
                &mut c1 as *mut Bin,
                &mut c2 as *mut Bin,
                &mut update as *mut Bin,
            );
        }
        assert!((update.g_folded[0] - 5.0).abs() < 1e-6);
        assert!((update.h_folded[1] - 2.0).abs() < 1e-6);
        assert_eq!(update.counts[0], 50);
    }

    #[test]
    fn test_sort_cat_bins_by_num() {
        let b1 = Bin::empty_const_hess(3, 0.0);
        let b2 = Bin::empty_const_hess(1, 0.0);
        let b3 = Bin::empty_const_hess(2, 0.0);
        let c1 = UnsafeCell::new(b1);
        let c2 = UnsafeCell::new(b2);
        let c3 = UnsafeCell::new(b3);
        let mut hist: Vec<&UnsafeCell<Bin>> = vec![&c1, &c2, &c3];
        sort_cat_bins_by_num(&mut hist);
        unsafe {
            assert_eq!(hist[0].get().as_ref().unwrap().num, 1);
            assert_eq!(hist[1].get().as_ref().unwrap().num, 2);
            assert_eq!(hist[2].get().as_ref().unwrap().num, 3);
        }
    }

    #[test]
    fn test_sort_cat_bins_by_stat_const_hess() {
        let b0 = Bin::empty_const_hess(0, 0.0); // num=0, always first
        let mut b1 = Bin::empty_const_hess(1, 0.0);
        b1.g_folded = [1.0; 5];
        b1.counts = [10; 5]; // g/c = 0.1
        let mut b2 = Bin::empty_const_hess(2, 0.0);
        b2.g_folded = [5.0; 5];
        b2.counts = [10; 5]; // g/c = 0.5
        let c0 = UnsafeCell::new(b0);
        let c1 = UnsafeCell::new(b1);
        let c2 = UnsafeCell::new(b2);
        let mut hist: Vec<&UnsafeCell<Bin>> = vec![&c2, &c0, &c1];
        sort_cat_bins_by_stat(&mut hist, true, 1.0);
        unsafe {
            assert_eq!(hist[0].get().as_ref().unwrap().num, 0); // num=0 always first
        }
    }

    #[test]
    fn test_sort_cat_bins_by_stat_non_const() {
        let b0 = Bin::empty(0, 0.0); // num=0, always first
        let mut b1 = Bin::empty(1, 0.0);
        b1.g_folded = [1.0; 5];
        b1.h_folded = [10.0; 5]; // g/h = 0.1
        let mut b2 = Bin::empty(2, 0.0);
        b2.g_folded = [5.0; 5];
        b2.h_folded = [10.0; 5]; // g/h = 0.5
        let c0 = UnsafeCell::new(b0);
        let c1 = UnsafeCell::new(b1);
        let c2 = UnsafeCell::new(b2);
        let mut hist: Vec<&UnsafeCell<Bin>> = vec![&c2, &c0, &c1];
        sort_cat_bins_by_stat(&mut hist, false, 1.0);
        unsafe {
            assert_eq!(hist[0].get().as_ref().unwrap().num, 0); // num=0 always first
        }
    }

    #[test]
    fn test_cat_bin_score_shrinks_rare_categories() {
        let mut rare = Bin::empty_const_hess(1, 0.0);
        rare.g_folded = [2.0; 5];
        rare.counts = [1; 5];

        let mut dense = Bin::empty_const_hess(2, 0.0);
        dense.g_folded = [10.0; 5];
        dense.counts = [20; 5];

        let prior_strength = (rare.counts.iter().sum::<u32>() + dense.counts.iter().sum::<u32>()) as f32 / 2.0;
        let prior_numerator = rare.g_folded.iter().sum::<f32>() + dense.g_folded.iter().sum::<f32>();
        let prior_denom = (rare.counts.iter().sum::<u32>() + dense.counts.iter().sum::<u32>()) as f32;

        assert!(
            cat_bin_score(&dense, true, prior_numerator, prior_denom, prior_strength, 1.0)
                > cat_bin_score(&rare, true, prior_numerator, prior_denom, prior_strength, 1.0)
        );
    }

    #[test]
    fn test_cat_bin_score_penalizes_fold_instability() {
        let mut stable = Bin::empty(1, 0.0);
        stable.g_folded = [2.0; 5];
        stable.h_folded = [10.0; 5];

        let mut unstable = Bin::empty(2, 0.0);
        unstable.g_folded = [10.0, -6.0, 10.0, -6.0, 2.0];
        unstable.h_folded = [10.0; 5];

        let prior_numerator = stable.g_folded.iter().sum::<f32>() + unstable.g_folded.iter().sum::<f32>();
        let prior_denom = stable.h_folded.iter().sum::<f32>() + unstable.h_folded.iter().sum::<f32>();

        assert!(
            cat_bin_score(&stable, false, prior_numerator, prior_denom, 10.0, 1.0)
                > cat_bin_score(&unstable, false, prior_numerator, prior_denom, 10.0, 1.0)
        );
    }
}
