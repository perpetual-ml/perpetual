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

/// Sort categorical bins by their statistics (gradient/hessian or gradient/count).
pub fn sort_cat_bins_by_stat(histogram: &mut [&UnsafeCell<Bin>], is_const_hess: bool) {
    unsafe {
        if is_const_hess {
            histogram.sort_unstable_by(|bin1, bin2| {
                let b1 = bin1.get().as_ref().unwrap();
                let b2 = bin2.get().as_ref().unwrap();
                if b1.num == 0 {
                    return Ordering::Less;
                } else if b2.num == 0 {
                    return Ordering::Greater;
                }
                let div1: f32 = b1.g_folded.iter().sum::<f32>() / b1.counts.iter().sum::<u32>() as f32;
                let div2: f32 = b2.g_folded.iter().sum::<f32>() / b2.counts.iter().sum::<u32>() as f32;
                div2.partial_cmp(&div1).unwrap_or(Ordering::Less)
            });
        } else {
            histogram.sort_unstable_by(|bin1, bin2| {
                let b1 = bin1.get().as_ref().unwrap();
                let b2 = bin2.get().as_ref().unwrap();
                if b1.num == 0 {
                    return Ordering::Less;
                } else if b2.num == 0 {
                    return Ordering::Greater;
                }
                let div1: f32 = b1.g_folded.iter().sum::<f32>() / b1.h_folded.iter().sum::<f32>();
                let div2: f32 = b2.g_folded.iter().sum::<f32>() / b2.h_folded.iter().sum::<f32>();
                div2.partial_cmp(&div1).unwrap_or(Ordering::Less)
            });
        }
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
}
