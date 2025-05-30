use std::{cell::UnsafeCell, cmp::Ordering};

use crate::data::FloatData;

use serde::{Deserialize, Serialize};

/// Struct to hold the information of a given bin.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Bin {
    pub num: u16,
    pub cut_value: f64,
    pub g_folded: [f32; 5],
    pub h_folded: Option<[f32; 5]>,
    pub counts: [usize; 5],
}

impl Bin {
    pub fn empty_const_hess(num: u16, cut_value: f64) -> Self {
        Bin {
            num,
            cut_value,
            g_folded: [f32::ZERO; 5],
            h_folded: None,
            counts: [0; 5],
        }
    }
    pub fn empty(num: u16, cut_value: f64) -> Self {
        Bin {
            num,
            cut_value,
            g_folded: [f32::ZERO; 5],
            h_folded: Some([f32::ZERO; 5]),
            counts: [0; 5],
        }
    }

    pub fn from_parent_child(root_bin: *mut Bin, child_bin: *mut Bin, update_bin: *mut Bin) {
        let rb = unsafe { root_bin.as_ref().unwrap() };
        let cb = unsafe { child_bin.as_ref().unwrap() };
        let ub = unsafe { update_bin.as_mut().unwrap() };
        for ((z, a), b) in ub.g_folded.iter_mut().zip(rb.g_folded).zip(cb.g_folded) {
            *z = a - b;
        }
        for ((z, a), b) in ub.counts.iter_mut().zip(rb.counts).zip(cb.counts) {
            *z = a - b;
        }

        match rb.h_folded {
            Some(_h_folded) => {
                let h_f_iter = ub.h_folded.as_mut().unwrap().iter_mut();
                for ((zval, aval), bval) in h_f_iter.zip(rb.h_folded.unwrap()).zip(cb.h_folded.unwrap()) {
                    *zval = aval - bval;
                }
            }
            None => {
                ub.h_folded = None;
            }
        };
    }

    pub fn from_parent_two_children(
        root_bin: *mut Bin,
        first_bin: *mut Bin,
        second_bin: *mut Bin,
        update_bin: *mut Bin,
    ) {
        let rb = unsafe { root_bin.as_ref().unwrap() };
        let fb = unsafe { first_bin.as_ref().unwrap() };
        let sb = unsafe { second_bin.as_ref().unwrap() };
        let ub = unsafe { update_bin.as_mut().unwrap() };
        for (((z, a), b), c) in ub
            .g_folded
            .iter_mut()
            .zip(rb.g_folded)
            .zip(fb.g_folded)
            .zip(sb.g_folded)
        {
            *z = a - b - c;
        }
        for (((z, a), b), c) in ub.counts.iter_mut().zip(rb.counts).zip(fb.counts).zip(sb.counts) {
            *z = a - b - c;
        }

        match rb.h_folded {
            Some(_h_folded) => {
                let h_f_iter = ub.h_folded.as_mut().unwrap().iter_mut();
                for (((z, a), b), c) in h_f_iter
                    .zip(rb.h_folded.unwrap())
                    .zip(fb.h_folded.unwrap())
                    .zip(sb.h_folded.unwrap())
                {
                    *z = a - b - c;
                }
            }
            None => {
                ub.h_folded = None;
            }
        };
    }
}

pub fn sort_cat_bins_by_num(histogram: &mut [&UnsafeCell<Bin>]) {
    unsafe {
        histogram.sort_unstable_by_key(|bin| bin.get().as_ref().unwrap().num);
    }
}

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
                let div1: f32 = b1.g_folded.iter().sum::<f32>() / b1.counts.iter().sum::<usize>() as f32;
                let div2: f32 = b2.g_folded.iter().sum::<f32>() / b2.counts.iter().sum::<usize>() as f32;
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
                let div1: f32 = b1.g_folded.iter().sum::<f32>() / b1.h_folded.unwrap().iter().sum::<f32>();
                let div2: f32 = b2.g_folded.iter().sum::<f32>() / b2.h_folded.unwrap().iter().sum::<f32>();
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
        Bin::from_parent_child(
            &mut root_bin as *mut Bin,
            &mut child_bin as *mut Bin,
            &mut update_bin as *mut Bin,
        );
        assert!(update_bin.counts == [1, 2, 3, 4, 5]);
    }
}
