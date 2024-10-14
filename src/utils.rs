use crate::constants::HESSIAN_EPS;
use crate::constraints::Constraint;
use crate::data::FloatData;
use crate::errors::PerpetualError;
use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};
use std::convert::TryInto;

/// Create a string of all available items.
pub fn items_to_strings(items: Vec<&str>) -> String {
    let mut s = String::new();
    for i in items {
        s.push_str(i);
        s.push_str(&String::from(", "));
    }
    s
}

pub fn fmt_vec_output<T: FloatData<T>>(v: &[T]) -> String {
    let mut res = String::new();
    if let Some(last) = v.len().checked_sub(1) {
        if last == 0 {
            return format!("{:.4}", v[0]);
        }
        for n in &v[..last] {
            res.push_str(format!("{:.4}", n).as_str());
            res.push_str(", ");
        }
        res.push_str(format!("{:.4}", &v[last]).as_str());
    }
    res
}

// Validation
pub fn validate_positive_float_parameter<T: FloatData<T>>(value: T, parameter: &str) -> Result<(), PerpetualError> {
    validate_float_parameter(value, T::ZERO, T::INFINITY, parameter)
}
pub fn validate_float_parameter<T: FloatData<T>>(
    value: T,
    min: T,
    max: T,
    parameter: &str,
) -> Result<(), PerpetualError> {
    let mut msg = String::new();
    if value.is_nan() || value < min || max < value {
        msg.push_str(&value.to_string());
        let ex_msg = format!("real value within rang {} and {}", min, max);
        Err(PerpetualError::InvalidParameter(
            parameter.to_string(),
            ex_msg,
            value.to_string(),
        ))
    } else {
        Ok(())
    }
}

/*
macro_rules! validate_positive_float_field {
    ($var: expr) => {
        let var_name = stringify!($var).split(".").nth(1).unwrap();
        crate::utils::validate_positive_float_parameter($var, var_name)?
    };
}

pub(crate) use validate_positive_float_field;
*/

/// Calculate if a value is missing.
#[inline]
pub fn is_missing(value: &f64, missing: &f64) -> bool {
    if missing.is_nan() {
        value.is_nan()
    } else if value.is_nan() {
        panic!("Missing value is {}, however NAN value found in data.", missing)
    } else {
        value == missing
    }
}

/// Calculate the constraint weight given bounds
/// and a constraint.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn constrained_weight(
    gradient_sum: f32,
    hessian_sum: f32,
    lower_bound: f32,
    upper_bound: f32,
    constraint: Option<&Constraint>,
) -> f32 {
    let weight = weight(gradient_sum, hessian_sum);
    match constraint {
        None | Some(Constraint::Unconstrained) => weight,
        _ => {
            if weight > upper_bound {
                upper_bound
            } else if weight < lower_bound {
                lower_bound
            } else {
                weight
            }
        }
    }
}

/// Calculate the constraint weight given bounds
/// and a constraint.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn constrained_weight_const_hess(
    gradient_sum: f32,
    count_sum: usize,
    lower_bound: f32,
    upper_bound: f32,
    constraint: Option<&Constraint>,
) -> f32 {
    let weight = weight_const_hess(gradient_sum, count_sum);
    match constraint {
        None | Some(Constraint::Unconstrained) => weight,
        _ => {
            if weight > upper_bound {
                upper_bound
            } else if weight < lower_bound {
                lower_bound
            } else {
                weight
            }
        }
    }
}

/// Test if v is contained within the range i and j
#[inline]
pub fn between(i: f32, j: f32, v: f32) -> bool {
    if i > j {
        (i >= v) && (v >= j)
    } else {
        (i <= v) && (v <= j)
    }
}

#[inline]
pub fn bound_to_parent(parent_weight: f32, left_weight: f32, right_weight: f32) -> (f32, f32) {
    if between(left_weight, right_weight, parent_weight) {
        (left_weight, right_weight)
    } else {
        // If we are here, we know, the parent weight is above or bellow
        // the right and left weights range, because of the between check.
        if left_weight > right_weight {
            // Here is what it looks like on the number line if we are here
            // right...left
            // Is the parent above the range?
            // i.e. right...left...parent?
            if left_weight < parent_weight {
                (parent_weight, right_weight)
            } else {
                // Otherwise if we are here, it must be outside of the range on the other side..
                // i.e. parent...right...left
                // In which case make parent equal right.
                (left_weight, parent_weight)
            }
        } else {
            // Here is what the number line looks like at this point...
            // left_weight..right_weight
            // Is the parent above the range?
            // i.e. left...right...parent?
            if right_weight < parent_weight {
                // In which case set right equal to parent.
                (left_weight, parent_weight)
            } else {
                // Is the parent bellow the range?
                // i.e. parent...left...right...
                // In which case set the left equal to the parent.
                (parent_weight, right_weight)
            }
        }
    }
}

/// Convert Log odds to probability
#[inline]
pub fn odds(v: f64) -> f64 {
    1. / (1. + (-v).exp())
}

/// Calculate the weight of a given node, given the sum
/// of the gradients, and the hessians in a node.
#[inline]
pub fn weight(gradient_sum: f32, hessian_sum: f32) -> f32 {
    -gradient_sum / (hessian_sum + 1.0)
}
#[inline]
pub fn weight_const_hess(gradient_sum: f32, count_sum: usize) -> f32 {
    -gradient_sum / (count_sum as f32)
}

/// Calculate the gain given the gradient and hessian of the node.
#[inline]
pub fn gain(gradient_sum: f32, hessian_sum: f32) -> f32 {
    (gradient_sum * gradient_sum) / (hessian_sum + HESSIAN_EPS) // no -0.5 multiplier term!
}
#[inline]
pub fn gain_const_hess(gradient_sum: f32, count_sum: usize) -> f32 {
    (gradient_sum * gradient_sum) / (count_sum as f32) // no -0.5 multiplier term!
}

/// Calculate the gain of a split given a specific weight value.
/// This is for if the weight has to be constrained, for example for
/// monotonicity constraints.
#[inline]
pub fn gain_given_weight(gradient_sum: f32, hessian_sum: f32, weight: f32) -> f32 {
    -(2.0 * gradient_sum * weight + (hessian_sum + HESSIAN_EPS) * (weight * weight))
}
#[inline]
pub fn gain_given_weight_const_hess(gradient_sum: f32, counts: usize, weight: f32) -> f32 {
    -(2.0 * gradient_sum * weight + (counts as f32) * (weight * weight))
}

/// Cull gain, if it does not conform to constraints.
#[inline]
pub fn cull_gain(gain: f32, left_weight: f32, right_weight: f32, constraint: Option<&Constraint>) -> f32 {
    match constraint {
        None | Some(Constraint::Unconstrained) => gain,
        Some(Constraint::Negative) => {
            if left_weight <= right_weight {
                f32::NEG_INFINITY
            } else {
                gain
            }
        }
        Some(Constraint::Positive) => {
            if left_weight >= right_weight {
                f32::NEG_INFINITY
            } else {
                gain
            }
        }
    }
}

const LANES: usize = 16;

/// Fast summation, ends up being roughly 8 to 10 times faster
/// than values.iter().copied().sum().
/// Shamelessly stolen from https://stackoverflow.com/a/67191480
#[inline]
pub fn fast_sum<T: FloatData<T>>(values: &[T]) -> T {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let sum = chunks.fold([T::ZERO; LANES], |mut acc, chunk| {
        let chunk: [T; LANES] = chunk.try_into().unwrap();
        for i in 0..LANES {
            acc[i] += chunk[i];
        }
        acc
    });

    let remainder: T = remainder.iter().copied().sum();

    let mut reduced = T::ZERO;
    for s in sum.iter().take(LANES) {
        reduced += *s;
    }
    reduced + remainder
}

/// Fast summation, but using f64 as the internal representation so that
/// we don't have issues with the precision.
/// This way, we can still work with f32 values, but get the correct sum
/// value.
#[inline]
pub fn fast_f64_sum(values: &[f32]) -> f32 {
    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let sum = chunks.fold([f64::ZERO; LANES], |mut acc, chunk| {
        let chunk: [f32; LANES] = chunk.try_into().unwrap();
        for i in 0..LANES {
            acc[i] += f64::from(chunk[i]);
        }
        acc
    });

    let remainder: f64 = remainder.iter().fold(f64::ZERO, |acc, b| acc + f64::from(*b));

    let mut reduced: f64 = 0.;
    for s in sum.iter().take(LANES) {
        reduced += *s;
    }
    (reduced + remainder) as f32
}

pub fn naive_sum<T: FloatData<T>>(values: &[T]) -> T {
    values.iter().copied().sum()
}

/// Naive weighted percentiles calculation.
///
/// Currently this function does not support missing values.
///   
/// * `v` - A Vector of which to find percentiles for.
/// * `sample_weight` - Sample weights for the instances of the vector.
/// * `percentiles` - Percentiles to look for in the data. This should be
///     values from 0 to 1, and in sorted order.
pub fn percentiles<T>(v: &[T], sample_weight: &[T], percentiles: &[T]) -> Vec<T>
where
    T: FloatData<T>,
{
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());

    // Setup percentiles
    let mut pcts = VecDeque::from_iter(percentiles.iter());
    let mut current_pct = *pcts.pop_front().expect("No percentiles were provided");

    // Prepare a vector to put the percentiles in...
    let mut p = Vec::new();
    let mut cuml_pct = T::ZERO;
    let mut current_value = v[idx[0]];
    let total_values = fast_sum(sample_weight);

    for i in idx.iter() {
        if current_value != v[*i] {
            current_value = v[*i];
        }
        cuml_pct += sample_weight[*i] / total_values;
        if (current_pct == T::ZERO) || (cuml_pct >= current_pct) {
            // We loop here, because the same number might be a valid
            // value to make the percentile several times.
            while cuml_pct >= current_pct {
                p.push(current_value);
                match pcts.pop_front() {
                    Some(p_) => current_pct = *p_,
                    None => return p,
                }
            }
        } else if current_pct == T::ONE {
            if let Some(i_) = idx.last() {
                p.push(v[*i_]);
                break;
            }
        }
    }
    p
}

// Return the index of the first value in a slice that
// is less another number. This will return the first index for
// missing values.
/// Return the index of the first value in a sorted
/// vector that is greater than a provided value.
///
/// * `x` - The sorted slice of values.
/// * `v` - The value used to calculate the first
///   value larger than it.
#[inline]
pub fn map_bin<T: FloatData<T>>(x: &[T], v: &T, missing: &T) -> Option<u16> {
    if v.is_nan() || (v == missing) {
        return Some(0);
    }
    let mut low = 0;
    let mut high = x.len();
    while low != high {
        let mid = (low + high) / 2;
        // This will always be false for NaNs.
        // This it will force us to the bottom,
        // and thus Zero.
        if x[mid] <= *v {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    u16::try_from(low).ok()
}

/// Provided a list of index values, pivot those values
/// around a specific split value so all of the values less
/// than the split value are on one side, and then all of the
/// values greater than or equal to the split value are above.
///
/// * `index` - The index values to sort.
/// * `feature` - The feature vector to use to sort the index by.
/// * `split_value` - the split value to use to pivot on.
/// * `missing_right` - Should missing values go to the left, or
///    to the right of the split value.
#[inline]
pub fn pivot_on_split(
    start: usize,
    stop: usize,
    idx: &mut [usize],
    grad: &mut [f32],
    hess: &mut [f32],
    feature: &[u16],
    split_value: u16,
    missing_right: bool,
    left_cats: &HashSet<usize>,
) -> usize {
    let index = &mut idx[start..stop];
    let g = &mut grad[start..stop];
    let h = &mut hess[start..stop];

    let length = index.len();
    let mut last_idx = length - 1;
    let mut rv = None;

    for i in 0..length {
        loop {
            match missing_compare(&split_value, feature[index[i]], missing_right, left_cats) {
                Ordering::Less | Ordering::Equal => {
                    if last_idx <= i {
                        rv = Some(i);
                        break;
                    }
                    index.swap(i, last_idx);
                    g.swap(i, last_idx);
                    h.swap(i, last_idx);
                    if last_idx == 0 {
                        rv = Some(0);
                        break;
                    }
                    last_idx -= 1;
                }
                Ordering::Greater => break,
            }
        }
        if i >= last_idx {
            break;
        }
    }
    match rv {
        Some(r) => r,
        None => last_idx + 1,
    }
}

#[inline]
pub fn pivot_on_split_const_hess(
    start: usize,
    stop: usize,
    idx: &mut [usize],
    grad: &mut [f32],
    feature: &[u16],
    split_value: u16,
    missing_right: bool,
    left_cats: &HashSet<usize>,
) -> usize {
    let index = &mut idx[start..stop];
    let g = &mut grad[start..stop];

    let length = index.len();
    let mut last_idx = length - 1;
    let mut rv = None;

    for i in 0..length {
        loop {
            match missing_compare(&split_value, feature[index[i]], missing_right, left_cats) {
                Ordering::Less | Ordering::Equal => {
                    if last_idx <= i {
                        rv = Some(i);
                        break;
                    }
                    index.swap(i, last_idx);
                    g.swap(i, last_idx);
                    if last_idx == 0 {
                        rv = Some(0);
                        break;
                    }
                    last_idx -= 1;
                }
                Ordering::Greater => break,
            }
        }
        if i >= last_idx {
            break;
        }
    }
    match rv {
        Some(r) => r,
        None => last_idx + 1,
    }
}

/// Provided a list of index values, pivot those values
/// around a specific split value so all of the values less
/// than the split value are on one side, and then all of the
/// values greater than or equal to the split value are above.
/// Missing values, will be pushed to the bottom, a value of
/// zero is missing in this case.
/// Returns a tuple, the first is the first non-missing value
/// index, the second is the first value that is greater than
/// our provided split value.
///
/// WARNING!!! Currently, this function fails, if all the values are
/// missing...
///
/// * `index` - The index values to sort.
/// * `feature` - The feature vector to use to sort the index by.
/// * `split_value` - the split value to use to pivot on.
#[inline]
pub fn pivot_on_split_exclude_missing(
    start: usize,
    stop: usize,
    idx: &mut [usize],
    grad: &mut [f32],
    hess: &mut [f32],
    feature: &[u16],
    split_value: u16,
    left_cats: &HashSet<usize>,
) -> (usize, usize) {
    let index = &mut idx[start..stop];
    let gr = &mut grad[start..stop];
    let hs = &mut hess[start..stop];
    // I think we can do this in O(n) time...
    let mut low = 0;
    let mut high = index.len() - 1;
    // The index of the first value, that is not
    // missing.
    let mut missing = 0;
    let max_idx = high;
    while low < high {
        // Go until we find a low value that needs to
        // be swapped, this will be the first value
        // that our split value is less or equal to.
        while low < max_idx {
            let l = feature[index[low]];
            if l == 0 {
                index.swap(missing, low);
                gr.swap(missing, low);
                hs.swap(missing, low);
                missing += 1;
            }
            match exclude_missing_compare(&split_value, l, left_cats) {
                Ordering::Less | Ordering::Equal => break,
                Ordering::Greater => low += 1,
            }
        }
        while high > low {
            let h = feature[index[high]];
            // If this is missing, we need to
            // swap this value with missing, and
            // then that value with low.
            if h == 0 {
                index.swap(missing, high);
                gr.swap(missing, high);
                hs.swap(missing, high);
                missing += 1;
                // Low must be at least equal to
                // missing. Otherwise, we would get
                // stuck, because low will be zero
                // then...
                if missing > low {
                    low = missing;
                }
            }
            // Go until we find a high value that needs to be
            // swapped, this will be the first value that our
            // split_value is greater than.
            match exclude_missing_compare(&split_value, h, left_cats) {
                Ordering::Less | Ordering::Equal => high -= 1,
                Ordering::Greater => break,
            }
        }
        if low < high {
            index.swap(high, low);
            gr.swap(high, low);
            hs.swap(high, low);
        }
    }
    (missing, low)
}

#[inline]
pub fn pivot_on_split_exclude_missing_const_hess(
    start: usize,
    stop: usize,
    idx: &mut [usize],
    grad: &mut [f32],
    feature: &[u16],
    split_value: u16,
    left_cats: &HashSet<usize>,
) -> (usize, usize) {
    let index = &mut idx[start..stop];
    let gr = &mut grad[start..stop];
    // I think we can do this in O(n) time...
    let mut low = 0;
    let mut high = index.len() - 1;
    // The index of the first value, that is not
    // missing.
    let mut missing = 0;
    let max_idx = high;
    while low < high {
        // Go until we find a low value that needs to
        // be swapped, this will be the first value
        // that our split value is less or equal to.
        while low < max_idx {
            let l = feature[index[low]];
            if l == 0 {
                index.swap(missing, low);
                gr.swap(missing, low);
                missing += 1;
            }
            match exclude_missing_compare(&split_value, l, left_cats) {
                Ordering::Less | Ordering::Equal => break,
                Ordering::Greater => low += 1,
            }
        }
        while high > low {
            let h = feature[index[high]];
            // If this is missing, we need to
            // swap this value with missing, and
            // then that value with low.
            if h == 0 {
                index.swap(missing, high);
                gr.swap(missing, high);
                missing += 1;
                // Low must be at least equal to
                // missing. Otherwise, we would get
                // stuck, because low will be zero
                // then...
                if missing > low {
                    low = missing;
                }
            }
            // Go until we find a high value that needs to be
            // swapped, this will be the first value that our
            // split_value is greater than.
            match exclude_missing_compare(&split_value, h, left_cats) {
                Ordering::Less | Ordering::Equal => high -= 1,
                Ordering::Greater => break,
            }
        }
        if low < high {
            index.swap(high, low);
            gr.swap(high, low);
        }
    }
    (missing, low)
}

/// Function to compare a value to our split value.
/// Our split value will _never_ be missing (0), thus we
/// don't have to worry about that.
#[inline]
pub fn exclude_missing_compare(split_value: &u16, cmp_value: u16, left_cats: &HashSet<usize>) -> Ordering {
    if !left_cats.is_empty() {
        if left_cats.contains(&(cmp_value as usize)) {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else {
        split_value.cmp(&cmp_value)
    }
}

/// Function to compare a value to our split value.
/// Our split value will _never_ be missing (0), thus we
/// don't have to worry about that.
#[inline]
pub fn missing_compare(split_value: &u16, cmp_value: u16, missing_right: bool, left_cats: &HashSet<usize>) -> Ordering {
    if cmp_value == 0 {
        if missing_right {
            // If missing is right, then our split_value
            // will always be considered less than missing.
            Ordering::Less
        } else {
            // Otherwise less to send it left by considering
            // our split value being always greater than missing
            Ordering::Greater
        }
    } else if !left_cats.is_empty() {
        if left_cats.contains(&(cmp_value as usize)) {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    } else {
        split_value.cmp(&cmp_value)
    }
}

#[inline]
pub fn precision_round(n: f64, precision: i32) -> f64 {
    let p = (10.0_f64).powi(precision);
    (n * p).round() / p
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand::SeedableRng;
    #[test]
    fn test_round() {
        assert_eq!(0.3, precision_round(0.3333, 1));
        assert_eq!(0.2343, precision_round(0.2343123123123, 4));
    }
    #[test]
    fn test_percentiles() {
        let v = vec![4., 5., 6., 1., 2., 3., 7., 8., 9., 10.];
        let w = vec![1.; v.len()];
        let p = vec![0.3, 0.5, 0.75, 1.0];
        let p = percentiles(&v, &w, &p);
        assert_eq!(p, vec![3.0, 5.0, 8.0, 10.0]);
    }

    #[test]
    fn test_percentiles_weighted() {
        let v = vec![10., 8., 9., 1., 2., 3., 6., 7., 4., 5.];
        let w = vec![1., 1., 1., 1., 1., 2., 1., 1., 5., 1.];
        let p = vec![0.3, 0.5, 0.75, 1.0];
        let p = percentiles(&v, &w, &p);
        assert_eq!(p, vec![4.0, 4.0, 7.0, 10.0]);
    }

    #[test]
    fn test_map_bin_or_equal() {
        let v = vec![f64::MIN, 1., 4., 8., 9.];
        assert_eq!(1, map_bin(&v, &0., &f64::NAN).unwrap());
        assert_eq!(2, map_bin(&v, &1., &f64::NAN).unwrap());
        // Less than the bin value of 2, means the value is less
        // than 4...
        assert_eq!(2, map_bin(&v, &2., &f64::NAN).unwrap());
        assert_eq!(3, map_bin(&v, &4., &f64::NAN).unwrap());
        assert_eq!(5, map_bin(&v, &9., &f64::NAN).unwrap());
        assert_eq!(5, map_bin(&v, &10., &f64::NAN).unwrap());
        assert_eq!(2, map_bin(&v, &1., &f64::NAN).unwrap());
        assert_eq!(0, map_bin(&v, &f64::NAN, &f64::NAN).unwrap());
    }

    #[test]
    fn test_map_bin_or_equal_num_miss() {
        let v = vec![f64::MIN, 1., 4., 8., 9.];
        assert_eq!(1, map_bin(&v, &0., &-99.).unwrap());
        assert_eq!(2, map_bin(&v, &1., &-99.).unwrap());
        // Less than the bin value of 2, means the value is less
        // than 4...
        assert_eq!(2, map_bin(&v, &2., &-99.).unwrap());
        assert_eq!(3, map_bin(&v, &4., &-99.).unwrap());
        assert_eq!(5, map_bin(&v, &9., &-99.).unwrap());
        assert_eq!(5, map_bin(&v, &10., &-99.).unwrap());
        assert_eq!(2, map_bin(&v, &1., &-99.).unwrap());
        assert_eq!(0, map_bin(&v, &-99., &-99.).unwrap());
    }

    #[test]
    fn test_missing_compare() {
        assert_eq!(missing_compare(&10, 0, true, &HashSet::new()), Ordering::Less);
        assert_eq!(missing_compare(&10, 0, false, &HashSet::new()), Ordering::Greater);
        assert_eq!(missing_compare(&10, 11, true, &HashSet::new()), Ordering::Less);
        assert_eq!(missing_compare(&10, 1, true, &HashSet::new()), Ordering::Greater);
    }

    #[test]
    fn test_pivot() {
        fn pivot_assert(f: &[u16], idx: &[usize], split_i: usize, missing_right: bool, split_val: u16) {
            if missing_right {
                for i in 0..split_i {
                    assert!((f[idx[i]] < split_val) && f[idx[i]] != 0);
                }
                for i in idx[split_i..].iter() {
                    assert!((f[*i] >= split_val) || (f[*i] == 0));
                }
            } else {
                for i in 0..split_i {
                    assert!((f[idx[i]] < split_val) || (f[idx[i]] == 0));
                }
                for i in idx[split_i..].iter() {
                    assert!((f[*i] >= split_val) || (f[*i] != 0));
                }
            }
        }

        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            10,
            true,
            &HashSet::new(),
        );
        println!("split_i: {}", split_i);
        println!("idx: {:?}", idx);
        println!("sorted: {:?}", idx.iter().map(|i| f[*i]).collect::<Vec<_>>());
        pivot_assert(&f, &idx, split_i, true, 10);

        let missing_right = true;
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 9, 3, 5, 2, 6, 13, 19, 14];
        idx.sort_by_key(|i| {
            if f[*i] == 0 {
                if missing_right {
                    u16::MAX
                } else {
                    u16::MIN
                }
            } else {
                f[*i]
            }
        });
        println!("idx: {:?}", idx);
        println!("sorted: {:?}", idx.iter().map(|i| f[*i]).collect::<Vec<_>>());
        let split_idx = idx.partition_point(|&x| f[x] < 10);
        println!("split_idx: {}", split_idx);
        pivot_assert(&f, &idx, split_idx, true, 10);

        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            10,
            false,
            &HashSet::new(),
        );
        println!("{}", split_i);
        println!("{:?}", idx);
        println!("{:?}", idx.iter().map(|i| f[*i]).collect::<Vec<_>>());
        pivot_assert(&f, &idx, split_i, false, 10);

        // Test Minimum value...
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            1,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 1);

        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            1,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 1);

        // Test Maximum value...
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            19,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 19);

        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let mut grad = vec![0.2, 0.6, 0.9, 0.5, 0.8, 0.1, 0.1, 0.7];
        let mut hess = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let f = vec![15, 10, 10, 11, 3, 18, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            19,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 19);

        // Random tests... right...
        // With missing
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 7);

        // Already sorted...
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 7);

        // Reversed
        idx.reverse();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 7);

        // Without missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            5,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, 5);

        // Using max...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().map(|i| f[*i]).max().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, sv);

        // Using non-0 minimum...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, sv);

        // Using non-0 minimum with no missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            true,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, true, sv);

        // Left
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 7);

        // Already sorted...
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 7);

        // Reversed
        idx.reverse();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            7,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 7);

        // Without missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            5,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, 5);

        // Using max...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().map(|i| f[*i]).max().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, sv);

        // Using non-0 minimum...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, sv);

        // Using non-0 minimum with no missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i = pivot_on_split(
            0,
            idx.len(),
            &mut idx,
            &mut grad,
            &mut hess,
            &f,
            sv,
            false,
            &HashSet::new(),
        );
        pivot_assert(&f, &idx, split_i, false, sv);
    }

    #[test]
    fn test_pivot_missing() {
        fn pivot_missing_assert(split_value: u16, idx: &[usize], f: &[u16], split_i: &(usize, usize)) {
            // Check they are lower than..
            for i in 0..split_i.1 {
                assert!(f[idx[i]] < split_value);
            }
            // Check missing got moved
            for i in 0..split_i.0 {
                assert!(f[idx[i]] == 0);
            }
            // Check none are less than...
            for i in split_i.1..(idx.len()) {
                assert!(!(f[idx[i]] < split_value));
            }
            // Check none other are missing...
            for i in split_i.0..(idx.len()) {
                assert!(f[idx[i]] != 0);
            }
        }
        // TODO: Add more tests for this...
        // Using minimum value...
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 0, 3, 0, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 1, &HashSet::new());
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(1, &idx, &f, &split_i);

        // Higher value...
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 0, 3, 0, 0, 9, 3, 5, 2, 6, 13, 19, 14];
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        //let split_i = pivot_on_split(&mut idx, &f, 10, false);
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Run it again, and ensure it works on an already sorted list...
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        //let split_i = pivot_on_split(&mut idx, &f, 10, false);
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Run it again, and ensure it works on reversed list...
        idx.reverse();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        //let split_i = pivot_on_split(&mut idx, &f, 10, false);
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Small test done with python
        let mut idx = vec![0, 1, 2, 3, 4, 5];
        let f = vec![1, 0, 1, 3, 0, 4];
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 2, &HashSet::new());
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(2, &idx, &f, &split_i);

        // Ensure it works on all missing...
        // let mut idx = vec![0, 1, 2, 3, 4, 5];
        // let f: Vec<u16> = vec![3; idx.len()];
        // let split_i = pivot_on_split_exclude_missing(&mut idx, &f, 2);
        // // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        // pivot_missing_assert(2, &idx, &f, &split_i);

        // Check if none missing...
        // TODO: Add more tests for this...
        let mut idx = vec![2, 6, 9, 5, 8, 13, 11, 7];
        let f = vec![15, 10, 10, 2, 3, 5, 7, 9, 3, 5, 2, 6, 13, 19, 14];
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        //let split_i = pivot_on_split(&mut idx, &f, 10, false);
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Random tests...
        // With missing
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Already sorted...
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 10, &HashSet::new());
        pivot_missing_assert(10, &idx, &f, &split_i);

        // Without missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, 5, &HashSet::new());
        // let map_ = idx.iter().map(|i| f[*i]).collect::<Vec<u16>>();
        // println!("{:?}, {:?}, {:?}", split_i, idx, map_);
        pivot_missing_assert(5, &idx, &f, &split_i);

        // Using max...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().map(|i| f[*i]).max().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, sv, &HashSet::new());
        pivot_missing_assert(sv, &idx, &f, &split_i);

        // Using non-0 minimum...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(0..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, sv, &HashSet::new());
        pivot_missing_assert(sv, &idx, &f, &split_i);

        // Using non-0 minimum with no missing...
        let index = (0..100).collect::<Vec<usize>>();
        let mut rng = StdRng::seed_from_u64(0);
        let f = (0..100).map(|_| rng.gen_range(1..15)).collect::<Vec<u16>>();
        let mut idx = index.choose_multiple(&mut rng, 73).copied().collect::<Vec<usize>>();
        let sv = idx.iter().filter(|i| f[**i] > 0).map(|i| f[*i]).min().unwrap();
        let mut grad = idx.iter().map(|i| *i as f32).collect::<Vec<f32>>();
        let mut hess = grad.clone();
        let split_i =
            pivot_on_split_exclude_missing(0, idx.len(), &mut idx, &mut grad, &mut hess, &f, sv, &HashSet::new());
        pivot_missing_assert(sv, &idx, &f, &split_i);
    }

    #[test]
    fn test_fast_f64_sum() {
        let records = 300000;
        let vec = vec![0.23500371; records];
        assert_ne!(vec.iter().sum::<f32>(), vec[0] * (records as f32));
        assert_eq!(vec[0] * (records as f32), fast_f64_sum(&vec));
        // println!("Sum Result: {}", vec.iter().sum::<f32>());
        // println!("Multiplication Results {}", vec[0] * (records as f32));
        // println!("f64_sum Results {}", f64_sum(&vec));
    }

    #[test]
    fn test_fmt_vec_output() {
        let v = Vec::<f32>::new();
        assert_eq!(fmt_vec_output(&v), String::from(""));
        let v: Vec<f32> = vec![0.1, 1.0];
        assert_eq!(fmt_vec_output(&v), String::from("0.1000, 1.0000"));
    }
}
