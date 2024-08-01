use serde::{Deserialize, Serialize};
use std::fmt::{self, Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::str::FromStr;

/// Data trait used throughout the package
/// to control for floating point numbers.
pub trait FloatData<T>:
    Mul<Output = T>
    + Display
    + Add<Output = T>
    + Div<Output = T>
    + Neg<Output = T>
    + Copy
    + Debug
    + PartialEq
    + PartialOrd
    + AddAssign
    + Sub<Output = T>
    + SubAssign
    + Sum
    + std::marker::Send
    + std::marker::Sync
{
    const ZERO: T;
    const ONE: T;
    const MIN: T;
    const MAX: T;
    const NAN: T;
    const INFINITY: T;
    fn from_usize(v: usize) -> T;
    fn from_u16(v: u16) -> T;
    fn is_nan(self) -> bool;
    fn ln(self) -> T;
    fn exp(self) -> T;
}
impl FloatData<f64> for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const MIN: f64 = f64::MIN;
    const MAX: f64 = f64::MAX;
    const NAN: f64 = f64::NAN;
    const INFINITY: f64 = f64::INFINITY;

    fn from_usize(v: usize) -> f64 {
        v as f64
    }
    fn from_u16(v: u16) -> f64 {
        f64::from(v)
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn ln(self) -> f64 {
        self.ln()
    }
    fn exp(self) -> f64 {
        self.exp()
    }
}

impl FloatData<f32> for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const MIN: f32 = f32::MIN;
    const MAX: f32 = f32::MAX;
    const NAN: f32 = f32::NAN;
    const INFINITY: f32 = f32::INFINITY;

    fn from_usize(v: usize) -> f32 {
        v as f32
    }
    fn from_u16(v: u16) -> f32 {
        f32::from(v)
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn ln(self) -> f32 {
        self.ln()
    }
    fn exp(self) -> f32 {
        self.exp()
    }
}

/// Contigious Column major matrix data container. This is
/// used throughout the crate, to house both the user provided data
/// as well as the binned data.
pub struct Matrix<'a, T> {
    pub data: &'a [T],
    pub index: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    stride1: usize,
    stride2: usize,
}

impl<'a, T> Matrix<'a, T> {
    // Defaults to column major
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Matrix {
            data,
            index: (0..rows).collect(),
            rows,
            cols,
            stride1: rows,
            stride2: 1,
        }
    }

    /// Get a single reference to an item in the matrix.
    ///
    /// * `i` - The ith row of the data to get.
    /// * `j` - the jth column of the data to get.
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[self.item_index(i, j)]
    }

    fn item_index(&self, i: usize, j: usize) -> usize {
        let mut idx = self.stride2 * i;
        idx += j * self.stride1;
        idx
    }

    /// Get access to a row of the data, as an iterator.
    pub fn get_row_iter(&self, row: usize) -> std::iter::StepBy<std::iter::Skip<std::slice::Iter<T>>> {
        self.data.iter().skip(row).step_by(self.rows)
    }

    /// Get a slice of a column in the matrix.
    ///
    /// * `col` - The index of the column to select.
    /// * `start_row` - The index of the start of the slice.
    /// * `end_row` - The index of the end of the slice of the column to select.
    pub fn get_col_slice(&self, col: usize, start_row: usize, end_row: usize) -> &[T] {
        let i = self.item_index(start_row, col);
        let j = self.item_index(end_row, col);
        &self.data[i..j]
    }

    /// Get an entire column in the matrix.
    ///
    /// * `col` - The index of the column to get.
    pub fn get_col(&self, col: usize) -> &[T] {
        self.get_col_slice(col, 0, self.rows)
    }
}

impl<'a, T> Matrix<'a, T>
where
    T: Copy,
{
    /// Get a row of the data as a vector.
    pub fn get_row(&self, row: usize) -> Vec<T> {
        self.get_row_iter(row).copied().collect()
    }
}

/// A lightweight row major matrix, this is primarily
/// for returning data to the user, it is especially
/// suited for appending rows to, such as when building
/// up a matrix of data to return to the
/// user, the added benefit is it will be even
/// faster to return to numpy.
#[derive(Debug, Serialize, Deserialize)]
pub struct RowMajorMatrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
    stride1: usize,
    stride2: usize,
}

impl<T> RowMajorMatrix<T> {
    // Defaults to column major
    pub fn new(data: Vec<T>, rows: usize, cols: usize) -> Self {
        RowMajorMatrix {
            data,
            rows,
            cols,
            stride1: 1,
            stride2: cols,
        }
    }

    /// Get a single reference to an item in the matrix.
    ///
    /// * `i` - The ith row of the data to get.
    /// * `j` - the jth column of the data to get.
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[self.item_index(i, j)]
    }

    fn item_index(&self, i: usize, j: usize) -> usize {
        let mut idx = self.stride2 * i;
        idx += j * self.stride1;
        idx
    }

    /// Add a rows to the matrix, this can be multiple
    /// rows, if they are in sequential order in the items.
    pub fn append_row(&mut self, items: Vec<T>) {
        assert!(items.len() % self.cols == 0);
        let new_rows = items.len() / self.cols;
        self.rows += new_rows;
        self.data.extend(items);
    }
}

impl<'a, T> fmt::Display for Matrix<'a, T>
where
    T: FromStr + std::fmt::Display,
    <T as FromStr>::Err: 'static + std::error::Error,
{
    // This trait requires `fmt` with this exact signature.
    /// Format a Matrix.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut val = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                val.push_str(self.get(i, j).to_string().as_str());
                if j == (self.cols - 1) {
                    val.push('\n');
                } else {
                    val.push(' ');
                }
            }
        }
        write!(f, "{}", val)
    }
}

/// A jagged column aligned matrix, that owns its data contents.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct JaggedMatrix<T> {
    /// The contents of the matrix.
    pub data: Vec<T>,
    /// The end index's of the matrix.
    pub ends: Vec<usize>,
    /// Number of columns in the matrix
    pub cols: usize,
    /// The number of elements in the matrix.
    pub n_records: usize,
}

impl<T> JaggedMatrix<T>
where
    T: Copy,
{
    /// Generate a jagged array from a vector of vectors
    pub fn from_vecs(vecs: &[Vec<T>]) -> Self {
        let mut data = Vec::new();
        let mut ends = Vec::new();
        let mut e = 0;
        let mut n_records = 0;
        for vec in vecs {
            for v in vec {
                data.push(*v);
            }
            e += vec.len();
            ends.push(e);
            n_records += e;
        }
        let cols = vecs.len();

        JaggedMatrix {
            data,
            ends,
            cols,
            n_records,
        }
    }
}

impl<T> JaggedMatrix<T> {
    /// Create a new jagged matrix.
    pub fn new() -> Self {
        JaggedMatrix {
            data: Vec::new(),
            ends: Vec::new(),
            cols: 0,
            n_records: 0,
        }
    }
    /// Get the column of a jagged array.
    pub fn get_col(&self, col: usize) -> &[T] {
        assert!(col < self.ends.len());
        let (i, j) = if col == 0 {
            (0, self.ends[col])
        } else {
            (self.ends[col - 1], self.ends[col])
        };
        &self.data[i..j]
    }

    /// Get a mutable reference to a column of the array.
    pub fn get_col_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.ends.len());
        let (i, j) = if col == 0 {
            (0, self.ends[col])
        } else {
            (self.ends[col - 1], self.ends[col])
        };
        &mut self.data[i..j]
    }
}

impl<T> Default for JaggedMatrix<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rowmatrix_get() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = RowMajorMatrix::new(v, 2, 3);
        println!("{:?}", m);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &5);
        assert_eq!(m.get(0, 2), &3);
        assert_eq!(m.get(1, 1), &6);
    }

    #[test]
    fn test_rowmatrix_append() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let mut m = RowMajorMatrix::new(v, 2, 3);
        m.append_row(vec![-1, -2, -3]);
        assert_eq!(m.get(2, 1), &-2);
    }

    #[test]
    fn test_matrix_get() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 2, 3);
        println!("{}", m);
        assert_eq!(m.get(0, 0), &1);
        assert_eq!(m.get(1, 0), &2);
    }
    #[test]
    fn test_matrix_get_col_slice() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col_slice(0, 0, 3), &vec![1, 2, 3]);
        assert_eq!(m.get_col_slice(1, 0, 2), &vec![5, 6]);
        assert_eq!(m.get_col_slice(1, 1, 3), &vec![6, 7]);
        assert_eq!(m.get_col_slice(0, 1, 2), &vec![2]);
    }

    #[test]
    fn test_matrix_get_col() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_col(1), &vec![5, 6, 7]);
    }

    #[test]
    fn test_matrix_row() {
        let v = vec![1, 2, 3, 5, 6, 7];
        let m = Matrix::new(&v, 3, 2);
        assert_eq!(m.get_row(2), vec![3, 7]);
        assert_eq!(m.get_row(0), vec![1, 5]);
        assert_eq!(m.get_row(1), vec![2, 6]);
    }

    #[test]
    fn test_jaggedmatrix_get_col() {
        let vecs = vec![vec![0], vec![5, 4, 3, 2], vec![4, 5]];
        let jmatrix = JaggedMatrix::from_vecs(&vecs);
        assert_eq!(jmatrix.get_col(1), vec![5, 4, 3, 2]);
        assert_eq!(jmatrix.get_col(0), vec![0]);
        assert_eq!(jmatrix.get_col(2), vec![4, 5]);
    }
}
