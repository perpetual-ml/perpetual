//! Histogram
//!
//! Efficient histogram calculations for finding optimal splits.
//! Histograms store aggregated gradient and hessian statistics for each bin.
use crate::Matrix;
use crate::bin::Bin;
use crate::data::{FloatData, JaggedMatrix};
use rayon::{ThreadPool, prelude::*};
use std::cell::UnsafeCell;

/// Owned Feature Histogram.
#[derive(Debug)]
pub struct FeatureHistogramOwned {
    /// The histogram data (bins).
    pub data: Vec<Bin>,
}

impl FeatureHistogramOwned {
    /// Create an empty histogram from cut points.
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

    /// Create an empty histogram with a maximum number of bins.
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

/// Feature Histogram using UnsafeCell for concurrent mutation.
#[derive(Copy, Clone, Debug)]
pub struct FeatureHistogram<'a> {
    /// Reference to the histogram data.
    pub data: &'a [UnsafeCell<Bin>],
}

unsafe impl<'a> Send for FeatureHistogram<'a> {}
unsafe impl<'a> Sync for FeatureHistogram<'a> {}

impl<'a> FeatureHistogram<'a> {
    /// Create a new FeatureHistogram from a mutable slice of bins.
    pub fn new(hist: &'a mut [Bin]) -> Self {
        let ptr = hist as *mut [Bin] as *const [UnsafeCell<Bin>];
        Self { data: unsafe { &*ptr } }
    }

    /// Updates the histogram data based on the provided gradients and Hessian values.
    ///
    /// # Arguments
    /// * `feature`: A slice of bin indices for each data element.
    /// * `sorted_grad`: The full gradient array (absolute indexing via `index`).
    /// * `sorted_hess`: An optional full Hessian array (absolute indexing via `index`).
    /// * `index`: A slice of original indices for the data elements in this node.
    ///
    /// # Safety
    /// This function is unsafe because it uses `get_unchecked` and `as_mut().unwrap_unchecked()`,
    /// which bypass Rust's standard safety checks. The caller must ensure the following:
    ///
    /// 1. The `feature` slice must be a valid bin index for every element.
    /// 2. All indices in the `index` slice must be within the bounds of the `feature` slice.
    /// 3. The `sorted_grad` slice must cover all indices referenced by `index`.
    /// 4. If `sorted_hess` is `Some`, it must also cover all indices referenced by `index`.
    /// 5. The internal `self.data` structure must not be modified externally while this function is running.
    /// 6. Each element in `self.data` must contain a valid `Some` value that can be mutated.
    pub unsafe fn update(&self, feature: &[u16], sorted_grad: &[f32], sorted_hess: Option<&[f32]>, index: &[usize]) {
        unsafe {
            let n_bins = self.data.len();
            let n = index.len();

            // For very small nodes, accumulate directly into Bin structs.
            // The overhead of zeroing a 6 KB+ stack array dominates for < ~64 samples.
            if n < 64 {
                match sorted_hess {
                    Some(sorted_hess) => {
                        self.data.iter().for_each(|b| {
                            let bin = b.get().as_mut().unwrap_unchecked();
                            bin.g_folded = [f32::ZERO; 5];
                            bin.h_folded = [f32::ZERO; 5];
                            bin.counts = [0; 5];
                        });
                        for k in 0..n {
                            let i = *index.get_unchecked(k);
                            let b = self.data.get_unchecked(*feature.get_unchecked(i) as usize).get();
                            let bin = b.as_mut().unwrap_unchecked();
                            let fold = i % 5;
                            *bin.g_folded.get_unchecked_mut(fold) += *sorted_grad.get_unchecked(i);
                            *bin.h_folded.get_unchecked_mut(fold) += *sorted_hess.get_unchecked(i);
                            *bin.counts.get_unchecked_mut(fold) += 1;
                        }
                    }
                    None => {
                        self.data.iter().for_each(|b| {
                            let bin = b.get().as_mut().unwrap_unchecked();
                            bin.g_folded = [f32::ZERO; 5];
                            bin.counts = [0; 5];
                            bin.h_folded = [f32::ZERO; 5];
                        });
                        for k in 0..n {
                            let i = *index.get_unchecked(k);
                            let b = self.data.get_unchecked(*feature.get_unchecked(i) as usize).get();
                            let bin = b.as_mut().unwrap_unchecked();
                            let fold = i % 5;
                            *bin.g_folded.get_unchecked_mut(fold) += *sorted_grad.get_unchecked(i);
                            *bin.counts.get_unchecked_mut(fold) += 1;
                        }
                    }
                }
                return;
            }

            // ── Flat-buffer histogram accumulation ──
            //
            // Instead of accumulating directly into 72-byte Bin structs (which causes
            // cache-line thrashing on random scatter writes), we accumulate into compact
            // flat arrays laid out as [bin0_fold0..bin0_fold4, bin1_fold0..].
            //
            // For 256 bins: grad = 256×5×4 = 5 KB, counts = 256×5×4 = 5 KB → 10 KB total,
            // fitting entirely in L1 cache. This eliminates the dominant cache-miss cost
            // of the histogram building phase.
            //
            // Stack-allocated with a compile-time max to avoid heap allocations.
            const MAX_FLAT: usize = 300 * 5; // supports up to 300 bins
            debug_assert!(n_bins * 5 <= MAX_FLAT, "n_bins {} exceeds flat buffer capacity", n_bins);
            let flat_len = n_bins * 5;

            // Use MaybeUninit to avoid zeroing the full MAX_FLAT array.
            // Only zero the flat_len portion we actually use via write_bytes.
            let mut flat_grad_storage: std::mem::MaybeUninit<[f32; MAX_FLAT]> = std::mem::MaybeUninit::uninit();
            let mut flat_counts_storage: std::mem::MaybeUninit<[u32; MAX_FLAT]> = std::mem::MaybeUninit::uninit();
            let gp = flat_grad_storage.as_mut_ptr() as *mut f32;
            let cp = flat_counts_storage.as_mut_ptr() as *mut u32;
            core::ptr::write_bytes(gp, 0, flat_len);
            core::ptr::write_bytes(cp, 0, flat_len);
            let flat_grad = core::slice::from_raw_parts_mut(gp, flat_len);
            let flat_counts = core::slice::from_raw_parts_mut(cp, flat_len);

            match sorted_hess {
                Some(sorted_hess) => {
                    let mut flat_hess_storage: std::mem::MaybeUninit<[f32; MAX_FLAT]> = std::mem::MaybeUninit::uninit();
                    let hp = flat_hess_storage.as_mut_ptr() as *mut f32;
                    core::ptr::write_bytes(hp, 0, flat_len);
                    let flat_hess = core::slice::from_raw_parts_mut(hp, flat_len);

                    // Prefetch source data to hide memory latency on reads
                    #[cfg(target_arch = "x86_64")]
                    {
                        use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
                        const PF_FAR: usize = 16;
                        const PF_NEAR: usize = 4;
                        let n = index.len();
                        for k in 0..n {
                            if k + PF_FAR < n {
                                let far = *index.get_unchecked(k + PF_FAR);
                                _mm_prefetch(feature.as_ptr().add(far) as *const i8, _MM_HINT_T1);
                                _mm_prefetch(sorted_grad.as_ptr().add(far) as *const i8, _MM_HINT_T1);
                                _mm_prefetch(sorted_hess.as_ptr().add(far) as *const i8, _MM_HINT_T1);
                            }
                            if k + PF_NEAR < n {
                                let near = *index.get_unchecked(k + PF_NEAR);
                                _mm_prefetch(feature.as_ptr().add(near) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(sorted_grad.as_ptr().add(near) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(sorted_hess.as_ptr().add(near) as *const i8, _MM_HINT_T0);
                            }
                            let i = *index.get_unchecked(k);
                            let bin_idx = *feature.get_unchecked(i) as usize;
                            let slot = bin_idx * 5 + (i % 5);
                            *flat_grad.get_unchecked_mut(slot) += *sorted_grad.get_unchecked(i);
                            *flat_hess.get_unchecked_mut(slot) += *sorted_hess.get_unchecked(i);
                            *flat_counts.get_unchecked_mut(slot) += 1;
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for k in 0..index.len() {
                            let i = *index.get_unchecked(k);
                            let bin_idx = *feature.get_unchecked(i) as usize;
                            let slot = bin_idx * 5 + (i % 5);
                            *flat_grad.get_unchecked_mut(slot) += *sorted_grad.get_unchecked(i);
                            *flat_hess.get_unchecked_mut(slot) += *sorted_hess.get_unchecked(i);
                            *flat_counts.get_unchecked_mut(slot) += 1;
                        }
                    }

                    // Scatter from flat buffers → Bin structs (sequential, cache-friendly)
                    for b_idx in 0..n_bins {
                        let bin = self.data.get_unchecked(b_idx).get().as_mut().unwrap_unchecked();
                        let base = b_idx * 5;
                        bin.g_folded.copy_from_slice(flat_grad.get_unchecked(base..base + 5));
                        bin.h_folded.copy_from_slice(flat_hess.get_unchecked(base..base + 5));
                        bin.counts.copy_from_slice(flat_counts.get_unchecked(base..base + 5));
                    }
                }
                None => {
                    // const_hess path — no hessian
                    #[cfg(target_arch = "x86_64")]
                    {
                        use core::arch::x86_64::{_MM_HINT_T0, _MM_HINT_T1, _mm_prefetch};
                        const PF_FAR: usize = 16;
                        const PF_NEAR: usize = 4;
                        let n = index.len();
                        for k in 0..n {
                            if k + PF_FAR < n {
                                let far = *index.get_unchecked(k + PF_FAR);
                                _mm_prefetch(feature.as_ptr().add(far) as *const i8, _MM_HINT_T1);
                                _mm_prefetch(sorted_grad.as_ptr().add(far) as *const i8, _MM_HINT_T1);
                            }
                            if k + PF_NEAR < n {
                                let near = *index.get_unchecked(k + PF_NEAR);
                                _mm_prefetch(feature.as_ptr().add(near) as *const i8, _MM_HINT_T0);
                                _mm_prefetch(sorted_grad.as_ptr().add(near) as *const i8, _MM_HINT_T0);
                            }
                            let i = *index.get_unchecked(k);
                            let bin_idx = *feature.get_unchecked(i) as usize;
                            let slot = bin_idx * 5 + (i % 5);
                            *flat_grad.get_unchecked_mut(slot) += *sorted_grad.get_unchecked(i);
                            *flat_counts.get_unchecked_mut(slot) += 1;
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for k in 0..index.len() {
                            let i = *index.get_unchecked(k);
                            let bin_idx = *feature.get_unchecked(i) as usize;
                            let slot = bin_idx * 5 + (i % 5);
                            *flat_grad.get_unchecked_mut(slot) += *sorted_grad.get_unchecked(i);
                            *flat_counts.get_unchecked_mut(slot) += 1;
                        }
                    }

                    // Scatter from flat buffers → Bin structs
                    for b_idx in 0..n_bins {
                        let bin = self.data.get_unchecked(b_idx).get().as_mut().unwrap_unchecked();
                        let base = b_idx * 5;
                        bin.g_folded.copy_from_slice(flat_grad.get_unchecked(base..base + 5));
                        bin.counts.copy_from_slice(flat_counts.get_unchecked(base..base + 5));
                        bin.h_folded = [f32::ZERO; 5];
                    }
                }
            }
        }
    }

    /// Updates the cut-off values for the histogram bins.
    ///
    /// This function is unsafe because...
    ///
    /// # Safety
    /// The `cuts` slice must be sorted in ascending order.
    /// The length of `cuts` must be exactly one less than the number of bins.
    /// Calling this function with an unsorted slice or an incorrect length
    /// could lead to incorrect binning logic and potential data corruption.
    pub unsafe fn update_cuts(&self, cuts: &[f64]) {
        unsafe {
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
}

/// Owned Node Histogram.
#[derive(Debug)]
pub struct NodeHistogramOwned {
    /// The histograms for each feature in the node.
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

/// Arena-based bulk histogram storage.
///
/// Stores all histogram bins for all nodes in a single contiguous allocation,
/// eliminating the overhead of thousands of individual Vec allocations.
/// For n_nodes=10000 with 8 features, this replaces ~80000 heap allocations
/// with a single one, dramatically reducing allocation overhead.
pub struct HistogramArena {
    /// Single contiguous allocation of all bins.
    bins: Vec<Bin>,
    /// Number of bins per feature (same order as col_index).
    col_bin_counts: Vec<usize>,
    /// Number of nodes.
    n_nodes: usize,
}

impl HistogramArena {
    /// Create arena from cuts (variable bins per feature).
    /// Uses alloc_zeroed for demand-paged zero memory, then parallel init of
    /// only num/cut_value fields to parallelize page faults across cores.
    pub fn from_cuts(cuts: &JaggedMatrix<f64>, col_index: &[usize], _is_const_hess: bool, n_nodes: usize) -> Self {
        let col_cuts: Vec<&[f64]> = col_index.iter().map(|&col| cuts.get_col(col)).collect();
        let col_bin_counts: Vec<usize> = col_cuts.iter().map(|c| c.len()).collect();
        let bins_per_node: usize = col_bin_counts.iter().sum();
        let total_bins = bins_per_node * n_nodes;

        // Allocate zeroed memory (OS provides demand-paged zero pages for large allocs).
        // g_folded, h_folded, counts are all correctly zero. Only num/cut_value need setting.
        let mut bins: Vec<Bin> = Self::alloc_zeroed_bins(total_bins);

        // Build template arrays for num and cut_value (per-node layout).
        let mut template_num: Vec<u16> = Vec::with_capacity(bins_per_node);
        let mut template_cv: Vec<f64> = Vec::with_capacity(bins_per_node);
        for col_cuts_slice in &col_cuts {
            let cuts_mod = &col_cuts_slice[..(col_cuts_slice.len() - 1)];
            template_num.push(0);
            template_cv.push(f64::NAN);
            for (it, c) in cuts_mod.iter().enumerate() {
                template_num.push(it as u16 + 1);
                template_cv.push(*c);
            }
        }

        // Parallel init of num/cut_value across all nodes.
        // This triggers page faults in parallel across rayon threads,
        // parallelizing the OS overhead of backing virtual memory with physical pages.
        bins.par_chunks_mut(bins_per_node).for_each(|node_bins| {
            for (bin, (&num, &cv)) in node_bins.iter_mut().zip(template_num.iter().zip(template_cv.iter())) {
                bin.num = num;
                bin.cut_value = cv;
            }
        });

        HistogramArena {
            bins,
            col_bin_counts,
            n_nodes,
        }
    }

    /// Create arena with fixed max_bin for all features.
    pub fn from_fixed(max_bin: u16, col_amount: usize, _is_const_hess: bool, n_nodes: usize) -> Self {
        let bins_per_feature = max_bin as usize + 2;
        let col_bin_counts: Vec<usize> = vec![bins_per_feature; col_amount];
        let bins_per_node = bins_per_feature * col_amount;
        let total_bins = bins_per_node * n_nodes;

        // Allocate zeroed memory
        let mut bins: Vec<Bin> = Self::alloc_zeroed_bins(total_bins);

        // Build template for fixed bins (all features have same layout)
        let mut template_num: Vec<u16> = Vec::with_capacity(bins_per_node);
        let mut template_cv: Vec<f64> = Vec::with_capacity(bins_per_node);
        for _col in 0..col_amount {
            template_num.push(0);
            template_cv.push(f64::NAN);
            for i in 0..(max_bin + 1) {
                template_num.push(i + 1);
                template_cv.push(f64::NAN);
            }
        }

        // Parallel init of num/cut_value
        bins.par_chunks_mut(bins_per_node).for_each(|node_bins| {
            for (bin, (&num, &cv)) in node_bins.iter_mut().zip(template_num.iter().zip(template_cv.iter())) {
                bin.num = num;
                bin.cut_value = cv;
            }
        });

        HistogramArena {
            bins,
            col_bin_counts,
            n_nodes,
        }
    }

    /// Allocate a Vec<Bin> of `count` bins, all zeroed.
    /// Uses alloc_zeroed so the OS can provide demand-paged zero memory
    /// for large allocations without physical zeroing overhead.
    fn alloc_zeroed_bins(count: usize) -> Vec<Bin> {
        if count == 0 {
            return Vec::new();
        }
        unsafe {
            let layout = std::alloc::Layout::array::<Bin>(count).unwrap();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Bin;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(ptr, count, count)
        }
    }

    /// Create NodeHistogram references into this arena.
    ///
    /// Returns a Vec of NodeHistogram that borrow from this arena.
    /// The arena must outlive the returned histograms.
    pub fn as_node_histograms(&mut self) -> Vec<NodeHistogram<'_>> {
        let mut result: Vec<NodeHistogram<'_>> = Vec::with_capacity(self.n_nodes);
        let n_cols = self.col_bin_counts.len();

        // Cast the entire bins slice to UnsafeCell<Bin> slice (same layout, repr(transparent))
        let all_cells: &[UnsafeCell<Bin>] = unsafe {
            let ptr = self.bins.as_ptr() as *const UnsafeCell<Bin>;
            std::slice::from_raw_parts(ptr, self.bins.len())
        };

        let mut offset = 0;
        for _node in 0..self.n_nodes {
            let mut features: Vec<FeatureHistogram<'_>> = Vec::with_capacity(n_cols);
            for &n_bins in &self.col_bin_counts {
                features.push(FeatureHistogram {
                    data: &all_cells[offset..offset + n_bins],
                });
                offset += n_bins;
            }
            result.push(NodeHistogram { data: features });
        }

        result
    }
}

/// Node Histogram.
#[derive(Debug)]
pub struct NodeHistogram<'a> {
    /// The histograms for each feature in the node.
    pub data: Vec<FeatureHistogram<'a>>,
}

impl<'a> NodeHistogram<'a> {
    /// Create a NodeHistogram from an owned one.
    pub fn from_owned(hist: &'a mut NodeHistogramOwned) -> NodeHistogram<'a> {
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

/// Update the cut values in the histogram.
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

/// Update the histogram with new data.
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
    _sort: bool,
) {
    // With absolute indexing, we always pass the full grad/hess arrays.
    // The FeatureHistogram::update reads grad[index[k]] directly.
    let sorted_grad = grad;
    let sorted_hess = hess;

    unsafe {
        let n_samples = stop - start;
        // For small nodes, the Rayon scope overhead (~1-5μs per scope) exceeds
        // the parallelism benefit. Skip parallelism when there are fewer than
        // 512 samples to process per feature.
        if pool.current_num_threads() > 1 && n_samples >= 512 {
            pool.scope(|s| {
                for (i, &col) in col_index.iter().enumerate().take(hist.data.len()) {
                    let h = hist.data.get_unchecked(i);
                    let feature = data.get_col(col); // Use the value 'col' directly
                    s.spawn(|_| {
                        h.update(feature, sorted_grad, sorted_hess, &index[start..stop]);
                    });
                }
            });
        } else {
            col_index.iter().enumerate().for_each(|(i, col)| {
                hist.data
                    .get_unchecked(i)
                    .update(data.get_col(*col), sorted_grad, sorted_hess, &index[start..stop]);
            });
        }
    }
}

/// Build histogram for the smaller child AND derive the larger child's
/// histogram via subtraction from the parent, all in a single Rayon scope.
///
/// This fuses `update_histogram` + `NodeHistogram::from_parent_child` into one
/// parallel step, eliminating a separate sequential subtraction pass and
/// improving cache locality (each feature's histogram data is still hot in L1
/// when the subtraction runs).
///
/// # Safety
/// Same requirements as `update_histogram` plus:
/// - `parent_num`, `child_num`, and `update_num` must be valid indices into `hist_tree`.
/// - The parent histogram must already be populated.
#[allow(clippy::too_many_arguments)]
pub fn update_histogram_and_subtract(
    hist_tree: &[NodeHistogram],
    parent_num: usize,
    child_num: usize,
    update_num: usize,
    start: usize,
    stop: usize,
    data: &Matrix<u16>,
    grad: &[f32],
    hess: Option<&[f32]>,
    index: &[usize],
    col_index: &[usize],
    pool: &ThreadPool,
) {
    let sorted_grad = grad;
    let sorted_hess = hess;

    unsafe {
        let child_hist = hist_tree.get_unchecked(child_num);
        let parent_hist = hist_tree.get_unchecked(parent_num);
        let update_hist = hist_tree.get_unchecked(update_num);
        let n_samples = stop - start;

        if pool.current_num_threads() > 1 && n_samples >= 512 {
            pool.scope(|s| {
                for (i, &col) in col_index.iter().enumerate().take(child_hist.data.len()) {
                    let ch = child_hist.data.get_unchecked(i);
                    let ph = parent_hist.data.get_unchecked(i);
                    let uh = update_hist.data.get_unchecked(i);
                    let feature = data.get_col(col);
                    s.spawn(move |_| {
                        // Step 1: Build child histogram
                        ch.update(feature, sorted_grad, sorted_hess, &index[start..stop]);
                        // Step 2: Derive sibling histogram via subtraction
                        // (cache-local: child histogram data is still hot in L1)
                        ph.data.iter().zip(ch.data.iter()).zip(uh.data.iter()).for_each(
                            |((parent_cell, child_cell), update_cell)| {
                                Bin::from_parent_child(parent_cell.get(), child_cell.get(), update_cell.get())
                            },
                        );
                    });
                }
            });
        } else {
            // Sequential fallback: build child histogram then subtract
            col_index.iter().enumerate().for_each(|(i, col)| {
                child_hist.data.get_unchecked(i).update(
                    data.get_col(*col),
                    sorted_grad,
                    sorted_hess,
                    &index[start..stop],
                );
            });
            NodeHistogram::from_parent_child(hist_tree, parent_num, child_num, update_num);
        }
    }
}

/// Build histograms for two smaller children AND derive the largest child's
/// histogram via subtraction from the parent, all in a single Rayon scope.
///
/// Used when a node splits into 3 children (left, right, missing-branch).
/// The two smaller children get their histograms built from data, and the
/// largest child's histogram is derived as parent - first - second.
#[allow(clippy::too_many_arguments)]
pub fn update_two_histograms_and_subtract(
    hist_tree: &[NodeHistogram],
    parent_num: usize,
    first_num: usize,
    first_start: usize,
    first_stop: usize,
    second_num: usize,
    second_start: usize,
    second_stop: usize,
    update_num: usize,
    data: &Matrix<u16>,
    grad: &[f32],
    hess: Option<&[f32]>,
    index: &[usize],
    col_index: &[usize],
    pool: &ThreadPool,
) {
    let sorted_grad = grad;
    let sorted_hess = hess;

    unsafe {
        let first_hist = hist_tree.get_unchecked(first_num);
        let second_hist = hist_tree.get_unchecked(second_num);
        let parent_hist = hist_tree.get_unchecked(parent_num);
        let update_hist = hist_tree.get_unchecked(update_num);
        let n_samples = (first_stop - first_start) + (second_stop - second_start);

        if pool.current_num_threads() > 1 && n_samples >= 512 {
            pool.scope(|s| {
                for (i, &col) in col_index.iter().enumerate().take(first_hist.data.len()) {
                    let fh = first_hist.data.get_unchecked(i);
                    let sh = second_hist.data.get_unchecked(i);
                    let ph = parent_hist.data.get_unchecked(i);
                    let uh = update_hist.data.get_unchecked(i);
                    let feature = data.get_col(col);
                    s.spawn(move |_| {
                        // Build both children's histograms
                        fh.update(feature, sorted_grad, sorted_hess, &index[first_start..first_stop]);
                        sh.update(feature, sorted_grad, sorted_hess, &index[second_start..second_stop]);
                        // Derive largest child via: update = parent - first - second
                        ph.data
                            .iter()
                            .zip(fh.data.iter())
                            .zip(sh.data.iter())
                            .zip(uh.data.iter())
                            .for_each(|(((pc, fc), sc), uc)| {
                                Bin::from_parent_two_children(pc.get(), fc.get(), sc.get(), uc.get())
                            });
                    });
                }
            });
        } else {
            // Sequential fallback
            col_index.iter().enumerate().for_each(|(i, col)| {
                first_hist.data.get_unchecked(i).update(
                    data.get_col(*col),
                    sorted_grad,
                    sorted_hess,
                    &index[first_start..first_stop],
                );
                second_hist.data.get_unchecked(i).update(
                    data.get_col(*col),
                    sorted_grad,
                    sorted_hess,
                    &index[second_start..second_stop],
                );
            });
            NodeHistogram::from_parent_two_children(hist_tree, parent_num, first_num, second_num, update_num);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix;
    use crate::binning::bin_matrix;
    use crate::histogram::{
        FeatureHistogram, FeatureHistogramOwned, NodeHistogram, NodeHistogramOwned, update_histogram,
    };
    use crate::objective_functions::objective::{Objective, ObjectiveFunction};
    use approx::assert_relative_eq;
    use std::collections::HashSet;
    use std::fs;

    #[test]
    fn test_simple_histogram() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let nbins = 90;

        let data_vec: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();

        let data = Matrix::new(&data_vec, data_vec.len(), 1);

        let b = bin_matrix(&data, None, nbins, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (g, h) = objective_function.gradient(&y, &yhat, None, None);

        let col = 0;
        let mut hist_feat_owned = FeatureHistogramOwned::empty_from_cuts(b.cuts.get_col(col), false);
        let hist_feat = FeatureHistogram::new(&mut hist_feat_owned.data);
        unsafe { hist_feat.update(bdata.get_col(col), &g, h.as_deref(), &bdata.index) };

        let mut f = bdata.get_col(col).to_owned();

        println!("histogram:");
        println!("{:?}", hist_feat);
        println!("histogram.cuts:");
        println!(
            "{:?}",
            hist_feat
                .data
                .iter()
                .map(|b| unsafe { b.get().as_ref().unwrap().cut_value })
                .collect::<Vec<_>>()
        );
        println!("feature_data:");
        println!("{:?}", &data.get_col(col));
        println!("feature_data_bin_indices:");
        println!("{:?}", &bdata.get_col(col));
        println!("data_indices:");
        println!("{:?}", &bdata.index);
        println!("cuts:");
        println!("{:?}", &b.cuts.get_col(col));
        f.sort();
        f.dedup();
        println!("f:");
        println!("{:?}", &f);
        println!("{:?}", &f.len());
        println!("{:?}", &hist_feat.data.len());
        assert_eq!(f.len() + 1, hist_feat.data.len());
        println!("b.cuts:");
        println!("{:?}", &b.cuts);
        println!("b.nunique:");
        println!("{:?}", &b.nunique);
    }

    #[test]
    fn test_single_histogram() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

        let nbins = 10;

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let data = Matrix::new(&data_vec, 891, 5);
        let b = bin_matrix(&data, None, nbins, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (g, h) = objective_function.gradient(&y, &yhat, None, None);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut hist_init_owned = NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, true, false);
        let mut hist_init = NodeHistogram::from_owned(&mut hist_init_owned);

        let col = 1;

        unsafe {
            hist_init
                .data
                .get_mut(col)
                .unwrap()
                .update(bdata.get_col(col), &g, h.as_deref(), &bdata.index)
        };

        let mut f = bdata.get_col(col).to_owned();

        println!("histogram:");
        println!("{:?}", hist_init.data.get(col).unwrap());
        println!("feature_data:");
        println!("{:?}", &data.get_col(col));
        println!("feature_data_bin_indices:");
        println!("{:?}", &bdata.get_col(col));
        println!("data_indices:");
        println!("{:?}", &bdata.index);
        println!("cuts:");
        println!("{:?}", &b.cuts.get_col(col));
        f.sort();
        f.dedup();
        println!("f:");
        println!("{:?}", &f);
        println!("{:?}", &f.len());
        println!("{:?}", &hist_init.data.get(col).unwrap().data.len());
        assert_eq!(f.len() + 1, hist_init.data.get(col).unwrap().data.len());
    }

    #[test]
    fn test_histogram_categorical() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

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
        let b = bin_matrix(
            &data,
            None,
            256,
            f64::NAN,
            Some(&HashSet::from([0, 3, 4, 6, 7, 8, 10, 11])),
        )
        .unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let yhat = vec![0.5; y.len()];
        let (g, h) = objective_function.gradient(&y, &yhat, None, None);

        let col_index: Vec<usize> = (0..data.cols).collect();
        let mut hist_init_owned = NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, false, false);
        let hist_init = NodeHistogram::from_owned(&mut hist_init_owned);

        let col = 0;

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        update_histogram(
            &hist_init,
            0,
            bdata.index.len(),
            &bdata,
            &g,
            h.as_deref(),
            &bdata.index,
            &col_index,
            &pool,
            false,
        );

        let mut f = bdata.get_col(col).to_owned();

        println!("histogram:");
        println!("{:?}", unsafe { hist_init.data.get_unchecked(col) });
        println!("cuts:");
        println!("{:?}", &b.cuts.get_col(col));
        f.sort();
        f.dedup();
        println!("f:");
        println!("{:?}", &f);
        println!("{:?}", &f.len());
        println!("{:?}", unsafe { hist_init.data.get_unchecked(col) }.data.len());
        assert_eq!(f.len() + 1, unsafe { hist_init.data.get_unchecked(col) }.data.len());
    }

    #[test]
    fn test_histogram_parallel() {
        // instantiate objective function
        let objective_function = Objective::LogLoss;

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
        let b = bin_matrix(&data, None, 256, f64::NAN, Some(&HashSet::from([1]))).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let yhat = vec![0.5; y.len()];
        let (g, h) = objective_function.gradient(&y, &yhat, None, None);

        let col_index: Vec<usize> = (0..data.cols).collect();

        let mut hist_init_owned1 = NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, false, false);
        let hist_init1 = NodeHistogram::from_owned(&mut hist_init_owned1);

        let mut hist_init_owned2 = NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, false, false);
        let hist_init2 = NodeHistogram::from_owned(&mut hist_init_owned2);

        let col = 1;

        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let pool2 = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        update_histogram(
            &hist_init1,
            0,
            bdata.index.len(),
            &bdata,
            &g,
            h.as_deref(),
            &bdata.index,
            &col_index,
            &pool1,
            false,
        );
        update_histogram(
            &hist_init2,
            0,
            bdata.index.len(),
            &bdata,
            &g,
            h.as_deref(),
            &bdata.index,
            &col_index,
            &pool2,
            false,
        );

        let bins1 = unsafe { &hist_init_owned1.data.get_unchecked(col).data };
        let bins2 = unsafe { &hist_init_owned2.data.get_unchecked(col).data };

        println!("{:?}", bins1[0].g_folded);
        println!("{:?}", bins2[0].g_folded);

        bins1.iter().zip(bins2.iter()).for_each(|(b1, b2)| {
            b1.g_folded.iter().zip(b2.g_folded.iter()).for_each(|(g1, g2)| {
                assert_relative_eq!(g1, g2);
            });
            b1.h_folded.iter().zip(b2.h_folded.iter()).for_each(|(h1, h2)| {
                assert_relative_eq!(h1, h2);
            });
        });
    }
}
