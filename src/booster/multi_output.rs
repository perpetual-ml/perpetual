//! Multi-Output Booster
//!
//! A wrapper around multiple [`PerpetualBooster`] instances for multi-target
//! regression or classification tasks.
use crate::binning::bin_matrix;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::booster::config::MissingNodeTreatment;
use crate::booster::config::*;
use crate::constants::{ITER_LIMIT, N_NODES_ALLOC_MAX, N_NODES_ALLOC_MIN, STOPPING_ROUNDS};
use crate::constraints::ConstraintMap;
use crate::errors::PerpetualError;
use crate::histogram::{HistogramArena, NodeHistogram};
use crate::objective::Objective;
use crate::splitter::{MissingBranchSplitter, MissingImputerSplitter, SplitInfo, SplitInfoSlice, Splitter};
use crate::tree::core::Tree;
#[cfg(test)]
use crate::utils::odds;
use crate::{ColumnarMatrix, Matrix, PerpetualBooster};
use rayon::ThreadPoolBuilder;

/// Multi-Output Gradient Boosting Machine.
///
/// Wraps `n_boosters` independent [`PerpetualBooster`] instances — one per target column —
/// and exposes a unified `fit` / `predict` API.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiOutputBooster {
    /// Number of independent boosters (one per output).
    pub n_boosters: usize,
    /// Shared configuration applied to every sub-booster.
    pub cfg: BoosterConfig,
    /// The individual [`PerpetualBooster`] instances.
    pub boosters: Vec<PerpetualBooster>,
    /// Empirical class priors used for multiclass one-vs-rest coupling.
    #[serde(default)]
    pub class_priors: Vec<f64>,
    /// Whether the multiclass model was trained with a shared-logit native path.
    #[serde(default)]
    pub native_multiclass: bool,
    /// Arbitrary metadata key-value pairs.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Default for MultiOutputBooster {
    fn default() -> Self {
        let cfg = BoosterConfig::default();
        let n_boosters = 1;
        let boosters = vec![{
            PerpetualBooster {
                cfg: cfg.clone(),
                ..Default::default()
            }
        }];

        Self {
            n_boosters,
            cfg,
            boosters,
            class_priors: Vec::new(),
            native_multiclass: false,
            metadata: HashMap::new(),
        }
    }
}

impl MultiOutputBooster {
    #[cfg(test)]
    #[allow(dead_code)]
    const OVR_FRONTIER_EXPANSION_FACTOR: usize = 2;

    fn center_multiclass_leaf_values(values: &mut [f32]) {
        if values.is_empty() {
            return;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter_mut().for_each(|value| *value -= mean);
    }

    fn native_multiclass_candidate_limit(&self) -> usize {
        if self.n_boosters <= 8 {
            self.n_boosters
        } else if self.n_boosters <= 16 {
            6
        } else {
            8
        }
    }

    fn native_multiclass_candidate_classes(&self, class_difficulty: &[f64]) -> Vec<usize> {
        let mut ranked = class_difficulty
            .iter()
            .enumerate()
            .map(|(class_idx, &difficulty)| (class_idx, difficulty))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        ranked
            .into_iter()
            .take(self.native_multiclass_candidate_limit())
            .map(|(class_idx, _)| class_idx)
            .collect()
    }

    fn compute_native_multiclass_leaf_weights(
        &self,
        tree: &Tree,
        labels: &[usize],
        sample_weight: Option<&[f64]>,
        probabilities_by_class: &[Vec<f64>],
        leaf_regularization: f64,
        eta: f32,
    ) -> Vec<Vec<f32>> {
        let mut per_class_leaf_weights = vec![vec![0.0_f32; tree.leaf_node_assignments.len()]; self.n_boosters];
        for (leaf_position, &(_, start, stop)) in tree.leaf_node_assignments.iter().enumerate() {
            let dimension = self.n_boosters;
            let use_full_hessian = dimension <= 5;
            let mut gradient = vec![0.0_f64; dimension];
            let mut hessian = vec![0.0_f64; dimension * dimension];

            for &row_idx in &tree.train_index[start..stop] {
                let sample_weight_value = sample_weight.map_or(1.0, |weights| weights[row_idx]);
                for class_idx in 0..dimension {
                    let probability = probabilities_by_class[class_idx][row_idx];
                    let target = if labels[row_idx] == class_idx { 1.0 } else { 0.0 };
                    gradient[class_idx] += (probability - target) * sample_weight_value;
                    hessian[class_idx * dimension + class_idx] +=
                        probability * (1.0 - probability) * sample_weight_value;
                }

                if use_full_hessian {
                    for class_idx in 0..dimension {
                        let probability = probabilities_by_class[class_idx][row_idx];
                        for other_class_idx in (class_idx + 1)..dimension {
                            let cross_hessian =
                                -probability * probabilities_by_class[other_class_idx][row_idx] * sample_weight_value;
                            hessian[class_idx * dimension + other_class_idx] += cross_hessian;
                            hessian[other_class_idx * dimension + class_idx] += cross_hessian;
                        }
                    }
                }
            }

            let mut rhs = gradient.iter().map(|value| -value).collect::<Vec<_>>();
            for class_idx in 0..dimension {
                hessian[class_idx * dimension + class_idx] += leaf_regularization;
            }

            if use_full_hessian && Self::solve_dense_linear_system(&mut hessian, &mut rhs, dimension) {
                for class_idx in 0..dimension {
                    per_class_leaf_weights[class_idx][leaf_position] = rhs[class_idx] as f32 * eta;
                }
            } else {
                for class_idx in 0..dimension {
                    let diagonal = hessian[class_idx * dimension + class_idx].max(f64::EPSILON);
                    per_class_leaf_weights[class_idx][leaf_position] = (-gradient[class_idx] / diagonal) as f32 * eta;
                }
            }

            let mut leaf_values = per_class_leaf_weights
                .iter()
                .map(|weights| weights[leaf_position])
                .collect::<Vec<_>>();
            Self::center_multiclass_leaf_values(&mut leaf_values);
            for (class_idx, leaf_weight) in leaf_values.into_iter().enumerate() {
                per_class_leaf_weights[class_idx][leaf_position] = leaf_weight;
            }
        }

        per_class_leaf_weights
    }

    fn solve_dense_linear_system(matrix: &mut [f64], rhs: &mut [f64], dimension: usize) -> bool {
        for col in 0..dimension {
            let mut pivot_row = col;
            let mut pivot_abs = matrix[col * dimension + col].abs();
            for row in (col + 1)..dimension {
                let candidate_abs = matrix[row * dimension + col].abs();
                if candidate_abs > pivot_abs {
                    pivot_abs = candidate_abs;
                    pivot_row = row;
                }
            }

            if pivot_abs <= 1e-12 {
                return false;
            }

            if pivot_row != col {
                for inner in col..dimension {
                    matrix.swap(col * dimension + inner, pivot_row * dimension + inner);
                }
                rhs.swap(col, pivot_row);
            }

            let pivot = matrix[col * dimension + col];
            for row in (col + 1)..dimension {
                let factor = matrix[row * dimension + col] / pivot;
                if factor.abs() <= f64::EPSILON {
                    continue;
                }

                matrix[row * dimension + col] = 0.0;
                for inner in (col + 1)..dimension {
                    matrix[row * dimension + inner] -= factor * matrix[col * dimension + inner];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        for row in (0..dimension).rev() {
            let mut residual = rhs[row];
            for inner in (row + 1)..dimension {
                residual -= matrix[row * dimension + inner] * rhs[inner];
            }

            let diagonal = matrix[row * dimension + row];
            if diagonal.abs() <= 1e-12 {
                return false;
            }
            rhs[row] = residual / diagonal;
        }

        true
    }

    fn apply_multiclass_leaf_weights(tree: &mut Tree, leaf_weights: &[f32]) {
        tree.leaf_bounds.clear();

        for (leaf_position, &(leaf_node_idx, start, stop)) in tree.leaf_node_assignments.iter().enumerate() {
            let leaf_weight = leaf_weights[leaf_position];

            if let Some(node) = tree.nodes.get_mut(&leaf_node_idx) {
                node.weight_value = leaf_weight;
                node.leaf_weights = Some([leaf_weight; 5]);
                if let Some(stats) = &mut node.stats {
                    stats.weights = [leaf_weight; 5];
                }
            }

            tree.leaf_bounds.push((leaf_weight as f64, start, stop));
        }

        if !tree.nodes.is_empty() {
            Self::refresh_internal_node_weights(tree, 0);
        }
    }

    fn predict_leaf_positions(&self, tree: &Tree, data: &Matrix<f64>) -> Vec<usize> {
        let leaf_lookup = tree
            .leaf_node_assignments
            .iter()
            .enumerate()
            .map(|(leaf_position, (node_idx, _, _))| (*node_idx, leaf_position))
            .collect::<HashMap<usize, usize>>();
        let mut leaf_positions = vec![0; data.rows];

        for &row_idx in &data.index {
            let mut node_idx = 0;
            loop {
                let node = tree.nodes.get(&node_idx).expect("node must exist");
                if node.is_leaf {
                    leaf_positions[row_idx] = *leaf_lookup.get(&node_idx).expect("leaf position must exist");
                    break;
                }
                node_idx = node.get_child_idx(data.get(row_idx, node.split_feature), &self.cfg.missing);
            }
        }

        leaf_positions
    }

    fn native_multiclass_candidate_loss(
        &self,
        leaf_positions: &[usize],
        labels: &[usize],
        sample_weight: Option<&[f64]>,
        scores_by_class: &[Vec<f64>],
        per_class_leaf_weights: &[Vec<f32>],
        weight_denom: f32,
    ) -> f32 {
        let mut total_loss = 0.0_f32;

        for row_idx in 0..leaf_positions.len() {
            let leaf_position = leaf_positions[row_idx];
            let mut max_logit = f64::NEG_INFINITY;
            for class_idx in 0..self.n_boosters {
                let logit =
                    scores_by_class[class_idx][row_idx] + per_class_leaf_weights[class_idx][leaf_position] as f64;
                max_logit = max_logit.max(logit);
            }

            let mut normalizer = 0.0_f64;
            let mut true_probability = 0.0_f64;
            for class_idx in 0..self.n_boosters {
                let logit =
                    scores_by_class[class_idx][row_idx] + per_class_leaf_weights[class_idx][leaf_position] as f64;
                let probability = (logit - max_logit).exp();
                normalizer += probability;
                if class_idx == labels[row_idx] {
                    true_probability = probability;
                }
            }

            let true_probability = (true_probability / normalizer.max(f64::EPSILON)).clamp(f64::EPSILON, 1.0);
            let sample_weight_value = sample_weight.map_or(1.0, |weights| weights[row_idx]);
            total_loss += (-true_probability.ln() * sample_weight_value) as f32;
        }

        total_loss / weight_denom
    }

    fn update_scores_with_leaf_positions(
        scores_by_class: &mut [Vec<f64>],
        leaf_positions: &[usize],
        per_class_leaf_weights: &[Vec<f32>],
    ) {
        for (class_idx, class_scores) in scores_by_class.iter_mut().enumerate() {
            for (row_idx, score) in class_scores.iter_mut().enumerate() {
                *score += per_class_leaf_weights[class_idx][leaf_positions[row_idx]] as f64;
            }
        }
    }

    fn is_multiclass_logloss(&self) -> bool {
        matches!(self.cfg.objective, Objective::LogLoss) && self.n_boosters > 1
    }

    fn update_class_priors(&mut self, y: &Matrix<f64>) {
        if !self.is_multiclass_logloss() {
            self.class_priors.clear();
            return;
        }

        self.class_priors = (0..self.n_boosters)
            .map(|idx| y.get_col(idx).iter().sum::<f64>() / y.rows.max(1) as f64)
            .collect();
    }

    fn build_multiclass_sample_weight(&self, y_col: &[f64], sample_weight: Option<&[f64]>) -> Option<Vec<f64>> {
        if !self.is_multiclass_logloss() || y_col.len() < 2 {
            return sample_weight.map(|weights| weights.to_vec());
        }

        // OVR LogLoss submodels already receive binary class balancing inside
        // PerpetualBooster::fit, so avoid double-weighting positives here.
        sample_weight.map(|weights| weights.to_vec())
    }

    fn multiclass_class_balance_strength(class_counts: &[usize]) -> f64 {
        if class_counts.len() <= 1 {
            return 0.0;
        }

        let total = class_counts.iter().sum::<usize>() as f64;
        if total <= 0.0 {
            return 0.0;
        }

        let class_count = class_counts.len() as f64;
        let mut entropy = 0.0;
        let mut majority = 0.0_f64;
        let mut minority = f64::INFINITY;

        for &count in class_counts {
            if count == 0 {
                continue;
            }

            let proportion = count as f64 / total;
            entropy -= proportion * proportion.ln();
            majority = majority.max(count as f64);
            minority = minority.min(count as f64);
        }

        if !minority.is_finite() || majority <= 0.0 {
            return 0.0;
        }

        let normalized_entropy = (entropy / class_count.ln().max(f64::EPSILON)).clamp(0.0, 1.0);
        let entropy_gap = 1.0 - normalized_entropy;
        let ratio_pressure = 1.0 - (minority / majority).clamp(0.0, 1.0);

        (0.75 * entropy_gap + 0.25 * ratio_pressure).clamp(0.0, 1.0)
    }

    fn effective_number_weight(class_count: usize, beta: f64) -> f64 {
        if class_count <= 1 {
            return 1.0;
        }

        let denominator = 1.0 - beta.powf(class_count as f64);
        if denominator <= f64::EPSILON {
            1.0
        } else {
            (1.0 - beta) / denominator
        }
    }

    fn build_native_multiclass_sample_weight(
        &self,
        labels: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        if !self.is_multiclass_logloss() || labels.len() < 2 {
            return sample_weight.map(|weights| weights.to_vec());
        }

        let mut class_counts = vec![0_usize; self.n_boosters];
        for &label in labels {
            if let Some(count) = class_counts.get_mut(label) {
                *count += 1;
            }
        }

        let active_class_counts = class_counts
            .iter()
            .copied()
            .filter(|&count| count > 0)
            .collect::<Vec<_>>();
        let balance_strength = Self::multiclass_class_balance_strength(&active_class_counts);
        if balance_strength <= 1e-6 {
            return sample_weight.map(|weights| weights.to_vec());
        }

        let beta = ((labels.len().saturating_sub(1)) as f64 / labels.len().max(1) as f64).clamp(0.9, 0.999_99);
        let normalization = labels.len() as f64
            / class_counts
                .iter()
                .copied()
                .filter(|&count| count > 0)
                .map(|count| count as f64 * Self::effective_number_weight(count, beta))
                .sum::<f64>()
                .max(f64::EPSILON);

        Some(
            labels
                .iter()
                .enumerate()
                .map(|(idx, &label)| {
                    let base_weight = sample_weight.map_or(1.0, |weights| weights[idx]);
                    let class_weight = class_counts
                        .get(label)
                        .copied()
                        .filter(|&count| count > 0)
                        .map(|count| Self::effective_number_weight(count, beta) * normalization)
                        .unwrap_or(1.0);
                    let blended_weight = 1.0 + balance_strength * (class_weight - 1.0);
                    base_weight * blended_weight
                })
                .collect(),
        )
    }

    fn categorical_feature_count(&self) -> usize {
        self.cfg.categorical_features.as_ref().map_or(0, HashSet::len)
    }

    fn is_categorical_heavy_task(&self, cols: usize) -> bool {
        let categorical_count = self.categorical_feature_count() as f32;
        categorical_count > 0.0 && categorical_count / cols.max(1) as f32 >= 0.35
    }

    fn multiclass_eta_power_for_training(&self, budget: f32, rows: usize, cols: usize) -> f32 {
        let budget = f32::max(0.0, budget);
        if budget <= 1.0 {
            budget
        } else if self.is_categorical_heavy_task(cols) {
            if rows >= 50_000 {
                1.0 + 0.15 * (budget - 1.0)
            } else {
                budget
            }
        } else if cols <= 96 {
            1.0 + 0.65 * (budget - 1.0)
        } else {
            budget
        }
    }

    fn multiclass_leaf_regularization(&self, rows: usize, cols: usize) -> f32 {
        if rows < 32 {
            return 0.0;
        }

        let row_feature_ratio = (rows as f32 / cols.max(1) as f32).max(1.0);
        let density_scale = row_feature_ratio.ln_1p();
        let categorical_ratio = self.categorical_feature_count() as f32 / cols.max(1) as f32;
        let budget_relief = self.cfg.budget.max(0.75).powf(0.35);
        let base_regularization = 0.015 + 0.012 * density_scale + 0.02 * categorical_ratio;

        (base_regularization / budget_relief).clamp(0.0, 0.12)
    }

    fn native_multiclass_categorical_generalization_min_folds(&self, rows: usize, cols: usize) -> u8 {
        if self.is_multiclass_logloss()
            && self.n_boosters == 3
            && rows >= 2_000
            && cols >= 512
            && self.categorical_feature_count() >= cols.saturating_mul(3) / 4
        {
            3
        } else {
            5
        }
    }

    fn native_multiclass_target_loss_decrement(&self, loss_avg: f32) -> f32 {
        let base = 10.0_f32;
        let effective_budget = self.cfg.budget.max(0.1);
        let n = base / effective_budget;
        let reciprocals_of_powers = n / (n - 1.0);
        let truncated_series_sum = reciprocals_of_powers - (1.0 + 1.0 / n);
        let c = 1.0 / n - truncated_series_sum;
        let tree_budget = effective_budget.clamp(0.0, 3.0);

        c * base.powf(-tree_budget) * loss_avg.max(f32::EPSILON) * 1.2
    }

    fn should_use_native_multiclass(&self, rows: usize, cols: usize) -> bool {
        let rows_per_class = rows / self.n_boosters.max(1);
        let categorical_features = self.categorical_feature_count();
        cols >= 96
            || (rows >= 10_000 && (cols >= 24 || categorical_features >= 8 || self.n_boosters >= 6))
            || (self.n_boosters >= 12 && rows >= 2_000)
            || (self.n_boosters >= 6 && rows_per_class >= 24 && cols <= 10)
            || (self.n_boosters >= 6 && rows_per_class >= 48 && cols >= 16)
            || ((4..=8).contains(&self.n_boosters) && rows_per_class >= 96 && cols >= 24)
    }

    fn ovr_multiclass_budget(&self, rows: usize, cols: usize) -> f32 {
        if self.is_multiclass_logloss() && self.n_boosters == 3 && rows >= 50_000 && cols <= 24 {
            self.cfg.budget.min(1.0)
        } else {
            self.cfg.budget
        }
    }

    #[cfg(test)]
    fn ovr_multiclass_probability_alpha(&self) -> f64 {
        if self.n_boosters == 3 {
            0.25
        } else if (4..=5).contains(&self.n_boosters) {
            0.5
        } else {
            0.0
        }
    }

    #[cfg(test)]
    fn ovr_multiclass_probability_beta(&self) -> f64 {
        0.25
    }

    #[cfg(test)]
    fn normalize_ovr_multiclass_probabilities(&self, probabilities: &mut [f64]) {
        let alpha = self.ovr_multiclass_probability_alpha();
        let beta = self.ovr_multiclass_probability_beta();

        probabilities.iter_mut().enumerate().for_each(|(idx, value)| {
            let clipped = value.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
            let prior = self
                .class_priors
                .get(idx)
                .copied()
                .unwrap_or(1.0)
                .clamp(f64::EPSILON, 1.0);
            *value = clipped / (1.0 - clipped).powf(alpha) / prior.powf(beta);
        });

        let sum = probabilities.iter().sum::<f64>();
        if sum <= f64::EPSILON {
            let uniform = 1.0 / probabilities.len().max(1) as f64;
            probabilities.iter_mut().for_each(|value| *value = uniform);
        } else {
            probabilities.iter_mut().for_each(|value| *value /= sum);
        }
    }

    #[cfg(test)]
    #[inline]
    fn ovr_virtual_fold(row_idx: usize) -> usize {
        let mut mixed = (row_idx as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        (mixed % 5) as usize
    }

    #[cfg(test)]
    fn ovr_multiclass_loss_from_fold_logits(
        &self,
        fold_logits_by_class: &[Vec<[f64; 5]>],
        row_index: &[usize],
        labels: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> f32 {
        let weight_denom = sample_weight
            .map(|weights| weights.iter().sum::<f64>() as f32)
            .unwrap_or(labels.len() as f32)
            .max(f32::EPSILON);
        let mut total_loss = 0.0_f32;

        for row_idx in 0..labels.len() {
            let mut probabilities = fold_logits_by_class
                .iter()
                .map(|class_logits| {
                    let fold = Self::ovr_virtual_fold(row_index[row_idx]);
                    odds(class_logits[row_idx][fold])
                })
                .collect::<Vec<_>>();
            self.normalize_ovr_multiclass_probabilities(&mut probabilities);

            let sample_weight_value = sample_weight.map_or(1.0, |weights| weights[row_idx]);
            total_loss +=
                (-(probabilities[labels[row_idx]].clamp(f64::EPSILON, 1.0)).ln() * sample_weight_value) as f32;
        }

        total_loss / weight_denom
    }

    #[cfg(test)]
    fn select_ovr_tree_prefix(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
    ) -> Option<(usize, usize)> {
        if !self.is_multiclass_logloss() || self.native_multiclass || self.n_boosters <= 2 {
            return None;
        }

        let min_tree_count = self
            .boosters
            .iter()
            .map(|booster| booster.trees.len())
            .min()
            .unwrap_or(0);
        if min_tree_count == 0 {
            return None;
        }

        let labels = self.multiclass_labels(y);
        let mut fold_logits_by_class = self
            .boosters
            .iter()
            .map(|booster| vec![[booster.base_score; 5]; data.rows])
            .collect::<Vec<_>>();

        let mut best_tree_count = 0_usize;
        let mut best_loss =
            self.ovr_multiclass_loss_from_fold_logits(&fold_logits_by_class, &data.index, &labels, sample_weight);

        for tree_idx in 0..min_tree_count {
            for (class_idx, booster) in self.boosters.iter().enumerate() {
                let tree_weights = booster.trees[tree_idx].predict_weights(data, false, &booster.cfg.missing);
                for (row_logits, row_weights) in fold_logits_by_class[class_idx].iter_mut().zip(tree_weights.iter()) {
                    for fold_idx in 0..5 {
                        row_logits[fold_idx] += row_weights[fold_idx] as f64;
                    }
                }
            }

            let loss =
                self.ovr_multiclass_loss_from_fold_logits(&fold_logits_by_class, &data.index, &labels, sample_weight);
            if loss + 1e-7 < best_loss {
                best_loss = loss;
                best_tree_count = tree_idx + 1;
            }
        }

        for booster in &mut self.boosters {
            if booster.trees.len() > best_tree_count {
                booster.trees.truncate(best_tree_count);
            }
        }

        Some((best_tree_count, min_tree_count))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn select_ovr_tree_prefix_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
    ) -> Option<(usize, usize)> {
        if !self.is_multiclass_logloss() || self.native_multiclass || self.n_boosters <= 2 {
            return None;
        }

        let min_tree_count = self
            .boosters
            .iter()
            .map(|booster| booster.trees.len())
            .min()
            .unwrap_or(0);
        if min_tree_count == 0 {
            return None;
        }

        let labels = self.multiclass_labels(y);
        let mut fold_logits_by_class = self
            .boosters
            .iter()
            .map(|booster| vec![[booster.base_score; 5]; data.rows])
            .collect::<Vec<_>>();

        let mut best_tree_count = 0_usize;
        let mut best_loss =
            self.ovr_multiclass_loss_from_fold_logits(&fold_logits_by_class, &data.index, &labels, sample_weight);

        for tree_idx in 0..min_tree_count {
            for (class_idx, booster) in self.boosters.iter().enumerate() {
                let tree_weights = booster.trees[tree_idx].predict_weights_columnar(data, false, &booster.cfg.missing);
                for (row_logits, row_weights) in fold_logits_by_class[class_idx].iter_mut().zip(tree_weights.iter()) {
                    for fold_idx in 0..5 {
                        row_logits[fold_idx] += row_weights[fold_idx] as f64;
                    }
                }
            }

            let loss =
                self.ovr_multiclass_loss_from_fold_logits(&fold_logits_by_class, &data.index, &labels, sample_weight);
            if loss + 1e-7 < best_loss {
                best_loss = loss;
                best_tree_count = tree_idx + 1;
            }
        }

        for booster in &mut self.boosters {
            if booster.trees.len() > best_tree_count {
                booster.trees.truncate(best_tree_count);
            }
        }

        Some((best_tree_count, min_tree_count))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn maybe_continue_ovr_training(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        effective_budget: f32,
    ) -> Result<(), PerpetualError> {
        let Some((best_tree_count, frontier_tree_count)) = self.select_ovr_tree_prefix(data, y, sample_weight) else {
            return Ok(());
        };
        if best_tree_count != frontier_tree_count || frontier_tree_count == 0 {
            return Ok(());
        }

        let expanded_iteration_limit = Some(
            self.cfg
                .iteration_limit
                .unwrap_or(frontier_tree_count * Self::OVR_FRONTIER_EXPANSION_FACTOR)
                .max(frontier_tree_count * Self::OVR_FRONTIER_EXPANSION_FACTOR),
        );

        for i in 0..self.n_boosters {
            self.boosters[i].cfg.budget = effective_budget;
            self.boosters[i].cfg.auto_class_weights = false;
            self.boosters[i].cfg.iteration_limit = expanded_iteration_limit;
            self.boosters[i].cfg.reset = Some(true);
            let y_col = y.get_col(i);
            let adjusted_weight = self.build_multiclass_sample_weight(y_col, sample_weight);
            self.boosters[i].fit(data, y_col, adjusted_weight.as_deref().or(sample_weight), group)?;
            self.boosters[i].cfg.iteration_limit = self.cfg.iteration_limit;
            self.boosters[i].cfg.reset = self.cfg.reset;
        }
        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn maybe_continue_ovr_training_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        effective_budget: f32,
    ) -> Result<(), PerpetualError> {
        let Some((best_tree_count, frontier_tree_count)) = self.select_ovr_tree_prefix_columnar(data, y, sample_weight)
        else {
            return Ok(());
        };
        if best_tree_count != frontier_tree_count || frontier_tree_count == 0 {
            return Ok(());
        }

        let expanded_iteration_limit = Some(
            self.cfg
                .iteration_limit
                .unwrap_or(frontier_tree_count * Self::OVR_FRONTIER_EXPANSION_FACTOR)
                .max(frontier_tree_count * Self::OVR_FRONTIER_EXPANSION_FACTOR),
        );

        for i in 0..self.n_boosters {
            self.boosters[i].cfg.budget = effective_budget;
            self.boosters[i].cfg.auto_class_weights = false;
            self.boosters[i].cfg.iteration_limit = expanded_iteration_limit;
            self.boosters[i].cfg.reset = Some(true);
            let y_col = y.get_col(i);
            let adjusted_weight = self.build_multiclass_sample_weight(y_col, sample_weight);
            self.boosters[i].fit_columnar(data, y_col, adjusted_weight.as_deref().or(sample_weight), group)?;
            self.boosters[i].cfg.iteration_limit = self.cfg.iteration_limit;
            self.boosters[i].cfg.reset = self.cfg.reset;
        }
        Ok(())
    }

    fn multiclass_base_scores(&self) -> Vec<f64> {
        if self.class_priors.is_empty() {
            return vec![0.0; self.n_boosters];
        }

        let mut scores = self
            .class_priors
            .iter()
            .map(|prior| prior.clamp(f64::EPSILON, 1.0).ln())
            .collect::<Vec<_>>();
        let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        scores.iter_mut().for_each(|score| *score -= mean);
        scores
    }

    fn multiclass_labels(&self, y: &Matrix<f64>) -> Vec<usize> {
        (0..y.rows)
            .map(|row| {
                (0..self.n_boosters)
                    .find(|&class_idx| *y.get(row, class_idx) >= 0.5)
                    .unwrap_or(0)
            })
            .collect()
    }

    fn refresh_internal_node_weights(tree: &mut Tree, node_idx: usize) -> (f32, f32) {
        let (is_leaf, left_child, right_child, missing_node, node_hessian) = {
            let node = tree.nodes.get(&node_idx).expect("node must exist");
            (
                node.is_leaf,
                node.left_child,
                node.right_child,
                node.missing_node,
                node.hessian_sum.max(f32::EPSILON),
            )
        };

        if is_leaf {
            let weight = tree.nodes.get(&node_idx).map(|node| node.weight_value).unwrap_or(0.0);
            return (weight, node_hessian);
        }

        let mut weighted_sum = 0.0_f32;
        let mut total_cover = 0.0_f32;
        for child_idx in [Some(left_child), Some(right_child)]
            .into_iter()
            .flatten()
            .chain((missing_node != left_child && missing_node != right_child).then_some(missing_node))
        {
            let (child_weight, child_cover) = Self::refresh_internal_node_weights(tree, child_idx);
            weighted_sum += child_weight * child_cover;
            total_cover += child_cover;
        }

        let new_weight = if total_cover > f32::EPSILON {
            weighted_sum / total_cover
        } else {
            0.0
        };

        if let Some(node) = tree.nodes.get_mut(&node_idx) {
            node.weight_value = new_weight;
            node.leaf_weights = None;
            if let Some(stats) = &mut node.stats {
                stats.weights = [new_weight; 5];
            }
        }

        (new_weight, total_cover.max(node_hessian))
    }

    fn fit_native_multiclass_with_splitter<T: Splitter>(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        splitter: &T,
    ) -> Result<(), PerpetualError> {
        let n_threads_available = std::thread::available_parallelism().unwrap().get();
        let num_threads = self.cfg.num_threads.unwrap_or(n_threads_available);
        let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

        let binned_data = bin_matrix(
            data,
            sample_weight,
            self.cfg.max_bin,
            self.cfg.missing,
            self.cfg.categorical_features.as_ref(),
        )?;
        let bdata = Matrix::new(&binned_data.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let n_nodes_alloc = (data.rows * 2).clamp(N_NODES_ALLOC_MIN, N_NODES_ALLOC_MAX);
        let mut hist_arena = HistogramArena::from_cuts(&binned_data.cuts, &col_index, false, n_nodes_alloc);
        let mut hist_tree: Vec<NodeHistogram> = hist_arena.as_node_histograms();
        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let mut split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        let labels = self.multiclass_labels(y);
        let adjusted_sample_weight = self.build_native_multiclass_sample_weight(&labels, sample_weight);
        let sample_weight = adjusted_sample_weight.as_deref().or(sample_weight);
        let base_scores = self.multiclass_base_scores();
        let mut scores_by_class = base_scores
            .iter()
            .map(|&base_score| vec![base_score; data.rows])
            .collect::<Vec<_>>();
        let mut probabilities_by_class = vec![vec![0.0; data.rows]; self.n_boosters];
        let mut multiclass_loss = vec![0.0_f32; data.rows];
        let weight_denom = sample_weight
            .map(|weights| weights.iter().sum::<f64>() as f32)
            .unwrap_or(data.rows as f32)
            .max(f32::EPSILON);

        for (booster, &base_score) in self.boosters.iter_mut().zip(base_scores.iter()) {
            booster.cfg.objective = Objective::LogLoss;
            booster.base_score = base_score;
            booster.eta = splitter.get_eta();
            booster.trees.clear();
        }

        let mut best_loss = f32::INFINITY;
        let mut best_tree_count = 0_usize;
        let mut no_improvement_rounds = 0_usize;
        let effective_iteration_limit = self
            .boosters
            .first()
            .map(|booster| booster.effective_iteration_limit(data.rows, data.cols))
            .unwrap_or(ITER_LIMIT);
        let effective_stopping_rounds = self
            .boosters
            .first()
            .map(|booster| booster.effective_stopping_rounds(data.rows, data.cols))
            .unwrap_or(STOPPING_ROUNDS)
            .max(1);
        let leaf_regularization = splitter.get_leaf_regularization() as f64 + f64::EPSILON;

        for _round in 0..effective_iteration_limit {
            let mut class_difficulty = vec![0.0_f64; self.n_boosters];
            for row in 0..data.rows {
                let max_score = scores_by_class
                    .iter()
                    .map(|scores| scores[row])
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut normalizer = 0.0_f64;
                for class_idx in 0..self.n_boosters {
                    let value = (scores_by_class[class_idx][row] - max_score).exp();
                    probabilities_by_class[class_idx][row] = value;
                    normalizer += value;
                }

                let label = labels[row];
                let sample_weight_value = sample_weight.map_or(1.0, |weights| weights[row]);
                let mut true_probability = f64::EPSILON;
                let mut dominant_abs_grad = -1.0_f64;
                for class_idx in 0..self.n_boosters {
                    let probability = (probabilities_by_class[class_idx][row] / normalizer).clamp(f64::EPSILON, 1.0);
                    probabilities_by_class[class_idx][row] = probability;
                    let target = if class_idx == label { 1.0 } else { 0.0 };
                    let class_grad = probability - target;
                    let class_abs_grad = class_grad.abs();
                    if class_abs_grad > dominant_abs_grad {
                        dominant_abs_grad = class_abs_grad;
                    }
                    class_difficulty[class_idx] += class_abs_grad * sample_weight_value;
                    if class_idx == label {
                        true_probability = probability;
                    }
                }

                multiclass_loss[row] = (-true_probability.ln() * sample_weight_value) as f32;
            }

            let candidate_classes = self.native_multiclass_candidate_classes(&class_difficulty);
            let mut best_round_tree: Option<Tree> = None;
            let mut best_round_leaf_positions: Option<Vec<usize>> = None;
            let mut best_round_leaf_weights: Option<Vec<Vec<f32>>> = None;
            let mut best_round_loss = f32::INFINITY;

            for &candidate_class in &candidate_classes {
                let mut candidate_grad = vec![0.0_f32; data.rows];
                let mut candidate_hess = vec![0.0_f32; data.rows];
                let mut candidate_loss = vec![0.0_f32; data.rows];
                for row in 0..data.rows {
                    let target = if labels[row] == candidate_class { 1.0 } else { 0.0 };
                    let probability = probabilities_by_class[candidate_class][row];
                    let sample_weight_value = sample_weight.map_or(1.0, |weights| weights[row]);
                    candidate_grad[row] = ((probability - target) * sample_weight_value) as f32;
                    candidate_hess[row] =
                        (probability * (1.0 - probability)).max(f64::EPSILON) as f32 * sample_weight_value as f32;
                    candidate_loss[row] = (-(target * probability.ln()
                        + (1.0 - target) * (1.0 - probability).clamp(f64::EPSILON, 1.0).ln())
                        * sample_weight_value) as f32;
                }
                let candidate_loss_avg = candidate_loss.iter().sum::<f32>() / weight_denom;
                let target_loss_decrement = (data.rows >= 2_000 || data.cols >= 96)
                    .then(|| self.native_multiclass_target_loss_decrement(candidate_loss_avg));

                let mut candidate_tree = Tree::new();
                candidate_tree.fit(
                    &Objective::LogLoss,
                    &bdata,
                    data.index.to_owned(),
                    &col_index,
                    &mut candidate_grad,
                    Some(candidate_hess.as_mut_slice()),
                    splitter,
                    &pool,
                    target_loss_decrement,
                    &candidate_loss,
                    y.get_col(candidate_class),
                    &scores_by_class[candidate_class],
                    sample_weight,
                    None,
                    false,
                    &mut hist_tree,
                    self.cfg.categorical_features.as_ref(),
                    false,
                    &mut split_info_slice,
                    n_nodes_alloc,
                    self.cfg.save_node_stats,
                );

                if candidate_tree.leaf_node_assignments.is_empty() {
                    continue;
                }

                let candidate_leaf_weights = self.compute_native_multiclass_leaf_weights(
                    &candidate_tree,
                    &labels,
                    sample_weight,
                    &probabilities_by_class,
                    leaf_regularization,
                    splitter.get_eta(),
                );
                let candidate_leaf_positions = self.predict_leaf_positions(&candidate_tree, data);
                let candidate_round_loss = self.native_multiclass_candidate_loss(
                    &candidate_leaf_positions,
                    &labels,
                    sample_weight,
                    &scores_by_class,
                    &candidate_leaf_weights,
                    weight_denom,
                );

                if candidate_round_loss + 1e-7 < best_round_loss {
                    best_round_loss = candidate_round_loss;
                    best_round_tree = Some(candidate_tree);
                    best_round_leaf_positions = Some(candidate_leaf_positions);
                    best_round_leaf_weights = Some(candidate_leaf_weights);
                }
            }

            let Some(tree) = best_round_tree else {
                break;
            };
            let Some(leaf_positions) = best_round_leaf_positions else {
                break;
            };
            let Some(per_class_leaf_weights) = best_round_leaf_weights else {
                break;
            };

            let current_loss = self.native_multiclass_candidate_loss(
                &leaf_positions,
                &labels,
                sample_weight,
                &scores_by_class,
                &per_class_leaf_weights,
                weight_denom,
            );
            Self::update_scores_with_leaf_positions(&mut scores_by_class, &leaf_positions, &per_class_leaf_weights);

            for (booster, leaf_weights) in self.boosters.iter_mut().zip(per_class_leaf_weights.iter()) {
                let mut class_tree = tree.clone();
                Self::apply_multiclass_leaf_weights(&mut class_tree, leaf_weights);
                class_tree.leaf_node_assignments.clear();
                class_tree.train_index.clear();
                class_tree.leaf_bounds.clear();
                booster.trees.push(class_tree);
            }

            if current_loss + 1e-7 < best_loss {
                best_loss = current_loss;
                best_tree_count = self.boosters.first().map(|booster| booster.trees.len()).unwrap_or(0);
                no_improvement_rounds = 0;
            } else {
                no_improvement_rounds += 1;
            }

            if tree.nodes.len() == 1 || no_improvement_rounds >= effective_stopping_rounds {
                break;
            }
        }

        if best_tree_count > 0 {
            for booster in &mut self.boosters {
                if booster.trees.len() > best_tree_count {
                    booster.trees.truncate(best_tree_count);
                }
            }
        }

        Ok(())
    }

    fn fit_native_multiclass(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(), PerpetualError> {
        let constraints_map = self
            .cfg
            .monotone_constraints
            .as_ref()
            .unwrap_or(&ConstraintMap::new())
            .to_owned();
        let eta = 10_f32.powf(-self.multiclass_eta_power_for_training(self.cfg.budget, data.rows, data.cols));
        let leaf_regularization = self.multiclass_leaf_regularization(data.rows, data.cols);
        let categorical_generalization_min_folds =
            self.native_multiclass_categorical_generalization_min_folds(data.rows, data.cols);

        if self.cfg.create_missing_branch {
            let splitter = MissingBranchSplitter::new(
                eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.terminate_missing_features.clone(),
                self.cfg.missing_node_treatment,
                self.cfg.force_children_to_bound_parent,
            )
            .with_categorical_generalization_min_folds(categorical_generalization_min_folds);
            self.fit_native_multiclass_with_splitter(data, y, sample_weight, &splitter)
        } else {
            let splitter = MissingImputerSplitter::new(
                eta,
                leaf_regularization,
                self.cfg.allow_missing_splits,
                constraints_map,
                self.cfg.interaction_constraints.clone(),
            )
            .with_categorical_generalization_min_folds(categorical_generalization_min_folds);
            self.fit_native_multiclass_with_splitter(data, y, sample_weight, &splitter)
        }
    }

    /// Create a new `MultiOutputBooster`.
    ///
    /// * `n_boosters` - Number of independent boosters (one per target column).
    /// * `objective` - The name of objective function used to optimize. Valid options are:
    ///   "LogLoss" to use logistic loss as the objective function,
    ///   "SquaredLoss" to use Squared Error as the objective function,
    ///   "QuantileLoss" for quantile regression.
    ///   "AdaptiveHuberLoss" for adaptive huber loss regression.
    ///   "HuberLoss" for huber loss regression.
    ///   "ListNetLoss" for listnet loss ranking.
    /// * `budget` - budget to fit the model.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    /// * `num_threads` - Number of threads to be used during training
    /// * `monotone_constraints` - Constraints that are used to enforce a specific relationship
    ///   between the training features and the target variable.
    /// * `force_children_to_bound_parent` - force_children_to_bound_parent.
    /// * `missing` - Value to consider missing.
    /// * `allow_missing_splits` - Whether the algorithm allows splits that completely separate
    ///   missing and non-missing values. When `create_missing_branch` is true, setting this to
    ///   true will result in the missing branch being further split.
    /// * `create_missing_branch` - Should missing be split out into its own separate branch?
    /// * `missing_node_treatment` - specify how missing nodes should be handled during training.
    /// * `log_iterations` - Setting to a value (N) other than zero will result in information being logged about ever N iterations.
    /// * `seed` - Integer value used to seed any randomness used in the algorithm.
    /// * `reset` - Reset the model or continue training.
    /// * `categorical_features` - categorical features.
    /// * `timeout` - fit timeout limit in seconds.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    /// * `memory_limit` - optional limit for memory allocation.
    /// * `stopping_rounds` - optional limit for auto stopping rounds.
    /// * `save_node_stats` - whether to save node statistics during training.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_boosters: usize,
        objective: Objective,
        budget: f32,
        max_bin: u16,
        num_threads: Option<usize>,
        monotone_constraints: Option<ConstraintMap>,
        interaction_constraints: Option<Vec<Vec<usize>>>,
        force_children_to_bound_parent: bool,
        missing: f64,
        allow_missing_splits: bool,
        create_missing_branch: bool,
        terminate_missing_features: HashSet<usize>,
        missing_node_treatment: MissingNodeTreatment,
        log_iterations: usize,
        seed: u64,
        reset: Option<bool>,
        categorical_features: Option<HashSet<usize>>,
        timeout: Option<f32>,
        iteration_limit: Option<usize>,
        memory_limit: Option<f32>,
        stopping_rounds: Option<usize>,
        save_node_stats: bool,
    ) -> Result<Self, PerpetualError> {
        // Build the common configuration object.
        let cfg = BoosterConfig {
            objective: objective.clone(),
            budget,
            max_bin,
            num_threads,
            monotone_constraints: monotone_constraints.clone(),
            interaction_constraints: interaction_constraints.clone(),
            force_children_to_bound_parent,
            missing,
            allow_missing_splits,
            create_missing_branch,
            terminate_missing_features: terminate_missing_features.clone(),
            missing_node_treatment,
            log_iterations,
            seed,
            reset,
            categorical_features: categorical_features.clone(),
            timeout: timeout.map(|t| t / n_boosters.max(1) as f32),
            iteration_limit,
            memory_limit,
            stopping_rounds,
            auto_class_weights: true,
            save_node_stats,
            calibration_method: CalibrationMethod::default(),
        };

        // Base booster template that child boosters will clone.
        let template_booster = PerpetualBooster {
            cfg: cfg.clone(),
            ..Default::default()
        };
        template_booster.validate_parameters()?;

        // Assemble the wrapper with `n_boosters` copies.
        let boosters = vec![template_booster; n_boosters.max(1)];

        Ok(MultiOutputBooster {
            n_boosters: n_boosters.max(1),
            cfg,
            boosters,
            class_priors: Vec::new(),
            native_multiclass: false,
            metadata: HashMap::new(),
        })
    }

    /// Fit the multi-output booster.
    pub fn fit(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        self.update_class_priors(y);
        if self.is_multiclass_logloss()
            && self.n_boosters > 2
            && group.is_none()
            && self.should_use_native_multiclass(data.rows, data.cols)
        {
            self.native_multiclass = true;
            return self.fit_native_multiclass(data, y, sample_weight);
        }

        self.native_multiclass = false;
        let effective_budget = self.ovr_multiclass_budget(data.rows, data.cols);
        for i in 0..self.n_boosters {
            self.boosters[i].cfg.budget = effective_budget;
            self.boosters[i].cfg.auto_class_weights = false;
            self.boosters[i].cfg.iteration_limit = self.cfg.iteration_limit;
            let y_col = y.get_col(i);
            let adjusted_weight = self.build_multiclass_sample_weight(y_col, sample_weight);
            self.boosters[i].fit(data, y_col, adjusted_weight.as_deref().or(sample_weight), group)?;
        }
        Ok(())
    }

    /// Fit the multi-output booster on columnar data.
    pub fn fit_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        self.update_class_priors(y);
        self.native_multiclass = false;
        let effective_budget = self.ovr_multiclass_budget(data.rows, data.cols);
        for i in 0..self.n_boosters {
            self.boosters[i].cfg.budget = effective_budget;
            self.boosters[i].cfg.auto_class_weights = false;
            self.boosters[i].cfg.iteration_limit = self.cfg.iteration_limit;
            let y_col = y.get_col(i);
            let adjusted_weight = self.build_multiclass_sample_weight(y_col, sample_weight);
            self.boosters[i].fit_columnar(data, y_col, adjusted_weight.as_deref().or(sample_weight), group)?;
        }
        Ok(())
    }

    /// Prune the trees in the boosters.
    pub fn prune(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
    ) -> Result<(), PerpetualError> {
        for i in 0..self.n_boosters {
            self.boosters[i].prune(data, y.get_col(i), sample_weight, group)?;
        }
        Ok(())
    }

    /// Calibrate the boosters using a selected non-conformal method.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - A tuple of (features, targets, alphas) representing the dedicated calibration set.
    pub fn calibrate(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&Matrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        self.cfg.calibration_method = method;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate(method, (x_cal, y_cal_col, alpha))?;
        }
        Ok(())
    }

    /// Calibrate the boosters using Conformal Prediction (CQR).
    pub fn calibrate_conformal(
        &mut self,
        data: &Matrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        data_cal: (&Matrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        self.cfg.calibration_method = CalibrationMethod::Conformal;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_conformal(
                data,
                y.get_col(i),
                sample_weight,
                group,
                (x_cal, y_cal_col, alpha),
            )?;
        }
        Ok(())
    }

    /// Calibrate the boosters on columnar data using a selected non-conformal method.
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use (MinMax, GRP, or WeightVariance).
    /// * `data_cal` - Dedicated calibration set (features, targets, alphas).
    pub fn calibrate_columnar(
        &mut self,
        method: CalibrationMethod,
        data_cal: (&ColumnarMatrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        if !self.cfg.save_node_stats {
            return Err(PerpetualError::InvalidParameter(
                "save_node_stats".to_string(),
                "true".to_string(),
                "false".to_string(),
            ));
        }
        self.cfg.calibration_method = method;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_columnar(method, (x_cal, y_cal_col, alpha))?;
        }
        Ok(())
    }

    /// Calibrate the boosters on columnar data using Conformal Prediction (CQR).
    pub fn calibrate_conformal_columnar(
        &mut self,
        data: &ColumnarMatrix<f64>,
        y: &Matrix<f64>,
        sample_weight: Option<&[f64]>,
        group: Option<&[u64]>,
        data_cal: (&ColumnarMatrix<f64>, &Matrix<f64>, &[f64]),
    ) -> Result<(), PerpetualError> {
        self.cfg.calibration_method = CalibrationMethod::Conformal;
        let (x_cal, ys_cal, alpha) = data_cal;
        for i in 0..self.n_boosters {
            let y_cal_col = ys_cal.get_col(i);
            self.boosters[i].calibrate_conformal_columnar(
                data,
                y.get_col(i),
                sample_weight,
                group,
                (x_cal, y_cal_col, alpha),
            )?;
        }
        Ok(())
    }

    /// Get the boosters
    pub fn get_boosters(&self) -> &[PerpetualBooster] {
        &self.boosters
    }

    // Set methods for paramters

    /// Set n_boosters on the booster. This will also initialize the boosters by cloning the first one.
    /// * `n_boosters` - The number of boosters.
    pub fn set_n_boosters(mut self, n_boosters: usize) -> Self {
        self.n_boosters = n_boosters;
        self.boosters = (0..n_boosters).map(|_| self.boosters[0].clone()).collect();
        self
    }

    /// Set the objective on the booster.
    /// * `objective` - The objective type of the booster.
    pub fn set_objective(mut self, objective: Objective) -> Self {
        let tree_objective = objective.clone();

        self.boosters = self
            .boosters
            .into_iter()
            .map(|b| b.set_objective(tree_objective.clone()))
            .collect();

        self.cfg.objective = objective;

        self
    }

    /// Set the budget on the booster.
    /// * `budget` - Budget to fit the booster.
    pub fn set_budget(mut self, budget: f32) -> Self {
        self.cfg.budget = budget;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_budget(budget)).collect();
        self
    }

    /// Set the number of bins on the booster.
    /// * `max_bin` - Number of bins to calculate to partition the data. Setting this to
    ///   a smaller number, will result in faster training time, while potentially sacrificing
    ///   accuracy. If there are more bins, than unique values in a column, all unique values
    ///   will be used.
    pub fn set_max_bin(mut self, max_bin: u16) -> Self {
        self.cfg.max_bin = max_bin;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_max_bin(max_bin)).collect();
        self
    }

    /// Set the number of threads on the booster.
    /// * `num_threads` - Set the number of threads to be used during training.
    pub fn set_num_threads(mut self, num_threads: Option<usize>) -> Self {
        self.cfg.num_threads = num_threads;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_num_threads(num_threads))
            .collect();
        self
    }

    /// Set the monotone_constraints on the booster.
    /// * `monotone_constraints` - The monotone constraints of the booster.
    pub fn set_monotone_constraints(mut self, monotone_constraints: Option<ConstraintMap>) -> Self {
        self.cfg.monotone_constraints = monotone_constraints.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_monotone_constraints(monotone_constraints.clone()))
            .collect();
        self
    }

    /// Set the interaction_constraints on the booster.
    /// * `interaction_constraints` - The interaction constraints of the booster.
    pub fn set_interaction_constraints(mut self, interaction_constraints: Option<Vec<Vec<usize>>>) -> Self {
        self.cfg.interaction_constraints = interaction_constraints.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_interaction_constraints(interaction_constraints.clone()))
            .collect();
        self
    }

    /// Set the force_children_to_bound_parent on the booster.
    /// * `force_children_to_bound_parent` - Set force children to bound parent.
    pub fn set_force_children_to_bound_parent(mut self, force_children_to_bound_parent: bool) -> Self {
        self.cfg.force_children_to_bound_parent = force_children_to_bound_parent;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| {
                b.clone()
                    .set_force_children_to_bound_parent(force_children_to_bound_parent)
            })
            .collect();
        self
    }

    /// Set missing value of the booster
    /// * `missing` - Float value to consider as missing.
    pub fn set_missing(mut self, missing: f64) -> Self {
        self.cfg.missing = missing;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_missing(missing)).collect();
        self
    }

    /// Set the allow_missing_splits on the booster.
    /// * `allow_missing_splits` - Set if missing splits are allowed for the booster.
    pub fn set_allow_missing_splits(mut self, allow_missing_splits: bool) -> Self {
        self.cfg.create_missing_branch = allow_missing_splits;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_allow_missing_splits(allow_missing_splits))
            .collect();
        self
    }

    /// Set create missing value of the booster
    /// * `create_missing_branch` - Bool specifying if missing should get it's own
    ///   branch.
    pub fn set_create_missing_branch(mut self, create_missing_branch: bool) -> Self {
        self.cfg.create_missing_branch = create_missing_branch;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_create_missing_branch(create_missing_branch))
            .collect();
        self
    }

    /// Set the features where whose missing nodes should
    /// always be terminated.
    /// * `terminate_missing_features` - Hashset of the feature indices for the features that should always terminate the missing node, if create_missing_branch is true.
    pub fn set_terminate_missing_features(mut self, terminate_missing_features: HashSet<usize>) -> Self {
        self.cfg.terminate_missing_features = terminate_missing_features.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| {
                b.clone()
                    .set_terminate_missing_features(terminate_missing_features.clone())
            })
            .collect();
        self
    }

    /// Set the missing_node_treatment on the booster.
    /// * `missing_node_treatment` - The missing node treatment of the booster.
    pub fn set_missing_node_treatment(mut self, missing_node_treatment: MissingNodeTreatment) -> Self {
        self.cfg.missing_node_treatment = missing_node_treatment;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_missing_node_treatment(missing_node_treatment))
            .collect();
        self
    }

    /// Set the log iterations on the booster.
    /// * `log_iterations` - The number of log iterations of the booster.
    pub fn set_log_iterations(mut self, log_iterations: usize) -> Self {
        self.cfg.log_iterations = log_iterations;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_log_iterations(log_iterations))
            .collect();
        self
    }

    /// Set the seed on the booster.
    /// * `seed` - Integer value used to see any randomness used in the algorithm.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_seed(seed)).collect();
        self
    }

    /// Set the reset on the booster.
    /// * `reset` - Reset the model or continue training.
    pub fn set_reset(mut self, reset: Option<bool>) -> Self {
        self.cfg.reset = reset;
        self.boosters = self.boosters.iter().map(|b| b.clone().set_reset(reset)).collect();
        self
    }

    /// Set the categorical features on the booster.
    /// * `categorical_features` - categorical features.
    pub fn set_categorical_features(mut self, categorical_features: Option<HashSet<usize>>) -> Self {
        self.cfg.categorical_features = categorical_features.clone();
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_categorical_features(categorical_features.clone()))
            .collect();
        self
    }

    /// Set the timeout on the booster.
    /// * `timeout` - fit timeout limit in seconds.
    pub fn set_timeout(mut self, timeout: Option<f32>) -> Self {
        self.cfg.timeout = timeout.map(|t| t / self.n_boosters.max(1) as f32);
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_timeout(self.cfg.timeout))
            .collect();
        self
    }

    /// Set the iteration limit on the booster.
    /// * `iteration_limit` - optional limit for the number of boosting rounds.
    pub fn set_iteration_limit(mut self, iteration_limit: Option<usize>) -> Self {
        self.cfg.iteration_limit = iteration_limit;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_iteration_limit(iteration_limit))
            .collect();
        self
    }

    /// Set the memory limit on the booster.
    /// * `memory_limit` - optional limit for memory allocation.
    pub fn set_memory_limit(mut self, memory_limit: Option<f32>) -> Self {
        self.cfg.memory_limit = memory_limit;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_memory_limit(memory_limit))
            .collect();
        self
    }

    /// Set the stopping rounds on the booster.
    /// * `stopping_rounds` - optional limit for auto stopping rounds.
    pub fn set_stopping_rounds(mut self, stopping_rounds: Option<usize>) -> Self {
        self.cfg.stopping_rounds = stopping_rounds;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_stopping_rounds(stopping_rounds))
            .collect();
        self
    }

    /// Set whether to save node stats on the booster.
    /// * `save_node_stats` - Whether to save node statistics during training.
    pub fn set_save_node_stats(mut self, save_node_stats: bool) -> Self {
        self.cfg.save_node_stats = save_node_stats;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_save_node_stats(save_node_stats))
            .collect();
        self
    }

    /// Set the calibration_method on the booster.
    /// * `calibration_method` - The calibration method of the booster.
    pub fn set_calibration_method(mut self, calibration_method: CalibrationMethod) -> Self {
        self.cfg.calibration_method = calibration_method;
        self.boosters = self
            .boosters
            .iter()
            .map(|b| b.clone().set_calibration_method(calibration_method))
            .collect();
        self
    }

    /// Insert metadata
    /// * `key` - String value for the metadata key.
    /// * `value` - value to assign to the metadata key.
    pub fn insert_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get Metadata
    /// * `key` - Get the associated value for the metadata key.
    pub fn get_metadata(&self, key: &String) -> Option<String> {
        self.metadata.get(key).cloned()
    }

    /// Given a value, return the partial dependence value of that value for that
    /// feature in the model.
    ///
    /// * `feature` - The index of the feature.
    /// * `value` - The value for which to calculate the partial dependence.
    pub fn value_partial_dependence(&self, feature: usize, value: f64) -> f64 {
        self.boosters
            .iter()
            .map(|b| b.value_partial_dependence(feature, value))
            .sum::<f64>()
            / self.n_boosters as f64
    }

    /// Calculate feature importance measure for the features
    /// in the model.
    /// - `method`: variable importance method to use.
    /// - `normalize`: whether to normalize the importance values with the sum.
    pub fn calculate_feature_importance(&self, method: ImportanceMethod, normalize: bool) -> HashMap<usize, f32> {
        let cumulative_importance = self.boosters.iter().fold(HashMap::new(), |mut acc, booster| {
            let importance = booster.calculate_feature_importance(method.clone(), normalize);
            for (feature, value) in importance {
                *acc.entry(feature).or_insert(0.0) += value;
            }
            acc
        });
        cumulative_importance
            .into_iter()
            .map(|(k, v)| (k, v / self.n_boosters as f32))
            .collect()
    }
}

impl BoosterIO for MultiOutputBooster {
    fn from_json(json_str: &str) -> Result<Self, PerpetualError> {
        let mut value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| PerpetualError::UnableToRead(e.to_string()))?;
        crate::booster::core::fix_legacy_value(&mut value);
        serde_json::from_value::<Self>(value).map_err(|e| PerpetualError::UnableToRead(e.to_string()))
    }
}

#[cfg(test)]
mod multi_output_booster_test {

    use crate::Matrix;
    use crate::objective::Objective;
    use crate::{MultiOutputBooster, utils::between};
    use std::error::Error;
    use std::fs::File;
    use std::io::BufReader;

    fn read_data(path: &str, feature_names: &[&str]) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let target_name = "Cover_Type";

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = csv::ReaderBuilder::new().has_headers(true).from_reader(reader);

        let headers = csv_reader.headers()?.clone();
        let feature_indices: Vec<usize> = feature_names
            .iter()
            .map(|&name| headers.iter().position(|h| h == name).unwrap())
            .collect();
        let target_index = headers.iter().position(|h| h == target_name).unwrap();

        let mut data_columns: Vec<Vec<f64>> = vec![Vec::new(); feature_names.len()];
        let mut y = Vec::new();

        for result in csv_reader.records() {
            let record = result?;

            // Parse target
            let target_str = &record[target_index];
            let target_val = if target_str.is_empty() {
                f64::NAN
            } else {
                target_str.parse::<f64>().unwrap_or(f64::NAN)
            };
            y.push(target_val);

            // Parse features
            for (i, &idx) in feature_indices.iter().enumerate() {
                let val_str = &record[idx];
                let val = if val_str.is_empty() {
                    f64::NAN
                } else {
                    val_str.parse::<f64>().unwrap_or(f64::NAN)
                };
                data_columns[i].push(val);
            }
        }

        let data: Vec<f64> = data_columns.into_iter().flatten().collect();
        Ok((data, y))
    }

    #[test]
    fn test_multi_output_booster() -> Result<(), Box<dyn Error>> {
        let n_classes = 7;
        let n_columns = 54;
        let n_rows = 500;
        let max_bin = 5;

        let mut features: Vec<&str> = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area_0",
            "Wilderness_Area_1",
            "Wilderness_Area_2",
            "Wilderness_Area_3",
        ]
        .to_vec();

        let soil_types = (0..40).map(|i| format!("{}_{}", "Soil_Type", i)).collect::<Vec<_>>();
        let s_types = soil_types.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        features.extend(s_types);

        // Read data using csv crate
        // NOTE: The original test performed a `.head(Some(n_rows))` operation via polars.
        // We will read all and then slice, or we can just use all if n_rows is small enough.
        // n_rows is 500. `resources/cover_types_test.csv` might be larger.
        // But since this is a test, let's just use 500 rows to match original behavior exactly for performance.

        // Actually, slicing column-major data is tedious.
        // Let's modify `read_data` to take a limit optionally?
        // Or just read everything and slice `y` and `data`.
        // `data` is [col1_val1...col1_valN, col2_val1...].
        // To slice 500 rows, we need to reconstruct new vector.

        let (data_full, y_full) = read_data("resources/cover_types_test.csv", &features)?;

        let rows_full = y_full.len();
        let limit = n_rows.min(rows_full);

        let mut data = Vec::new();
        // Extract n_columns columns
        for c in 0..n_columns {
            let col_start = c * rows_full;
            data.extend_from_slice(&data_full[col_start..col_start + limit]);
        }
        let y_test = y_full[0..limit].to_vec();

        // Create Matrix from ndarray.
        let data_matrix = Matrix::new(&data, y_test.len(), n_columns);

        let mut y_vec: Vec<Vec<f64>> = Vec::new();
        for i in 0..n_classes {
            y_vec.push(
                y_test
                    .iter()
                    .map(|y| if (*y as usize) == (i + 1) { 1.0 } else { 0.0 })
                    .collect(),
            );
        }
        let y_data = y_vec.into_iter().flatten().collect::<Vec<f64>>();
        let y = Matrix::new(&y_data, y_test.len(), n_classes);

        let mut booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_max_bin(max_bin)
            .set_n_boosters(n_classes)
            .set_budget(0.1)
            .set_iteration_limit(Some(5))
            .set_memory_limit(Some(0.001));

        println!("The number of boosters: {:?}", booster.get_boosters().len());
        assert!(booster.get_boosters().len() == n_classes);

        booster.fit(&data_matrix, &y, None, None).unwrap();

        let probas = booster.predict_proba(&data_matrix, true);

        assert!(between(0.999, 1.001, probas[0..n_classes].iter().sum::<f64>() as f32));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Matrix;
    use crate::PerpetualBooster;
    use crate::booster::config::BoosterIO;
    use crate::node::Node;
    use crate::objective::Objective;
    use crate::tree::core::{Tree, TreeStopper};

    fn constant_leaf_tree(weight: f32) -> Tree {
        let mut tree = Tree::new();
        tree.nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: weight,
                leaf_weights: Some([weight; 5]),
                hessian_sum: 1.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        tree
    }

    #[test]
    fn test_multi_output_new() {
        let booster = MultiOutputBooster::new(
            2,
            Objective::SquaredLoss,
            0.5,
            256,
            None,
            None,
            None,
            false,
            f64::NAN,
            true,
            true,
            std::collections::HashSet::new(),
            crate::booster::config::MissingNodeTreatment::AverageNodeWeight,
            10,
            42,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(booster.n_boosters, 2);
        assert_eq!(booster.boosters.len(), 2);
    }

    #[test]
    fn test_multi_output_setters() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(3);
        assert_eq!(booster.n_boosters, 3);
        assert_eq!(booster.boosters.len(), 3);

        booster = booster.set_objective(Objective::LogLoss);
        for b in &booster.boosters {
            match b.cfg.objective {
                Objective::LogLoss => {}
                _ => panic!("Objective is not LogLoss"),
            }
        }

        booster = booster.set_budget(1.0);
        for b in &booster.boosters {
            assert_eq!(b.cfg.budget, 1.0);
        }

        booster = booster.set_max_bin(128);
        for b in &booster.boosters {
            assert_eq!(b.cfg.max_bin, 128);
        }
    }

    #[test]
    fn test_multiclass_sample_weight_policy_defers_to_binary_core_balance() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);
        let y_col = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        assert!(booster.build_multiclass_sample_weight(&y_col, None).is_none());
    }

    #[test]
    fn test_multiclass_sample_weight_policy_preserves_explicit_sample_weight() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);
        let y_col = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let sample_weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let weights = booster
            .build_multiclass_sample_weight(&y_col, Some(&sample_weight))
            .unwrap();

        assert_eq!(weights, sample_weight);
    }

    #[test]
    fn test_native_multiclass_sample_weight_upweights_rare_classes() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);
        let labels = vec![0, 0, 0, 0, 1, 2];

        let weights = booster.build_native_multiclass_sample_weight(&labels, None).unwrap();
        let average_weight = weights.iter().sum::<f64>() / weights.len() as f64;

        assert!(weights[4] > weights[0]);
        assert!(weights[5] > weights[0]);
        assert!((average_weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_native_multiclass_sample_weight_skips_balanced_labels() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);
        let labels = vec![0, 0, 1, 1, 2, 2];

        assert!(booster.build_native_multiclass_sample_weight(&labels, None).is_none());
    }

    #[test]
    fn test_center_multiclass_leaf_values_zero_centers_logits() {
        let mut values = vec![0.4_f32, -0.1_f32, 0.2_f32];

        MultiOutputBooster::center_multiclass_leaf_values(&mut values);

        assert!(values.iter().sum::<f32>().abs() < 1e-6);
    }

    #[test]
    fn test_native_multiclass_gate_opens_for_balanced_medium_problem() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(6);

        assert!(booster.should_use_native_multiclass(900, 38));
        assert!(!booster.should_use_native_multiclass(200, 12));
    }

    #[test]
    fn test_native_multiclass_gate_stays_off_for_large_low_dim_ovr_problem() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);

        assert!(!booster.should_use_native_multiclass(78_053, 11));
    }

    #[test]
    fn test_native_multiclass_gate_stays_off_for_three_class_medium_width_problem() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);

        assert!(!booster.should_use_native_multiclass(4_424, 36));
    }

    #[test]
    fn test_ovr_multiclass_budget_softens_large_low_dim_three_class_problem() {
        let booster = MultiOutputBooster::default()
            .set_budget(2.0)
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);

        assert!((booster.ovr_multiclass_budget(78_053, 11) - 1.0).abs() < 1e-6);
        assert!((booster.ovr_multiclass_budget(8_000, 11) - 2.0).abs() < 1e-6);
        assert!((booster.ovr_multiclass_budget(1_699, 111) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_native_multiclass_categorical_generalization_min_folds_relaxes_46933_shape() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3)
            .set_categorical_features(Some((0..1617).collect()));

        assert_eq!(
            3,
            booster.native_multiclass_categorical_generalization_min_folds(3_845, 1_617)
        );
        assert_eq!(
            5,
            booster.native_multiclass_categorical_generalization_min_folds(4_424, 36)
        );
        assert_eq!(
            5,
            booster.native_multiclass_categorical_generalization_min_folds(1_699, 112)
        );
    }

    #[test]
    fn test_ovr_prefix_selection_truncates_harmful_shared_tail() {
        let mut booster = MultiOutputBooster::default()
            .set_budget(2.0)
            .set_objective(Objective::LogLoss)
            .set_n_boosters(3);

        booster.class_priors = vec![1.0 / 3.0; 3];
        booster.boosters = vec![
            PerpetualBooster {
                base_score: 0.0,
                trees: vec![constant_leaf_tree(2.0), constant_leaf_tree(-4.0)],
                ..PerpetualBooster::default().set_objective(Objective::LogLoss)
            },
            PerpetualBooster {
                base_score: 0.0,
                trees: vec![constant_leaf_tree(-2.0), constant_leaf_tree(4.0)],
                ..PerpetualBooster::default().set_objective(Objective::LogLoss)
            },
            PerpetualBooster {
                base_score: 0.0,
                trees: vec![constant_leaf_tree(-2.0), constant_leaf_tree(4.0)],
                ..PerpetualBooster::default().set_objective(Objective::LogLoss)
            },
        ];

        let data = Matrix::new(&[0.0], 1, 1);
        let y = Matrix::new(&[1.0, 0.0, 0.0], 1, 3);

        booster.select_ovr_tree_prefix(&data, &y, None);

        assert_eq!(1, booster.boosters[0].trees.len());
        assert_eq!(1, booster.boosters[1].trees.len());
        assert_eq!(1, booster.boosters[2].trees.len());
    }

    #[test]
    fn test_predict_leaf_positions_uses_full_dataset_not_train_index() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(2);
        let data = Matrix::new(&[0.0, 1.0, 2.0], 3, 1);

        let mut tree = Tree::new();
        tree.stopper = TreeStopper::Generalization;
        tree.train_index = vec![0, 1];
        tree.leaf_node_assignments = vec![(1, 0, 1), (2, 1, 2)];
        tree.nodes.insert(
            0,
            Node {
                num: 0,
                weight_value: 0.0,
                leaf_weights: None,
                hessian_sum: 3.0,
                split_value: 1.5,
                split_feature: 0,
                split_gain: 1.0,
                missing_node: 1,
                left_child: 1,
                right_child: 2,
                is_leaf: false,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        tree.nodes.insert(
            1,
            Node {
                num: 1,
                weight_value: 0.0,
                leaf_weights: Some([0.0; 5]),
                hessian_sum: 2.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );
        tree.nodes.insert(
            2,
            Node {
                num: 2,
                weight_value: 0.0,
                leaf_weights: Some([0.0; 5]),
                hessian_sum: 1.0,
                split_value: 0.0,
                split_feature: 0,
                split_gain: 0.0,
                missing_node: 0,
                left_child: 0,
                right_child: 0,
                is_leaf: true,
                parent_node: 0,
                left_cats: None,
                stats: None,
            },
        );

        let leaf_positions = booster.predict_leaf_positions(&tree, &data);
        let loss = booster.native_multiclass_candidate_loss(
            &leaf_positions,
            &[0, 0, 1],
            None,
            &[vec![0.0; 3], vec![0.0; 3]],
            &[vec![2.0, 2.0], vec![-2.0, -2.0]],
            3.0,
        );

        assert_eq!(leaf_positions, vec![0, 0, 1]);
        assert!(loss > 1.0);
    }

    #[test]
    fn test_full_hessian_leaf_weights_improve_coupled_multiclass_leaf_loss() {
        let booster = MultiOutputBooster::default()
            .set_objective(Objective::LogLoss)
            .set_n_boosters(5);

        let probabilities: [[f64; 5]; 7] = [
            [
                0.0900183928048201,
                0.0228071061125261,
                0.0883486818412407,
                0.38102488288397174,
                0.4178009363574414,
            ],
            [
                0.5517278582092251,
                0.2442604486047733,
                0.11435592888514583,
                0.05128128084890389,
                0.03837448345195187,
            ],
            [
                0.04067398727566605,
                0.06477967085394244,
                0.7010974132125435,
                0.15670979146791023,
                0.03673913718993785,
            ],
            [
                0.40976526191636686,
                0.4568723022569662,
                0.032990269275151626,
                0.05782526206575805,
                0.0425469044857573,
            ],
            [
                0.06438588103885809,
                0.5805311887238503,
                0.305559851223012,
                0.0374506159687909,
                0.012072463045488736,
            ],
            [
                0.08566888837128157,
                0.079_488_596_244_859_3,
                0.3545847064709137,
                0.11316435517708692,
                0.36709345373585853,
            ],
            [
                0.18266915516029163,
                0.03986330506501969,
                0.44124703109690717,
                0.26353217253201753,
                0.07268833614576402,
            ],
        ];
        let labels = vec![4, 2, 3, 3, 4, 0, 2];
        let scores_by_class = (0..5)
            .map(|class_idx| probabilities.iter().map(|row| row[class_idx].ln()).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let mut tree = Tree::new();
        tree.train_index = (0..labels.len()).collect();
        tree.leaf_node_assignments = vec![(0, 0, labels.len())];

        let full_leaf_weights = booster.compute_native_multiclass_leaf_weights(
            &tree,
            &labels,
            None,
            &scores_by_class
                .iter()
                .map(|class_scores| class_scores.iter().map(|score| score.exp()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            0.05,
            0.1,
        );

        let mut diagonal_leaf_weights = vec![vec![0.0_f32; 1]; 5];
        for class_idx in 0..5 {
            let mut grad_sum = 0.0_f64;
            let mut hess_sum = 0.0_f64;
            for row_idx in 0..labels.len() {
                let probability = probabilities[row_idx][class_idx];
                let target = if labels[row_idx] == class_idx { 1.0 } else { 0.0 };
                grad_sum += probability - target;
                hess_sum += probability * (1.0 - probability);
            }
            diagonal_leaf_weights[class_idx][0] = (-grad_sum / (hess_sum + 0.05)) as f32 * 0.1;
        }
        let mut diagonal_centered = diagonal_leaf_weights
            .iter()
            .map(|weights| weights[0])
            .collect::<Vec<_>>();
        MultiOutputBooster::center_multiclass_leaf_values(&mut diagonal_centered);
        for class_idx in 0..5 {
            diagonal_leaf_weights[class_idx][0] = diagonal_centered[class_idx];
        }

        let leaf_positions = vec![0; labels.len()];
        let full_loss = booster.native_multiclass_candidate_loss(
            &leaf_positions,
            &labels,
            None,
            &scores_by_class,
            &full_leaf_weights,
            labels.len() as f32,
        );
        let diagonal_loss = booster.native_multiclass_candidate_loss(
            &leaf_positions,
            &labels,
            None,
            &scores_by_class,
            &diagonal_leaf_weights,
            labels.len() as f32,
        );

        assert!(full_loss + 1e-6 < diagonal_loss);
    }

    #[test]
    fn test_multi_output_serialization() {
        let booster = MultiOutputBooster {
            n_boosters: 1,
            boosters: vec![PerpetualBooster::default()],
            ..Default::default()
        };
        let json = booster.json_dump().unwrap();
        let booster2 = MultiOutputBooster::from_json(&json).unwrap();
        assert_eq!(booster2.n_boosters, 1);
    }

    #[test]
    fn test_multi_output_fit() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);

        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit(&data, &y, None, None).unwrap();
        assert_eq!(booster.boosters.len(), 2);
        assert!(!booster.boosters[0].trees.is_empty());
        assert!(!booster.boosters[1].trees.is_empty());
    }

    #[test]
    fn test_multi_output_calibrate() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_save_node_stats(true);
        booster = booster.set_budget(0.1);

        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit(&data, &y, None, None).unwrap();

        let alpha = vec![0.05, 0.95];
        let data_cal = (&data, &y, alpha.as_slice());
        booster
            .calibrate(crate::booster::config::CalibrationMethod::WeightVariance, data_cal)
            .unwrap();

        for b in &booster.boosters {
            assert!(!b.cal_params.is_empty());
        }
    }

    #[test]
    fn test_multi_output_calibrate_conformal() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);

        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit(&data, &y, None, None).unwrap();

        let alpha = vec![0.05, 0.95];
        let data_cal = (&data, &y, alpha.as_slice());
        booster.calibrate_conformal(&data, &y, None, None, data_cal).unwrap();

        for b in &booster.boosters {
            assert!(!b.cal_models.is_empty());
        }
    }

    #[test]
    fn test_multi_output_fit_columnar() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);

        let data_vec = [1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit_columnar(&data, &y, None, None).unwrap();
        assert!(!booster.boosters[0].trees.is_empty());
    }

    #[test]
    fn test_multi_output_prune() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);

        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit(&data, &y, None, None).unwrap();
        booster.prune(&data, &y, None, None).unwrap();
    }

    #[test]
    fn test_multi_output_calibrate_columnar() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_save_node_stats(true);
        booster = booster.set_budget(0.1);

        let data_vec = [1.0, 2.0, 3.0, 4.0];
        let col0 = &data_vec[0..2];
        let col1 = &data_vec[2..4];
        let data = ColumnarMatrix::new(vec![col0, col1], None, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);

        booster.fit_columnar(&data, &y, None, None).unwrap();

        let alpha = vec![0.05, 0.95];
        let data_cal = (&data, &y, alpha.as_slice());
        booster
            .calibrate_columnar(crate::booster::config::CalibrationMethod::WeightVariance, data_cal)
            .unwrap();
    }

    #[test]
    fn test_multi_output_metadata() {
        let mut booster = MultiOutputBooster::default();
        booster.insert_metadata("key".to_string(), "value".to_string());
        assert_eq!(booster.get_metadata(&"key".to_string()), Some("value".to_string()));
    }

    #[test]
    fn test_multi_output_partial_dependence() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        booster.fit(&data, &y, None, None).unwrap();
        let pd = booster.value_partial_dependence(0, 1.5);
        assert!(pd != 0.0);
    }

    #[test]
    fn test_multi_output_feature_importance() {
        let mut booster = MultiOutputBooster::default();
        booster = booster.set_n_boosters(2);
        booster = booster.set_budget(0.1);
        let data = Matrix::new(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = Matrix::new(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        booster.fit(&data, &y, None, None).unwrap();
        let importance = booster.calculate_feature_importance(ImportanceMethod::Weight, true);
        assert!(!importance.is_empty());
    }

    #[test]
    fn test_multi_output_all_setters() {
        let booster = MultiOutputBooster::default()
            .set_num_threads(Some(2))
            .set_monotone_constraints(None)
            .set_interaction_constraints(None)
            .set_force_children_to_bound_parent(true)
            .set_missing(f64::NAN)
            .set_allow_missing_splits(true)
            .set_create_missing_branch(true)
            .set_terminate_missing_features(HashSet::new())
            .set_missing_node_treatment(MissingNodeTreatment::None)
            .set_log_iterations(0)
            .set_seed(123)
            .set_reset(None)
            .set_categorical_features(None)
            .set_timeout(None)
            .set_iteration_limit(None)
            .set_memory_limit(None)
            .set_stopping_rounds(None)
            .set_save_node_stats(false)
            .set_calibration_method(CalibrationMethod::WeightVariance);
        assert_eq!(booster.cfg.seed, 123);
    }
}
