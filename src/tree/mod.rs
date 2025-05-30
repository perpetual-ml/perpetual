pub mod predict;
pub mod tree;

// Unit-testing
#[cfg(test)]
mod tests {
    
    use crate::binning::bin_matrix;
    use crate::constraints::{Constraint, ConstraintMap};
    use crate::histogram::NodeHistogramOwned;
    use crate::objective_functions::{Objective, ObjectiveFunction};
    use crate::splitter::{MissingImputerSplitter, SplitInfo};
    use crate::utils::precision_round;
    
    use std::error::Error;
    use std::fs;
    
    
    use crate::Matrix;
    use crate::splitter::SplitInfoSlice;
    use crate::tree::tree::Tree;
    use crate::histogram::NodeHistogram;
    use std::collections::HashSet;

    #[test]
    fn test_tree_fit() {

        // instantiate objective function
        let objective_function = Objective::LogLoss.as_function();

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None);
        let loss = objective_function.loss(&y, &yhat, None);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut g,
            h.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            //loss_fn.clone(),
            &yhat,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &split_info_slice,
            n_nodes_alloc,
        );

        println!("{}", tree);
        let preds = tree.predict(&data, false, &f64::NAN);
        println!("{:?}", &preds[0..10]);
        assert_eq!(27, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_monotone() {

        // instantiate objective function
        let objective_function = Objective::LogLoss.as_function();

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None);
        let loss = objective_function.loss(&y, &yhat, None);
        println!("GRADIENT -- {:?}", g);

        let data_ = Matrix::new(&data_vec, 891, 5);
        let data = Matrix::new(data_.get_col(1), 891, 1);
        let map = ConstraintMap::from([(0, Constraint::Negative)]);
        let splitter = MissingImputerSplitter::new(0.3, true, map);
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 100, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut g,
            h.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            //loss_fn.clone(),
            &yhat,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &split_info_slice,
            n_nodes_alloc,
        );

        let mut pred_data_vec = data.get_col(0).to_owned();
        pred_data_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pred_data_vec.dedup();
        let pred_data = Matrix::new(&pred_data_vec, pred_data_vec.len(), 1);

        let preds = tree.predict(&pred_data, false, &f64::NAN);
        let increasing = preds.windows(2).all(|a| a[0] >= a[1]);
        assert!(increasing);

        let weights = tree.distribute_leaf_weights();

        // Average contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 3), precision_round(j, 3));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);
        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_fit_lossguide() {

        // instantiate objective function
        let objective_function = Objective::LogLoss.as_function();

        let file =
            fs::read_to_string("resources/contiguous_no_missing.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let file = fs::read_to_string("resources/performance.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();
        let yhat = vec![0.5; y.len()];
        let (mut g, mut h) = objective_function.gradient(&y, &yhat, None);
        let loss = objective_function.loss(&y, &yhat, None);

        let data = Matrix::new(&data_vec, 891, 5);
        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());
        let mut tree = Tree::new();

        let b = bin_matrix(&data, None, 300, f64::NAN, None).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        tree.fit(
            &objective_function,
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut g,
            h.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            //loss_fn.clone(),
            &yhat,
            None,
            is_const_hess,
            &mut hist_tree,
            None,
            &split_info_slice,
            n_nodes_alloc,
        );

        println!("{}", tree);
        // let preds = tree.predict(&data, false);
        // println!("{:?}", &preds[0..10]);
        // assert_eq!(25, tree.nodes.len());
        // Test contributions prediction...
        let weights = tree.distribute_leaf_weights();
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_average(&data, &mut contribs, &weights, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }

        // Weight contributions
        let mut contribs = vec![0.; (data.cols + 1) * data.rows];
        tree.predict_contributions_weight(&data, &mut contribs, &f64::NAN);
        let full_preds = tree.predict(&data, true, &f64::NAN);
        assert_eq!(contribs.len(), (data.cols + 1) * data.rows);

        let contribs_preds: Vec<f64> = contribs.chunks(data.cols + 1).map(|i| i.iter().sum()).collect();
        println!("{:?}", &contribs[0..10]);
        println!("{:?}", &contribs_preds[0..10]);

        assert_eq!(contribs_preds.len(), full_preds.len());
        for (i, j) in full_preds.iter().zip(contribs_preds) {
            assert_eq!(precision_round(*i, 7), precision_round(j, 7));
        }
    }

    #[test]
    fn test_tree_categorical() -> Result<(), Box<dyn Error>> {

        // instantiate objective function
        let objective_function = Objective::LogLoss.as_function();
        
        let n_bins = 256;
        let n_rows = 712;
        let n_columns = 13;

        let file =
            fs::read_to_string("resources/titanic_train_flat.csv").expect("Something went wrong reading the file");
        let data_vec: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap_or(f64::NAN)).collect();
        let data = Matrix::new(&data_vec, n_rows, n_columns);

        let file = fs::read_to_string("resources/titanic_train_y.csv").expect("Something went wrong reading the file");
        let y: Vec<f64> = file.lines().map(|x| x.parse::<f64>().unwrap()).collect();

        let y_avg = y.iter().sum::<f64>() / y.len() as f64;
        let yhat = vec![y_avg; y.len()];
        let (mut grad, mut hess) = objective_function.gradient(&y, &yhat, None);
        let loss = objective_function.loss(&y, &yhat, None);

        let splitter = MissingImputerSplitter::new(0.3, true, ConstraintMap::new());

        let cat_index = HashSet::from([0, 3, 4, 6, 7, 8, 10, 11]);

        let b = bin_matrix(&data, None, n_bins, f64::NAN, Some(&cat_index)).unwrap();
        let bdata = Matrix::new(&b.binned_data, data.rows, data.cols);
        let col_index: Vec<usize> = (0..data.cols).collect();
        let is_const_hess = false;

        let n_nodes_alloc = 100;

        let mut hist_tree_owned: Vec<NodeHistogramOwned> = (0..n_nodes_alloc)
            .map(|_| NodeHistogramOwned::empty_from_cuts(&b.cuts, &col_index, is_const_hess, true))
            .collect();

        let mut hist_tree: Vec<NodeHistogram> = hist_tree_owned
            .iter_mut()
            .map(|node_hist| NodeHistogram::from_owned(node_hist))
            .collect();

        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();

        let mut split_info_vec: Vec<SplitInfo> = (0..col_index.len()).map(|_| SplitInfo::default()).collect();
        let split_info_slice = SplitInfoSlice::new(&mut split_info_vec);

        let mut tree = Tree::new();
        tree.fit(
            &objective_function,
            &bdata,
            data.index.to_owned(),
            &col_index,
            &mut grad,
            hess.as_deref_mut(),
            &splitter,
            &pool,
            Some(f32::MAX),
            &loss,
            &y,
            //loss_fn.clone(),
            &yhat,
            None,
            false,
            &mut hist_tree,
            Some(&cat_index),
            &split_info_slice,
            n_nodes_alloc,
        );
        println!("{}", tree);
        println!("tree.nodes.len: {}", tree.nodes.len());
        println!("data.first: {:?}", data.get_row(10));
        println!("tree.predict: {}", tree.predict(&data, true, &f64::NAN)[10]);
        println!("hist_tree[0]:");
        for (i, item) in hist_tree[10].data.get(3).iter().enumerate() {
            println!("The {}th item is {:?}", i, item);
        }

        let pred_nodes = tree.predict_nodes(&data, true, &f64::NAN);
        println!("pred_nodes.len: {}", pred_nodes.len());
        println!("pred_nodes[0].len: {}", pred_nodes[0].len());
        println!("pred_nodes[0]: {:?}", pred_nodes[0]);
        println!("data.rows: {}", data.rows);
        assert_eq!(data.rows, pred_nodes.len());

        Ok(())
    }
}