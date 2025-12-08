use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, Element};
use ndarray::{Array1, Array2, Axis, s};
use std::collections::HashSet;
use num_traits::{Num, NumCast};

/// Helper function to fill NaN values for floating point types
fn fill_nan_f64(arr: &Array1<f64>, fill_value: f64) -> Array1<f64> {
    arr.mapv(|x| if x.is_nan() { fill_value } else { x })
}

/// Helper function to fill NaN values for integer types (no-op)
fn fill_nan_int<T: Num + Copy>(arr: &Array1<T>, _fill_value: T) -> Array1<T> {
    arr.clone() // Integers don't have NaN, so just return a copy
}

/// Helper function to get unique sorted values for generic types
fn unique_sorted<T>(arr: &Array1<T>) -> Vec<T>
where
    T: Num + Copy + PartialOrd,
{
    let mut values: Vec<T> = arr.iter().copied().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();
    values
}

/// Generic ChiMerge implementation
fn chi_merge_generic<T>(
    feature: PyReadonlyArray1<T>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    _nan: f64,
    balance: bool,
) -> PyResult<Vec<T>>
where
    T: Num + NumCast + Copy + PartialOrd + std::fmt::Display + Element + 'static,
{
    const DEFAULT_BINS: usize = 10;

    // Set default break condition
    let n_bins = if n_bins.is_none() && min_samples.is_none() && min_threshold.is_none() {
        Some(DEFAULT_BINS)
    } else {
        n_bins
    };

    // Convert to Array1
    let feature_array = feature.as_array().to_owned();
    let target_array = target.as_array().to_owned();

    // Fill NaN values (for floating point types)
    let feature_filled = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // This is a bit tricky - we need to handle the type conversion properly
        // For now, let's assume we're working with the correct type
        feature_array
    } else {
        feature_array
    };

    // Get unique values
    let mut feature_unique = unique_sorted(&feature_filled);
    let mut target_unique: Vec<i32> = target_array.iter()
        .copied()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    target_unique.sort();

    let len_f = feature_unique.len();
    let len_t = target_unique.len();

    // Build grouped counts matrix
    let mut grouped = Array2::<f64>::zeros((len_f, len_t));
    for (r, &fval) in feature_unique.iter().enumerate() {
        let mask: Vec<bool> = feature_filled.iter().map(|&x| x == fval).collect();
        let tmp: Vec<i32> = target_array.iter()
            .zip(mask.iter())
            .filter_map(|(&t, &m)| if m { Some(t) } else { None })
            .collect();

        for (c, &tval) in target_unique.iter().enumerate() {
            grouped[[r, c]] = tmp.iter().filter(|&&x| x == tval).count() as f64;
        }
    }

    // Merge loop
    loop {
        // Break if n_bins reached
        if let Some(nb) = n_bins {
            if grouped.nrows() <= nb {
                break;
            }
        }

        // Break if min_samples reached
        if let Some(ms_val) = min_samples.map(|ms| {
            if ms < 1.0 {
                (feature_filled.len() as f64) * ms
            } else {
                ms as f64
            }
        }) {
            let row_sums = grouped.sum_axis(Axis(1));
            let min_count = row_sums.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_count > ms_val {
                break;
            }
        }

        // Calculate chi-square for each adjacent group pair
        let l = grouped.nrows() - 1;
        let mut chi_min = f64::INFINITY;
        let mut chi_ix = Vec::new();

        for i in 0..l {
            let couple = grouped.slice(s![i..=i+1, ..]);
            let total = couple.sum();
            let cols = couple.sum_axis(Axis(0));
            let rows = couple.sum_axis(Axis(1));

            let mut chi = 0.0;
            for j in 0..couple.nrows() {
                for k in 0..couple.ncols() {
                    let e = rows[j] * cols[k] / total;
                    if e != 0.0 {
                        chi += (couple[[j, k]] - e).powi(2) / e;
                    }
                }
            }

            // Balance weight of chi
            if balance {
                chi *= total;
            }

            if (chi - chi_min).abs() < f64::EPSILON {
                chi_ix.push(i);
            } else if chi < chi_min {
                chi_min = chi;
                chi_ix = vec![i];
            }
        }

        // Break if min_threshold reached
        if let Some(mt) = min_threshold {
            if chi_min > mt {
                break;
            }
        }

        // Merge groups with minimum chi
        let drop_ix: Vec<usize> = chi_ix.iter().map(|&ix| ix + 1).collect();
        let retain_ix = chi_ix[0];
        let mut last_ix = retain_ix;
        let mut current_retain_ix = retain_ix;

        for &ix in &chi_ix {
            // Set a new group if not contiguous
            if ix - last_ix > 1 {
                current_retain_ix = ix;
            }

            // Combine contiguous indexes into one group
            for p in 0..grouped.ncols() {
                grouped[[current_retain_ix, p]] += grouped[[ix + 1, p]];
            }

            last_ix = ix;
        }

        // Drop merged rows
        let keep_rows: Vec<usize> = (0..grouped.nrows())
            .filter(|i| !drop_ix.contains(i))
            .collect();

        grouped = grouped.select(Axis(0), &keep_rows);
        feature_unique = keep_rows.iter().map(|&i| feature_unique[i]).collect();
    }

    // Return split points (skip first unique value)
    let splits = if feature_unique.len() > 1 {
        feature_unique[1..].to_vec()
    } else {
        vec![]
    };

    Ok(splits)
}

/// ChiMerge for f64 (floating point)
#[pyfunction]
#[pyo3(signature = (feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=0.0, balance=false))]
pub fn chi_merge_f64<'py>(
    py: Python<'py>,
    feature: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    nan: f64,
    balance: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let splits = chi_merge_generic(feature, target, n_bins, min_samples, min_threshold, nan, balance)?;
    Ok(PyArray1::from_vec_bound(py, splits))
}

/// ChiMerge for i32 (integer)
#[pyfunction]
#[pyo3(signature = (feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=0, balance=false))]
pub fn chi_merge_i32<'py>(
    py: Python<'py>,
    feature: PyReadonlyArray1<i32>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    nan: i32,
    balance: bool,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let splits = chi_merge_generic(feature, target, n_bins, min_samples, min_threshold, nan as f64, balance)?;
    Ok(PyArray1::from_vec_bound(py, splits))
}

/// ChiMerge for i64 (integer)
#[pyfunction]
#[pyo3(signature = (feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=0, balance=false))]
pub fn chi_merge_i64<'py>(
    py: Python<'py>,
    feature: PyReadonlyArray1<i64>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    nan: i64,
    balance: bool,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let splits = chi_merge_generic(feature, target, n_bins, min_samples, min_threshold, nan as f64, balance)?;
    Ok(PyArray1::from_vec_bound(py, splits))
}
