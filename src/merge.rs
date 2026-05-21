use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use crate::merge_generic::{chi_merge_generic, ConstraintMode};

/// ChiMerge - Chi-square based merging (core algorithm in Rust)
#[pyfunction]
#[pyo3(signature = (feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=-1.0, balance=true, constraint_mode="any"))]
fn chi_merge<'py>(
    py: Python<'py>,
    feature: PyReadonlyArray1<f64>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    nan: f64,
    balance: bool,
    constraint_mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let constraint_mode = constraint_mode.parse::<ConstraintMode>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    if constraint_mode == ConstraintMode::All && n_bins.is_none() && min_samples.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "`constraint_mode='all'` requires `n_bins` and/or `min_samples`; `min_threshold` is ignored in this mode",
        ));
    }

    let splits = chi_merge_generic(
        feature,
        target,
        n_bins,
        min_samples,
        min_threshold,
        nan,
        balance,
        constraint_mode,
    )?;

    Ok(PyArray1::from_vec_bound(py, splits))
}

/// Register merge functions to the module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chi_merge, m)?)?;
    m.add("DEFAULT_BINS", 10)?;
    Ok(())
}
