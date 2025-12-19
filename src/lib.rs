use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};

mod merge;

/// Unified chi_merge function that handles type dispatching
///
/// This function accepts numpy arrays of different types (f64, i32, i64)
/// and dispatches to the appropriate typed implementation.
#[pyfunction]
#[pyo3(signature = (feature, target, n_bins=None, min_samples=None, min_threshold=None, nan=None, balance=true))]
fn chi_merge<'py>(
    py: Python<'py>,
    feature: &Bound<'py, PyAny>,
    target: PyReadonlyArray1<i32>,
    n_bins: Option<usize>,
    min_samples: Option<f64>,
    min_threshold: Option<f64>,
    nan: Option<&Bound<'py, PyAny>>,
    balance: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let target_array = target.as_array().to_owned();

    // Check if feature is a numpy array and get its dtype
    if let Ok(arr_f64) = feature.downcast::<PyArray1<f64>>() {
        // f64 type
        let feature_array: PyReadonlyArray1<f64> = arr_f64.readonly();
        let feature_owned = feature_array.as_array().to_owned();

        // Extract nan value for f64
        let nan_value = if let Some(nan_obj) = nan {
            nan_obj.extract::<f64>().unwrap_or(-1.0)
        } else {
            -1.0
        };

        // Call the generic implementation
        let splits = merge::chi_merge_impl(
            feature_owned,
            target_array,
            n_bins,
            min_samples,
            min_threshold,
            nan_value,
            balance,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Convert back to numpy
        return Ok(PyArray1::from_vec_bound(py, splits).into_any());

    } else if let Ok(arr_i32) = feature.downcast::<PyArray1<i32>>() {
        // i32 type
        let feature_array: PyReadonlyArray1<i32> = arr_i32.readonly();
        let feature_owned = feature_array.as_array().to_owned();

        // Extract nan value for i32
        let nan_value = if let Some(nan_obj) = nan {
            nan_obj.extract::<i32>().unwrap_or(0)
        } else {
            0
        };

        // Call the generic implementation
        let splits = merge::chi_merge_impl(
            feature_owned,
            target_array,
            n_bins,
            min_samples,
            min_threshold,
            nan_value,
            balance,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Convert back to numpy
        return Ok(PyArray1::from_vec_bound(py, splits).into_any());

    } else if let Ok(arr_i64) = feature.downcast::<PyArray1<i64>>() {
        // i64 type
        let feature_array: PyReadonlyArray1<i64> = arr_i64.readonly();
        let feature_owned = feature_array.as_array().to_owned();

        // Extract nan value for i64
        let nan_value = if let Some(nan_obj) = nan {
            nan_obj.extract::<i64>().unwrap_or(0)
        } else {
            0
        };

        // Call the generic implementation
        let splits = merge::chi_merge_impl(
            feature_owned,
            target_array,
            n_bins,
            min_samples,
            min_threshold,
            nan_value,
            balance,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        // Convert back to numpy
        return Ok(PyArray1::from_vec_bound(py, splits).into_any());

    } else {
        // Fallback: try to convert to f64
        if let Ok(arr) = feature.call_method1("astype", ("float64",)) {
            if let Ok(arr_f64) = arr.downcast::<PyArray1<f64>>() {
                let feature_array: PyReadonlyArray1<f64> = arr_f64.readonly();
                let feature_owned = feature_array.as_array().to_owned();

                let nan_value = if let Some(nan_obj) = nan {
                    nan_obj.extract::<f64>().unwrap_or(-1.0)
                } else {
                    -1.0
                };

                let splits = merge::chi_merge_impl(
                    feature_owned,
                    target_array,
                    n_bins,
                    min_samples,
                    min_threshold,
                    nan_value,
                    balance,
                ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

                return Ok(PyArray1::from_vec_bound(py, splits).into_any());
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Feature must be a numpy array of type f64, i32, or i64"
        ))
    }
}

/// Toad Rust extensions
#[pymodule]
fn toad_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create merge submodule
    let merge_module = PyModule::new_bound(m.py(), "merge")?;

    // Add the unified chi_merge function
    merge_module.add_function(wrap_pyfunction!(chi_merge, &merge_module)?)?;

    // Add constants
    merge_module.add("DEFAULT_BINS", merge::DEFAULT_BINS)?;

    // Register the merge submodule
    m.add_submodule(&merge_module)?;

    Ok(())
}
