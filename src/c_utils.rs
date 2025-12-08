use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Axis;

/// Find minimum value in a 1D array
#[pyfunction]
fn c_min(arr: PyReadonlyArray1<f64>) -> f64 {
    let arr = arr.as_array();
    arr.iter().fold(f64::INFINITY, |a, &b| a.min(b))
}

/// Sum all elements in a 2D array
#[pyfunction]
fn c_sum(arr: PyReadonlyArray2<f64>) -> f64 {
    let arr = arr.as_array();
    arr.sum()
}

/// Sum along axis 0 (column-wise sum)
#[pyfunction]
fn c_sum_axis_0<'py>(py: Python<'py>, arr: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
    let arr = arr.as_array();
    let result = arr.sum_axis(Axis(0));
    PyArray1::from_array_bound(py, &result)
}

/// Sum along axis 1 (row-wise sum)
#[pyfunction]
fn c_sum_axis_1<'py>(py: Python<'py>, arr: PyReadonlyArray2<f64>) -> Bound<'py, PyArray1<f64>> {
    let arr = arr.as_array();
    let result = arr.sum_axis(Axis(1));
    PyArray1::from_array_bound(py, &result)
}

/// Register c_utils functions to the module
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(c_min, m)?)?;
    m.add_function(wrap_pyfunction!(c_sum, m)?)?;
    m.add_function(wrap_pyfunction!(c_sum_axis_0, m)?)?;
    m.add_function(wrap_pyfunction!(c_sum_axis_1, m)?)?;
    Ok(())
}
