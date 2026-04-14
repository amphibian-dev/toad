use pyo3::prelude::*;

mod c_utils;
mod merge;
mod merge_generic;

/// Toad Rust extensions
#[pymodule]
fn toad_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add c_utils submodule directly
    let c_utils_module = PyModule::new_bound(m.py(), "c_utils")?;
    c_utils::register_module(&c_utils_module)?;
    m.add_submodule(&c_utils_module)?;

    // Add merge submodule directly
    let merge_module = PyModule::new_bound(m.py(), "merge")?;
    merge::register_module(&merge_module)?;
    m.add_submodule(&merge_module)?;

    // Add merge_generic functions
    merge_module.add_function(wrap_pyfunction!(merge_generic::chi_merge_f64, &merge_module)?)?;
    merge_module.add_function(wrap_pyfunction!(merge_generic::chi_merge_i32, &merge_module)?)?;
    merge_module.add_function(wrap_pyfunction!(merge_generic::chi_merge_i64, &merge_module)?)?;

    Ok(())
}
