use pyo3::prelude::*;

mod c_utils;
mod merge;

/// Toad Rust extensions
#[pymodule]
fn toad(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add c_utils submodule
    let c_utils_module = PyModule::new_bound(m.py(), "c_utils")?;
    c_utils::register_module(&c_utils_module)?;
    m.add_submodule(&c_utils_module)?;

    // Add merge submodule
    let merge_module = PyModule::new_bound(m.py(), "merge")?;
    merge::register_module(&merge_module)?;
    m.add_submodule(&merge_module)?;

    Ok(())
}
