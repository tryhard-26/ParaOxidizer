use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use paraoxidizer_cli::commands;

/// Quantize a Hugging Face model or SafeTensors file to .pox format
#[pyfunction]
#[pyo3(signature = (model, output="model.pox", bits=4, group_size=128, outlier="automatic", algorithm="min-max"))]
fn quantize(
    model: &str,
    output: &str,
    bits: usize,
    group_size: usize,
    outlier: &str,
    algorithm: &str,
) -> PyResult<String> {
    commands::run_quantize(model, bits, group_size, outlier, algorithm, output)
        .map_err(|e| PyRuntimeError::new_err(format!("Quantization failed: {e}")))?;
    Ok(output.to_string())
}

/// Inspect a model's architecture, parameters, and estimated quantized footprints
#[pyfunction]
#[pyo3(signature = (path, format="text"))]
fn inspect(path: &str, format: &str) -> PyResult<String> {
    commands::run_inspect(path, format)
        .map_err(|e| PyRuntimeError::new_err(format!("Inspection failed: {e}")))?;
    Ok("Inspection complete".to_string())
}

/// Validate the numerical integrity and alignment of a .pox artifact
#[pyfunction]
fn validate(path: &str) -> PyResult<bool> {
    match commands::run_validate(path) {
        Ok(()) => Ok(true),
        Err(e) => Err(PyRuntimeError::new_err(format!("Validation failed: {e}"))),
    }
}

/// Verify cryptographic SHA-256 Merkle hashes and optional Ed25519 signature
#[pyfunction]
#[pyo3(signature = (path, pubkey=None))]
fn verify(path: &str, pubkey: Option<&str>) -> PyResult<bool> {
    match commands::run_verify(path, pubkey) {
        Ok(()) => Ok(true),
        Err(e) => Err(PyRuntimeError::new_err(format!("Verification failed: {e}"))),
    }
}

/// Compare precision distribution and structural drift between two artifacts
#[pyfunction]
fn diff(model_a: &str, model_b: &str) -> PyResult<bool> {
    match commands::run_diff(model_a, model_b) {
        Ok(()) => Ok(true),
        Err(e) => Err(PyRuntimeError::new_err(format!("Diff failed: {e}"))),
    }
}

/// Get ParaOxidizer version
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// ParaOxidizer Python extension module (`pox`)
#[pymodule]
fn pox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(inspect, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
