//! Checkpoint save/load for PCN networks.
//!
//! Serializes network weights, biases, and dimensions to JSON.
//! The `Box<dyn Activation>` field cannot be serialized directly,
//! so we store the activation name and reconstruct on load.

use crate::core::{Activation, IdentityActivation, TanhActivation, PCN};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Serializable checkpoint data.
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Network layer dimensions.
    pub dims: Vec<usize>,
    /// Name of the activation function ("identity" or "tanh").
    pub activation_name: String,
    /// Weight matrices as nested Vec for serialization.
    pub weights: Vec<Vec<Vec<f32>>>,
    /// Bias vectors as nested Vec for serialization.
    pub biases: Vec<Vec<f32>>,
    /// Epoch at which this checkpoint was saved.
    pub epoch: usize,
    /// Average energy at checkpoint time.
    pub avg_energy: f32,
    /// Accuracy at checkpoint time.
    pub accuracy: f32,
}

/// Convert an Array2 to Vec<Vec<f32>> for serialization.
fn array2_to_vecs(arr: &Array2<f32>) -> Vec<Vec<f32>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

/// Convert Vec<Vec<f32>> back to Array2.
fn vecs_to_array2(vecs: &[Vec<f32>]) -> Result<Array2<f32>, String> {
    if vecs.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }
    let nrows = vecs.len();
    let ncols = vecs[0].len();
    let flat: Vec<f32> = vecs.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| format!("Failed to reconstruct weight matrix: {e}"))
}

/// Reconstruct an activation function from its name.
fn activation_from_name(name: &str) -> Result<Box<dyn Activation>, String> {
    match name {
        "identity" => Ok(Box::new(IdentityActivation)),
        "tanh" => Ok(Box::new(TanhActivation)),
        _ => Err(format!("Unknown activation function: {name}")),
    }
}

/// Save a PCN checkpoint to a JSON file.
///
/// # Errors
///
/// Returns an error if the file cannot be written or the data cannot be serialized.
pub fn save_checkpoint(
    pcn: &PCN,
    path: &Path,
    epoch: usize,
    avg_energy: f32,
    accuracy: f32,
) -> Result<(), String> {
    let data = CheckpointData {
        dims: pcn.dims.clone(),
        activation_name: pcn.activation.name().to_string(),
        weights: pcn.w.iter().map(array2_to_vecs).collect(),
        biases: pcn.b.iter().map(|b| b.to_vec()).collect(),
        epoch,
        avg_energy,
        accuracy,
    };

    let json = serde_json::to_string_pretty(&data)
        .map_err(|e| format!("Failed to serialize checkpoint: {e}"))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create checkpoint directory: {e}"))?;
    }

    std::fs::write(path, json)
        .map_err(|e| format!("Failed to write checkpoint to {}: {e}", path.display()))
}

/// Load a PCN checkpoint from a JSON file.
///
/// Reconstructs the network with the stored activation function.
/// If `activation_override` is provided, it will be used instead of the stored one.
///
/// # Errors
///
/// Returns an error if the file cannot be read, parsed, or the network cannot be reconstructed.
pub fn load_checkpoint(
    path: &Path,
    activation_override: Option<Box<dyn Activation>>,
) -> Result<(CheckpointData, PCN), String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read checkpoint from {}: {e}", path.display()))?;

    let data: CheckpointData =
        serde_json::from_str(&json).map_err(|e| format!("Failed to parse checkpoint: {e}"))?;

    let activation = match activation_override {
        Some(a) => a,
        None => activation_from_name(&data.activation_name)?,
    };

    // Reconstruct weight matrices
    let mut w = Vec::with_capacity(data.weights.len());
    for weight_vecs in &data.weights {
        w.push(vecs_to_array2(weight_vecs)?);
    }

    // Reconstruct bias vectors
    let b: Vec<Array1<f32>> = data
        .biases
        .iter()
        .map(|bv| Array1::from(bv.clone()))
        .collect();

    let pcn = PCN {
        dims: data.dims.clone(),
        w,
        b,
        activation,
    };

    Ok((data, pcn))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_test_pcn() -> PCN {
        PCN::with_activation(vec![4, 3, 2], Box::new(TanhActivation)).expect("valid dims")
    }

    #[test]
    fn test_checkpoint_round_trip() {
        let pcn = make_test_pcn();
        let dir = std::env::temp_dir().join("pcn_test_checkpoint");
        let path = dir.join("test_checkpoint.json");

        // Save
        let result = save_checkpoint(&pcn, &path, 5, 0.42, 0.15);
        assert!(result.is_ok(), "Failed to save: {:?}", result.err());

        // Load
        let (data, loaded_pcn) = load_checkpoint(&path, None).expect("Failed to load");

        assert_eq!(data.epoch, 5);
        assert_eq!(data.dims, vec![4, 3, 2]);
        assert_eq!(data.activation_name, "tanh");
        assert_eq!(loaded_pcn.dims, pcn.dims);
        assert_eq!(loaded_pcn.w.len(), pcn.w.len());
        assert_eq!(loaded_pcn.b.len(), pcn.b.len());

        // Verify weight values match
        for (original, loaded) in pcn.w.iter().zip(loaded_pcn.w.iter()) {
            assert_eq!(original.shape(), loaded.shape());
            for (a, b) in original.iter().zip(loaded.iter()) {
                assert!((a - b).abs() < 1e-6, "Weight mismatch: {a} vs {b}");
            }
        }

        // Verify bias values match
        for (original, loaded) in pcn.b.iter().zip(loaded_pcn.b.iter()) {
            assert_eq!(original.len(), loaded.len());
            for (a, b) in original.iter().zip(loaded.iter()) {
                assert!((a - b).abs() < 1e-6, "Bias mismatch: {a} vs {b}");
            }
        }

        // Cleanup
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_with_activation_override() {
        let pcn = make_test_pcn();
        let dir = std::env::temp_dir().join("pcn_test_checkpoint_override");
        let path = dir.join("test_override.json");

        save_checkpoint(&pcn, &path, 1, 1.0, 0.0).expect("save");

        // Load with identity activation override
        let (_data, loaded_pcn) =
            load_checkpoint(&path, Some(Box::new(IdentityActivation))).expect("load");
        assert_eq!(loaded_pcn.activation.name(), "identity");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_creates_directory() {
        let dir = std::env::temp_dir()
            .join("pcn_test_nested")
            .join("deep")
            .join("path");
        let path = dir.join("checkpoint.json");

        let pcn = make_test_pcn();
        let result = save_checkpoint(&pcn, &path, 0, 0.0, 0.0);
        assert!(result.is_ok());
        assert!(path.exists());

        let _ = fs::remove_dir_all(std::env::temp_dir().join("pcn_test_nested"));
    }

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let result = load_checkpoint(Path::new("/nonexistent/path.json"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_activation_name() {
        let result = activation_from_name("relu");
        assert!(result.is_err());
    }
}
