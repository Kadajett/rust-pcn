//! Training loops, convergence checks, and metrics.

use crate::core::PCN;
use crate::Config;
use ndarray::Array1;

/// Metrics computed during training.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Total prediction error energy
    pub energy: f32,
    /// Layer-wise error magnitudes
    pub layer_errors: Vec<f32>,
    /// Classification accuracy (if applicable)
    pub accuracy: Option<f32>,
}

/// Train the network on a single sample.
pub fn train_sample(
    _pcn: &mut PCN,
    _input: &Array1<f32>,
    _target: &Array1<f32>,
    _config: &Config,
) -> Result<Metrics, String> {
    // TODO: implement training step
    // 1. Initialize state
    // 2. Clamp input and target
    // 3. Relax for config.relax_steps iterations
    // 4. Compute errors and update weights
    // 5. Return metrics
    Err("not yet implemented".to_string())
}

/// Compute energy of a state.
pub fn compute_energy(_state: &crate::core::State) -> f32 {
    // TODO: E = (1/2) * sum(||eps[l]||^2)
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics() {
        let metrics = Metrics {
            energy: 0.5,
            layer_errors: vec![0.1, 0.2],
            accuracy: Some(0.95),
        };
        assert!(metrics.energy > 0.0);
    }
}
