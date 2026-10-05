use ndarray::Array2;

use crate::{
    encode_pinball, NormalizationStats, NoulPrediction, PCNError, PCNResult, LEGACY_SENSORY_DIM,
    MULTIMODAL_INPUT_DIM, MULTIMODAL_OUTPUT_DIM, OUTPUT_DIM, PCN, PINBALL_NOUL_DIM,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReconstructionMetrics {
    pub samples: usize,
    pub missing_values: usize,
    pub missing_rmse: f32,
    pub final_energy: f32,
}

fn init_batch_from_input(pcn: &PCN, input: &Array2<f32>) -> crate::BatchState {
    let mut state = pcn.init_batch_state(input.nrows());
    state.x[0].assign(input);
    for layer in 1..pcn.dims().len() {
        let projection = state.x[layer - 1].dot(&pcn.w[layer]);
        state.x[layer] = pcn.activation.apply_matrix(&projection);
    }
    state
}

pub fn predict_pinball_compatible(
    pcn: &PCN,
    inputs: &[[f32; LEGACY_SENSORY_DIM]],
    normalization: &NormalizationStats,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> PCNResult<Vec<NoulPrediction>> {
    if pcn.dims().first() != Some(&MULTIMODAL_INPUT_DIM)
        || pcn.dims().last() != Some(&MULTIMODAL_OUTPUT_DIM)
        || relax_steps == 0
        || !alpha.is_finite()
        || alpha <= 0.0
    {
        return Err(PCNError::InvalidConfig(
            "Pinball compatibility requires a multimodal PCN and positive relaxation controls"
                .to_owned(),
        ));
    }
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let mut encoded = Array2::zeros((inputs.len(), MULTIMODAL_INPUT_DIM));
    for (row, input) in inputs.iter().enumerate() {
        let normalized = normalization
            .normalize(input)
            .map_err(|error| PCNError::InvalidConfig(error.to_string()))?
            .map(f32::tanh);
        let sensory = encode_pinball(&normalized)
            .map_err(|error| PCNError::InvalidConfig(error.to_string()))?;
        for (column, value) in sensory.values.into_iter().enumerate() {
            encoded[(row, column)] = value;
        }
    }
    let output_layer = pcn.dims().len() - 1;
    let mut state = init_batch_from_input(pcn, &encoded);
    for mut row in state.x[output_layer].rows_mut() {
        for value in row.iter_mut().skip(PINBALL_NOUL_DIM) {
            *value = 0.0;
        }
    }
    for _ in 0..relax_steps {
        pcn.compute_batch_errors(&mut state)?;
        pcn.relax_batch_step(&mut state, alpha, layer_alphas)?;
        state.x[0].assign(&encoded);
        for mut row in state.x[output_layer].rows_mut() {
            for value in row.iter_mut().skip(PINBALL_NOUL_DIM) {
                *value = 0.0;
            }
        }
    }
    pcn.compute_batch_errors(&mut state)?;
    Ok(state.x[output_layer]
        .rows()
        .into_iter()
        .map(|row| {
            let mut output = [0.0; OUTPUT_DIM];
            for index in 0..OUTPUT_DIM {
                output[index] = ((row[index] + 1.0) * 0.5).clamp(0.0, 1.0);
            }
            NoulPrediction::from(output)
        })
        .collect())
}

pub fn evaluate_masked_reconstruction(
    pcn: &PCN,
    clean_input: &Array2<f32>,
    observed_input: &Array2<f32>,
    relax_steps: usize,
    alpha: f32,
    layer_alphas: &[f32],
) -> PCNResult<ReconstructionMetrics> {
    if pcn.dims().len() < 2
        || clean_input.nrows() == 0
        || clean_input.dim() != observed_input.dim()
        || clean_input.ncols() != pcn.dims()[0]
        || relax_steps == 0
        || !alpha.is_finite()
        || alpha <= 0.0
        || clean_input.iter().any(|value| !value.is_finite())
        || observed_input
            .iter()
            .any(|value| *value != 0.0 && *value != 1.0)
    {
        return Err(PCNError::InvalidConfig(
            "reconstruction evaluation requires a non-empty finite batch and a binary mask"
                .to_owned(),
        ));
    }
    crate::core::validate_layer_alphas(layer_alphas, pcn.dims().len() - 1)?;
    let corrupted = clean_input * observed_input;
    let mut state = init_batch_from_input(pcn, &corrupted);
    for _ in 0..relax_steps {
        pcn.compute_batch_errors(&mut state)?;
        let input_error = state.eps[0].clone();
        pcn.relax_batch_step(&mut state, alpha, layer_alphas)?;
        state.x[0] = &state.x[0] - alpha * input_error;
        for row in 0..clean_input.nrows() {
            for column in 0..clean_input.ncols() {
                if observed_input[(row, column)] == 1.0 {
                    state.x[0][(row, column)] = clean_input[(row, column)];
                }
            }
        }
    }
    pcn.compute_batch_errors(&mut state)?;
    let mut squared_error = 0.0f64;
    let mut missing_values = 0usize;
    for row in 0..clean_input.nrows() {
        for column in 0..clean_input.ncols() {
            if observed_input[(row, column)] == 0.0 {
                let error = f64::from(state.x[0][(row, column)] - clean_input[(row, column)]);
                squared_error += error * error;
                missing_values += 1;
            }
        }
    }
    let missing_rmse = if missing_values == 0 {
        0.0
    } else {
        (squared_error / missing_values as f64).sqrt() as f32
    };
    Ok(ReconstructionMetrics {
        samples: clean_input.nrows(),
        missing_values,
        missing_rmse,
        final_energy: pcn.compute_batch_energy(&state) / clean_input.nrows() as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TanhActivation, PCN};

    #[test]
    fn reconstruction_reports_only_hidden_coordinates() {
        let pcn = PCN::with_activation_seeded(vec![4, 5, 3], Box::new(TanhActivation), 5).unwrap();
        let clean = Array2::from_shape_vec((1, 4), vec![1.0, -1.0, 0.5, -0.5]).unwrap();
        let observed = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let metrics = evaluate_masked_reconstruction(&pcn, &clean, &observed, 2, 0.03, &[]).unwrap();
        assert_eq!(metrics.missing_values, 2);
        assert!(metrics.missing_rmse.is_finite());
        assert!(metrics.final_energy.is_finite());
    }
}
