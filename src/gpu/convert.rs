//! Conversion helpers between ndarray and burn tensors.

use burn::prelude::*;
use ndarray::{Array1, Array2};

/// Convert an ndarray Array2<f32> to a burn Tensor<B, 2>.
pub fn ndarray2_to_tensor<B: Backend>(arr: &Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let (rows, cols) = arr.dim();
    let data: Vec<f32> = if arr.is_standard_layout() {
        arr.as_slice().expect("contiguous array").to_vec()
    } else {
        arr.iter().copied().collect()
    };
    Tensor::from_data(TensorData::new(data, [rows, cols]), device)
}

/// Convert an ndarray Array1<f32> to a burn Tensor<B, 1>.
pub fn ndarray1_to_tensor<B: Backend>(arr: &Array1<f32>, device: &B::Device) -> Tensor<B, 1> {
    let len = arr.len();
    let data: Vec<f32> = arr.iter().copied().collect();
    Tensor::from_data(TensorData::new(data, [len]), device)
}

/// Convert a burn Tensor<B, 2> to an ndarray Array2<f32>.
pub fn tensor_to_ndarray2<B: Backend>(tensor: Tensor<B, 2>) -> Array2<f32> {
    let shape = tensor.shape();
    let rows = shape.dims[0];
    let cols = shape.dims[1];
    let data: Vec<f32> = tensor.into_data().to_vec().expect("tensor to vec");
    Array2::from_shape_vec((rows, cols), data).expect("reshape to Array2")
}

/// Convert a burn Tensor<B, 1> to an ndarray Array1<f32>.
pub fn tensor_to_ndarray1<B: Backend>(tensor: Tensor<B, 1>) -> Array1<f32> {
    let data: Vec<f32> = tensor.into_data().to_vec().expect("tensor to vec");
    Array1::from_vec(data)
}
