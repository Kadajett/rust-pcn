//! Minimal GPU smoke test: proves the cubecl/NVRTC JIT path works on the local device
//! (e.g. sm_100 on a B200) by running a 1024x1024 matmul on device 0 through the same
//! backend type the trainer uses (`pcn::gpu::GpuBackend`).
//!
//! ```text
//! cargo run --release --features cuda --example gpu_smoke
//! ```

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Distribution;
use pcn::gpu::{init_device, GpuBackend};

fn main() {
    const N: usize = 1024;
    let device = init_device();
    let start = Instant::now();
    let a = Tensor::<GpuBackend, 2>::random([N, N], Distribution::Uniform(-1.0, 1.0), &device);
    let b = Tensor::<GpuBackend, 2>::random([N, N], Distribution::Uniform(-1.0, 1.0), &device);
    let product = a.matmul(b);
    GpuBackend::sync(&device);
    let first_sync = start.elapsed();
    let timed = Instant::now();
    let product = product.clone().matmul(product).mul_scalar(1.0 / N as f32);
    GpuBackend::sync(&device);
    let second = timed.elapsed();
    let checksum: f32 = product.sum().into_scalar();
    assert!(checksum.is_finite(), "gpu_smoke: checksum is not finite ({checksum})");
    println!(
        "gpu_smoke ok: backend={} n={N} jit+first_matmul={:.3}s second_matmul={:.4}s checksum={checksum:.6e}",
        std::any::type_name::<GpuBackend>(),
        first_sync.as_secs_f64(),
        second.as_secs_f64(),
    );
}
