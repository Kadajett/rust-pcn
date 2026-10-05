//! cuBLAS fast path: correctness against burn's matmul and exact cuBLAS f32, then
//! throughput of every path at River Song's hidden-layer shapes. Runs on the GPU in a
//! few GiB (largest operand 16384² f32 = 1 GiB), so it fits next to a live trainer.
//!
//! ```text
//! cargo build --release --features cuda --example cublas_matmul_check
//! target/release/examples/cublas_matmul_check [reference|check|bench|all] [W] [B]
//! ```
//!
//! `reference` compares cuBLAS f32, cuBLAS TF32 and burn against an f64 host product on
//! small trainer shapes (which paths are exact f32, which are TF32). `check` (default
//! W=16384, B=1024, out dim 4720) reports, per real shape, the error of cuBLAS TF32 and of
//! burn's matmul against exact cuBLAS f32: `max|a-b| / max|b|` and the Frobenius ratio.
//! `bench` reports TFLOPS for burn (autotuned cubecl), cubecl's tensor-core
//! `Strategy::Standard` kernel on its own, cuBLAS TF32 and cuBLAS exact f32.

use std::time::Instant;

use burn::prelude::*;
use burn::tensor::Distribution;
use burn_jit::tensor::JitTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::linalg::matmul::Strategy;
use pcn::gpu::cublas::cuda::{empty_jit, from_jit, into_jit};
use pcn::gpu::{init_device, matmul_with, GpuBackend, MatmulPath};

type T = Tensor<GpuBackend, 2>;

fn random(shape: [usize; 2], scale: f64, device: &<GpuBackend as Backend>::Device) -> T {
    Tensor::random(shape, Distribution::Uniform(-scale, scale), device)
}

/// `(max|a - reference| / max|reference|, ‖a - reference‖_F / ‖reference‖_F)`.
fn errors(a: T, reference: &T) -> (f32, f32) {
    let diff = a - reference.clone();
    let max_abs = |t: T| t.abs().max().into_scalar();
    let frob = |t: T| t.clone().mul(t).sum().into_scalar().sqrt();
    let scale = max_abs(reference.clone()).max(f32::MIN_POSITIVE);
    let norm = frob(reference.clone()).max(f32::MIN_POSITIVE);
    (max_abs(diff.clone()) / scale, frob(diff) / norm)
}

/// Every path against an f64 host reference on a small instance of each trainer shape:
/// establishes which GPU paths are exact f32 (≈1e-6) and which are TF32 (≈3e-4).
fn reference(w: usize, b: usize, d_out: usize, device: &<GpuBackend as Backend>::Device) -> bool {
    let host = |t: &T| -> ndarray::Array2<f64> {
        let [r, c] = t.dims();
        let values = t.clone().into_data().to_vec::<f32>().expect("host readback");
        ndarray::Array2::from_shape_vec((r, c), values.into_iter().map(f64::from).collect()).expect("shape")
    };
    let x_in = random([b, d_out], 1.0, device);
    let x_w = random([b, w], 1.0, device);
    let w_in = random([d_out, w], 0.02, device);
    let w_hid = random([w, w], 0.01, device);
    let cases: Vec<(&str, T, T)> = vec![
        ("[B,d]@[d,W]", x_in.clone(), w_in.clone()),
        ("[B,W]@[W,W]", x_w.clone(), w_hid.clone()),
        ("[B,W]@[W,W]ᵀ", x_w.clone(), w_hid.clone().transpose()),
        ("[B,d]ᵀ@[B,W]", x_in.clone().transpose(), x_w.clone()),
    ];
    let mut ok = true;
    println!("f64 host reference (W={w} B={b} d={d_out}): max|path - f64| / max|f64|");
    println!("shape            cublas-f32   cublas-tf32   burn");
    for (name, lhs, rhs) in cases {
        let truth = host(&lhs).dot(&host(&rhs));
        let scale = truth.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
        let error = |path: MatmulPath| {
            let got = host(&matmul_with(lhs.clone(), rhs.clone(), path));
            (&got - &truth).iter().fold(0.0f64, |m, v| m.max(v.abs())) / scale
        };
        let exact = error(MatmulPath::CublasF32);
        let tf32 = error(MatmulPath::CublasTf32);
        let burn = error(MatmulPath::Burn);
        let pass = exact <= 1.0e-5 && tf32 <= 1.0e-3;
        ok &= pass;
        println!("{name:<16} {exact:>10.3e} {tf32:>12.3e} {burn:>10.3e}  {}", if pass { "ok" } else { "FAIL" });
    }
    ok
}

/// cuBLAS TF32 and burn against exact cuBLAS f32 at the trainer's real shapes. Passes when
/// TF32 stays within 1e-3 of exact and exact is run-to-run deterministic; burn's deviation
/// is reported, not judged (see [`reference`] for what it means).
fn check(w: usize, b: usize, d_out: usize, device: &<GpuBackend as Backend>::Device) -> bool {
    let x_in = random([b, d_out], 1.0, device);
    let x_w = random([b, w], 1.0, device);
    let w_in = random([d_out, w], 0.02, device);
    let w_hid = random([w, w], 0.01, device);
    let direction = random([1, w], 1.0, device);
    let cases: Vec<(&str, T, T)> = vec![
        ("[B,d]@[d,W]        (projection)", x_in.clone(), w_in.clone()),
        ("[B,W]@[W,W]        (feedback)", x_w.clone(), w_hid.clone()),
        ("[B,W]@[W,W]ᵀ       (prediction)", x_w.clone(), w_hid.clone().transpose()),
        ("[B,W]@[d,W]ᵀ       (prediction, top)", x_w.clone(), w_in.clone().transpose()),
        ("[B,d]ᵀ@[B,W]       (weight delta)", x_in.clone().transpose(), x_w.clone()),
        ("[B,W]ᵀ@[B,W]       (weight delta, hidden)", x_w.clone().transpose(), x_w.clone()),
        ("[1,W]@[W,W]        (direction)", direction.clone(), w_hid.clone()),
        ("[W,W]ᵀ@[W,1]       (power iteration)", w_hid.clone().transpose(), direction.clone().transpose()),
        ("[1,W]ᵀ@[1,W]       (rank-one outer)", direction.clone().transpose(), direction.clone()),
        ("[B,W]@[W,2k]ᵀ sliced (byte slab)", x_w.clone(), w_in.clone().slice([0..2000, 0..w]).transpose()),
    ];
    let mut ok = true;
    println!("vs exact cuBLAS f32 (W={w} B={b} d={d_out})");
    println!("shape                                    tf32 max/scale  tf32 frob   burn max/scale  burn frob   exact-vs-exact");
    for (name, lhs, rhs) in cases {
        let exact = matmul_with(lhs.clone(), rhs.clone(), MatmulPath::CublasF32);
        let exact_again = matmul_with(lhs.clone(), rhs.clone(), MatmulPath::CublasF32);
        let tf32 = matmul_with(lhs.clone(), rhs.clone(), MatmulPath::CublasTf32);
        let burn = matmul_with(lhs, rhs, MatmulPath::Burn);
        let (tf32_max, tf32_frob) = errors(tf32, &exact);
        let (burn_max, burn_frob) = errors(burn, &exact);
        let (repeat_max, _) = errors(exact_again, &exact);
        let pass = tf32_max <= 1.0e-3 && repeat_max == 0.0;
        ok &= pass;
        println!(
            "{name:<40} {tf32_max:>10.3e} {tf32_frob:>10.3e} {burn_max:>13.3e} {burn_frob:>10.3e} {repeat_max:>12.1e}  {}",
            if pass { "ok" } else { "FAIL" }
        );
    }
    ok
}

fn tflops(m: usize, k: usize, n: usize, seconds: f64) -> f64 {
    2.0 * m as f64 * k as f64 * n as f64 / seconds / 1e12
}

fn time_path(lhs: &T, rhs: &T, path: MatmulPath, iterations: usize, device: &<GpuBackend as Backend>::Device) -> f64 {
    let _ = matmul_with(lhs.clone(), rhs.clone(), path).slice([0..1, 0..1]).into_scalar();
    GpuBackend::sync(device);
    let start = Instant::now();
    let mut acc = Tensor::<GpuBackend, 2>::zeros([1, 1], device);
    for _ in 0..iterations {
        acc = acc + matmul_with(lhs.clone(), rhs.clone(), path).slice([0..1, 0..1]);
    }
    let _ = acc.into_scalar();
    GpuBackend::sync(device);
    start.elapsed().as_secs_f64() / iterations as f64
}

/// cubecl 0.4's tensor-core kernel (`Strategy::Standard`, TF32 stage for f32 inputs)
/// launched directly, bypassing burn's autotune: the "can cubecl simply be fast" probe.
fn time_cubecl_standard(lhs: &T, rhs: &T, iterations: usize, device: &<GpuBackend as Backend>::Device) -> Result<f64, String> {
    let (lhs_jit, client) = into_jit(lhs.clone());
    let (rhs_jit, _) = into_jit(rhs.clone());
    let [m, _] = lhs.dims();
    let [_, n] = rhs.dims();
    let launch = |out: &JitTensor<CudaRuntime>| {
        cubecl::linalg::matmul::launch_ref::<CudaRuntime, f32>(
            &Strategy::Standard,
            &lhs_jit.client,
            &lhs_jit.as_handle_ref(),
            &rhs_jit.as_handle_ref(),
            &out.as_handle_ref(),
        )
        .map_err(|err| format!("{err:?}"))
    };
    let out = empty_jit(&lhs_jit, m, n);
    launch(&out)?;
    let probe = from_jit(out, &client);
    let _ = probe.slice([0..1, 0..1]).into_scalar();
    GpuBackend::sync(device);
    let start = Instant::now();
    for _ in 0..iterations {
        let out = empty_jit(&lhs_jit, m, n);
        launch(&out)?;
        drop(out);
    }
    GpuBackend::sync(device);
    Ok(start.elapsed().as_secs_f64() / iterations as f64)
}

fn bench(w: usize, b: usize, device: &<GpuBackend as Backend>::Device) {
    let iterations = 10;
    let w_hid = random([w, w], 0.01, device);
    let shapes: Vec<(&str, T, T)> = vec![
        ("[B,W]@[W,W]", random([b, w], 1.0, device), w_hid.clone()),
        ("[B,W]@[W,W]ᵀ", random([b, w], 1.0, device), w_hid.clone().transpose()),
        ("[B,W]ᵀ@[B,W]", random([b, w], 1.0, device).transpose(), random([b, w], 1.0, device)),
        ("[W,W]@[W,W]", random([w, w], 0.01, device), w_hid.clone()),
    ];
    for (name, lhs, rhs) in shapes {
        let [m, k] = lhs.dims();
        let [_, n] = rhs.dims();
        let mut line = format!("{name:<14} m={m} k={k} n={n}:");
        for path in [MatmulPath::Burn, MatmulPath::CublasTf32, MatmulPath::CublasF32] {
            let seconds = time_path(&lhs, &rhs, path, iterations, device);
            line.push_str(&format!(
                "  {}={:.2}ms/{:.0}TF",
                path.name(),
                seconds * 1e3,
                tflops(m, k, n, seconds)
            ));
        }
        match time_cubecl_standard(&lhs, &rhs, iterations, device) {
            Ok(seconds) => line.push_str(&format!(
                "  cubecl-standard={:.2}ms/{:.0}TF",
                seconds * 1e3,
                tflops(m, k, n, seconds)
            )),
            Err(err) => line.push_str(&format!("  cubecl-standard=unavailable({err})")),
        }
        println!("{line}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map_or("all", String::as_str);
    let w: usize = args.get(2).map_or(16384, |v| v.parse().expect("W"));
    let b: usize = args.get(3).map_or(1024, |v| v.parse().expect("B"));
    let device = init_device();
    let mut ok = true;
    if mode == "reference" || mode == "all" {
        ok &= reference(2048, 256, 590, &device);
    }
    if mode == "check" || mode == "all" {
        ok &= check(w, b, 4720, &device);
        println!("check: {}", if ok { "ok" } else { "FAIL" });
    }
    if mode == "bench" || mode == "all" {
        bench(w, b, &device);
    }
    assert!(ok, "cuBLAS correctness check failed");
}
