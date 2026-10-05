//! cuBLAS fast path for the trainer's two-dimensional products.
//!
//! burn 0.16 / cubecl 0.4 reach ~16 TFLOPS on a B200 for River Song's `(B×W)@(W×W)`
//! products; cuBLAS SGEMM reaches hundreds. [`matmul`] routes every hot product of
//! [`super::tensors`] through cuBLAS when the backend is the CUDA JIT backend and the
//! selected [`MatmulPath`] allows it, and through `Tensor::matmul` otherwise. The
//! checkpoint format and the training algorithm are untouched: only the GEMM kernel differs.
//!
//! `RIVER_MATMUL` (read once per process):
//! - `cublas-tf32` (default): cuBLAS with `CUBLAS_TF32_TENSOR_OP_MATH` — tensor-core
//!   products on 10-bit-mantissa inputs, f32 accumulation;
//! - `cublas-f32`: cuBLAS with `CUBLAS_DEFAULT_MATH` — exact IEEE f32 FMA products;
//! - `burn`: burn's own matmul (the historical path).
//!
//! Ordering: cubecl runs on its own non-blocking stream that this module cannot name, so
//! every product waits for cubecl's stream (`client.sync()`) after its output buffer is
//! allocated, runs the GEMM on a private cuBLAS stream, and waits for that stream before
//! returning. Both waits are host-side; next to a multi-millisecond product they are noise,
//! and they also keep cubecl's memory pool from recycling an operand under a running GEMM.
#![allow(unsafe_code)]

use std::sync::LazyLock;

use burn::prelude::*;

/// Which kernel computes a two-dimensional product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulPath {
    /// burn's `Tensor::matmul` (cubecl kernels).
    Burn,
    /// cuBLAS SGEMM with `CUBLAS_TF32_TENSOR_OP_MATH`.
    CublasTf32,
    /// cuBLAS SGEMM with `CUBLAS_DEFAULT_MATH` (exact f32).
    CublasF32,
}

impl MatmulPath {
    /// Environment variable selecting the path.
    pub const ENV: &'static str = "RIVER_MATMUL";

    /// Path used when [`Self::ENV`] is unset.
    pub const DEFAULT: Self = Self::CublasTf32;

    /// Parse an [`Self::ENV`] value.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "burn" => Some(Self::Burn),
            "cublas-tf32" | "cublas" | "tf32" => Some(Self::CublasTf32),
            "cublas-f32" | "f32" => Some(Self::CublasF32),
            _ => None,
        }
    }

    /// Canonical [`Self::ENV`] spelling.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::CublasTf32 => "cublas-tf32",
            Self::CublasF32 => "cublas-f32",
        }
    }
}

/// The process-wide path: [`MatmulPath::ENV`] parsed once, [`MatmulPath::DEFAULT`] when
/// unset. An unparseable value stops the process at the first product rather than
/// silently training on a different kernel.
pub fn matmul_path() -> MatmulPath {
    static PATH: LazyLock<MatmulPath> = LazyLock::new(|| {
        let path = match std::env::var(MatmulPath::ENV) {
            Ok(value) => MatmulPath::parse(&value).unwrap_or_else(|| {
                panic!(
                    "{}={value:?} is not one of burn, cublas-tf32, cublas-f32",
                    MatmulPath::ENV
                )
            }),
            Err(_) => MatmulPath::DEFAULT,
        };
        eprintln!("matmul path: {} ({}={})", path.name(), MatmulPath::ENV, path.name());
        path
    });
    *PATH
}

/// `lhs @ rhs` through the process-wide [`matmul_path`].
pub fn matmul<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Tensor<B, 2> {
    matmul_with(lhs, rhs, matmul_path())
}

/// `lhs @ rhs` through `path`. Non-CUDA backends and non-`cuda` builds always use burn's
/// matmul, whatever `path` says.
pub fn matmul_with<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>, path: MatmulPath) -> Tensor<B, 2> {
    #[cfg(feature = "cuda")]
    if path != MatmulPath::Burn && cuda::is_cuda_backend::<B>() {
        return cuda::matmul(lhs, rhs, path);
    }
    let _ = path;
    lhs.matmul(rhs)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    //! The CUDA side: burn-cuda tensor → device pointer → `cublasSgemm`.

    use std::any::{Any, TypeId};
    use std::sync::{LazyLock, Mutex};

    use burn::backend::CudaJit;
    use burn::prelude::*;
    use burn::tensor::{DType, Shape, TensorPrimitive};
    use burn_fusion::{client::FusionClient, stream::StreamId};
    use burn_jit::{fusion::JitFusionHandle, kernel::into_contiguous, tensor::JitTensor, JitBackend};
    use cubecl::cuda::CudaRuntime;
    use cudarc::cublas::{result as cublas, sys as cublas_sys};
    use cudarc::driver::{result as driver, sys as driver_sys};

    use super::MatmulPath;

    /// The JIT backend under burn's fusion layer (`CudaJit = Fusion<Jit>`).
    type Jit = JitBackend<CudaRuntime, f32, i32, u8>;
    /// burn's fusion client for the CUDA backend.
    pub type FusionClientHandle = burn_fusion::Client<<Jit as burn_fusion::FusionBackend>::FusionRuntime>;

    /// Whether `B` is the trainer's CUDA backend.
    #[must_use]
    pub fn is_cuda_backend<B: Backend>() -> bool {
        TypeId::of::<B>() == TypeId::of::<CudaJit>()
    }

    /// Reinterpret a tensor between a generic backend and the CUDA backend; the caller
    /// has checked [`is_cuda_backend`], so the `Any` round trip cannot fail.
    fn cast<T: 'static, U: 'static>(value: T) -> U {
        match (Box::new(value) as Box<dyn Any>).downcast::<U>() {
            Ok(value) => *value,
            Err(_) => unreachable!("backend type checked by is_cuda_backend"),
        }
    }

    /// `lhs @ rhs` with cuBLAS; `B` must be [`CudaJit`].
    pub fn matmul<B: Backend>(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>, path: MatmulPath) -> Tensor<B, 2> {
        let lhs: Tensor<CudaJit, 2> = cast(lhs);
        let rhs: Tensor<CudaJit, 2> = cast(rhs);
        cast(matmul_cuda(lhs, rhs, path))
    }

    /// Drain burn's lazy fusion stream and take the backing JIT tensor plus the fusion
    /// client that [`from_jit`] needs to hand a result back.
    #[must_use]
    pub fn into_jit(tensor: Tensor<CudaJit, 2>) -> (JitTensor<CudaRuntime>, FusionClientHandle) {
        let fusion = tensor.into_primitive().tensor();
        let client = fusion.client.clone();
        (client.resolve_tensor_float::<Jit>(fusion), client)
    }

    /// Register a JIT tensor with burn's fusion layer on the calling thread's stream.
    #[must_use]
    pub fn from_jit(tensor: JitTensor<CudaRuntime>, client: &FusionClientHandle) -> Tensor<CudaJit, 2> {
        let shape = tensor.shape.dims.clone();
        let fusion = client.register_tensor(JitFusionHandle::from(tensor), shape, StreamId::current(), DType::F32);
        Tensor::from_primitive(TensorPrimitive::Float(fusion))
    }

    /// Fresh contiguous `[m, n]` f32 buffer from cubecl's memory pool.
    #[must_use]
    pub fn empty_jit(like: &JitTensor<CudaRuntime>, m: usize, n: usize) -> JitTensor<CudaRuntime> {
        let client = like.client.clone();
        let handle = client.empty(m * n * std::mem::size_of::<f32>());
        JitTensor::new_contiguous(client, like.device.clone(), Shape::new([m, n]), handle, DType::F32)
    }

    fn matmul_cuda(lhs: Tensor<CudaJit, 2>, rhs: Tensor<CudaJit, 2>, path: MatmulPath) -> Tensor<CudaJit, 2> {
        let [m, k] = lhs.dims();
        let [k_rhs, n] = rhs.dims();
        assert_eq!(k, k_rhs, "matmul inner dimensions differ: [{m}, {k}] @ [{k_rhs}, {n}]");
        let fits = |d: usize| d > 0 && i32::try_from(d).is_ok();
        if !(fits(m) && fits(n) && fits(k)) || lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
            return lhs.matmul(rhs);
        }
        let (lhs, fusion_client) = into_jit(lhs);
        let (rhs, _) = into_jit(rhs);
        let out = empty_jit(&lhs, m, n);
        let a = Operand::new(lhs);
        let b = Operand::new(rhs);
        // Every operand is computed and the output buffer is free of in-flight cubecl work.
        cubecl::future::block_on(a.tensor.client.sync());
        let c_ptr = device_ptr(&out);
        let c_ld = cublas_dim(n);
        {
            let mut cublas = cublas_state().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            cublas.set_math(path);
            // Row-major `C = A·B` is column-major `Cᵀ = Bᵀ·Aᵀ`; each operand's own layout
            // decides its op flag, so transposed views cost no copy.
            let (alpha, beta) = (1.0f32, 0.0f32);
            unsafe {
                cublas::sgemm(
                    cublas.handle,
                    b.op,
                    a.op,
                    cublas_dim(n),
                    cublas_dim(m),
                    cublas_dim(k),
                    &alpha,
                    b.ptr(),
                    b.ld,
                    a.ptr(),
                    a.ld,
                    &beta,
                    c_ptr as usize as *mut f32,
                    c_ld,
                )
                .expect("cublasSgemm");
                driver::stream::synchronize(cublas.stream).expect("cuBLAS stream synchronize");
            }
        }
        drop((a, b));
        from_jit(out, &fusion_client)
    }

    fn cublas_dim(value: usize) -> i32 {
        i32::try_from(value).expect("dimension checked against i32")
    }

    /// Device address of a JIT tensor's first element (handle offset included).
    fn device_ptr(tensor: &JitTensor<CudaRuntime>) -> u64 {
        tensor.client.get_resource(tensor.handle.clone().binding()).resource().ptr
    }

    /// One GEMM operand: its device pointer, cuBLAS op flag and leading dimension, with
    /// the tensor kept alive (its cubecl handle pins the allocation) until the GEMM has
    /// completed.
    struct Operand {
        tensor: JitTensor<CudaRuntime>,
        ptr: u64,
        op: cublas_sys::cublasOperation_t,
        ld: i32,
    }

    impl Operand {
        fn new(tensor: JitTensor<CudaRuntime>) -> Self {
            let (rows, cols) = (tensor.shape.dims[0], tensor.shape.dims[1]);
            // A row-major `[rows, cols]` is column-major `(cols × rows)`, leading dimension
            // `cols`: op N. Its transposed view (strides `[1, rows]`) is column-major
            // `(rows × cols)`, leading dimension `rows`: op T. Anything else is copied.
            let (tensor, op, ld) = if tensor.strides == [cols, 1] {
                (tensor, cublas_sys::cublasOperation_t::CUBLAS_OP_N, cols)
            } else if tensor.strides == [1, rows] {
                (tensor, cublas_sys::cublasOperation_t::CUBLAS_OP_T, rows)
            } else {
                (into_contiguous(tensor), cublas_sys::cublasOperation_t::CUBLAS_OP_N, cols)
            };
            let ptr = device_ptr(&tensor);
            Self { tensor, ptr, op, ld: cublas_dim(ld) }
        }

        fn ptr(&self) -> *const f32 {
            self.ptr as usize as *const f32
        }
    }

    /// Process-wide cuBLAS handle on its own non-blocking stream.
    struct Cublas {
        handle: cublas_sys::cublasHandle_t,
        stream: driver_sys::CUstream,
        math: Option<MatmulPath>,
    }

    // Raw handles; every use is serialized by the mutex and the CUDA context is shared.
    unsafe impl Send for Cublas {}

    impl Cublas {
        fn set_math(&mut self, path: MatmulPath) {
            if self.math == Some(path) {
                return;
            }
            let mode = match path {
                MatmulPath::CublasF32 => cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
                MatmulPath::CublasTf32 => cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                MatmulPath::Burn => unreachable!("burn path never reaches cuBLAS"),
            };
            unsafe { cublas_sys::lib().cublasSetMathMode(self.handle, mode) }
                .result()
                .expect("cublasSetMathMode");
            self.math = Some(path);
        }
    }

    /// Created on first use, after cubecl has made its context current on this thread.
    fn cublas_state() -> &'static Mutex<Cublas> {
        static STATE: LazyLock<Mutex<Cublas>> = LazyLock::new(|| {
            let handle = cublas::create_handle().expect("cublasCreate");
            let stream = driver::stream::create(driver::stream::StreamKind::NonBlocking)
                .expect("cuBLAS stream create");
            unsafe { cublas::set_stream(handle, stream.cast()) }.expect("cublasSetStream");
            Mutex::new(Cublas { handle, stream, math: None })
        });
        &STATE
    }
}
