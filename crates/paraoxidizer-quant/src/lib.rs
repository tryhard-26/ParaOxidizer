//! Quantization algorithms, kernels, outlier tables, and SIMD execution paths.

pub mod kernels;
pub mod outlier;

pub use kernels::{
    compute_awq_scales, dequantize_awq, dequantize_int4_group, dequantize_int8_symmetric,
    dot_product_simd, gemv_int4, gemv_int8, pack_i4, quantize_awq, quantize_gptq,
    quantize_int4_group, quantize_int8_symmetric, unpack_i4,
};
pub use outlier::{OutlierPolicy, SparseOutlierTable};
