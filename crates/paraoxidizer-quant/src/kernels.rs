#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop
)]

use crate::outlier::SparseOutlierTable;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use half::f16;
use paraoxidizer_core::error::{PoxError, Result};
use std::io::Cursor;

/// Pack two 4-bit unsigned integers (0..=15) into one byte
#[inline(always)]
pub fn pack_i4(v0: u8, v1: u8) -> u8 {
    (v0 & 0x0F) | ((v1 & 0x0F) << 4)
}

/// Unpack a byte into two 4-bit unsigned integers (0..=15)
#[inline(always)]
pub fn unpack_i4(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

/// Quantize a slice of f32 to group-wise INT4
pub fn quantize_int4_group(weights: &[f32], group_size: usize) -> (Vec<u8>, Vec<u8>) {
    let numel = weights.len();
    let num_groups = (numel + group_size - 1) / group_size;

    let mut packed_data = Vec::with_capacity((numel + 1) / 2);
    // Scales and zero points stored as: for each group: f16 scale (2B), f16 zero_point (2B)
    let mut scale_data = Vec::with_capacity(num_groups * 4);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(numel);
        let group_slice = &weights[start..end];

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &w in group_slice {
            if w < min_val {
                min_val = w;
            }
            if w > max_val {
                max_val = w;
            }
        }

        if (max_val - min_val).abs() < 1e-8 {
            max_val = min_val + 1e-4;
        }

        let scale = (max_val - min_val) / 15.0;
        let min_offset = min_val;

        scale_data
            .write_u16::<LittleEndian>(f16::from_f32(scale).to_bits())
            .unwrap();
        scale_data
            .write_u16::<LittleEndian>(f16::from_f32(min_offset).to_bits())
            .unwrap();

        let mut i = 0;
        while i < group_slice.len() {
            let w0 = group_slice[i];
            let q0 = ((w0 - min_offset) / scale).round().clamp(0.0, 15.0) as u8;

            let q1 = if i + 1 < group_slice.len() {
                let w1 = group_slice[i + 1];
                ((w1 - min_offset) / scale).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };

            packed_data.push(pack_i4(q0, q1));
            i += 2;
        }
    }

    (packed_data, scale_data)
}

/// Dequantize group-wise INT4 back to f32 slice
pub fn dequantize_int4_group(
    packed_data: &[u8],
    scale_data: &[u8],
    group_size: usize,
    numel: usize,
    out: &mut [f32],
) -> Result<()> {
    if out.len() < numel {
        return Err(PoxError::Quantization(
            "Output buffer too small for dequantization".into(),
        ));
    }

    let num_groups = (numel + group_size - 1) / group_size;
    let mut scale_cursor = Cursor::new(scale_data);

    let mut packed_idx = 0;
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(numel);

        let scale = f16::from_bits(scale_cursor.read_u16::<LittleEndian>()?).to_f32();
        let min_offset = f16::from_bits(scale_cursor.read_u16::<LittleEndian>()?).to_f32();

        let mut current = start;
        while current < end {
            if packed_idx >= packed_data.len() {
                break;
            }
            let byte = packed_data[packed_idx];
            packed_idx += 1;
            let (q0, q1) = unpack_i4(byte);

            out[current] = min_offset + (q0 as f32) * scale;
            current += 1;
            if current < end {
                out[current] = min_offset + (q1 as f32) * scale;
                current += 1;
            }
        }
    }

    Ok(())
}

/// Quantize a slice of f32 to symmetric INT8
pub fn quantize_int8_symmetric(weights: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs < 1e-8 {
        max_abs = 1e-4;
    }

    let scale = max_abs / 127.0;
    let mut q_data = Vec::with_capacity(weights.len());
    for &w in weights {
        let q = (w / scale).round().clamp(-127.0, 127.0) as i8;
        q_data.push(q as u8);
    }

    let mut scale_data = Vec::with_capacity(4);
    scale_data.write_f32::<LittleEndian>(scale).unwrap();

    (q_data, scale_data)
}

/// Dequantize symmetric INT8 to f32
pub fn dequantize_int8_symmetric(q_data: &[u8], scale_data: &[u8], out: &mut [f32]) -> Result<()> {
    if scale_data.len() < 4 {
        return Err(PoxError::Quantization("Invalid INT8 scale data".into()));
    }
    let mut cursor = Cursor::new(scale_data);
    let scale = cursor.read_f32::<LittleEndian>()?;

    for (i, &b) in q_data.iter().enumerate() {
        if i >= out.len() {
            break;
        }
        let q = b as i8;
        out[i] = (q as f32) * scale;
    }

    Ok(())
}

/// Quantized Matrix-Vector Multiplication: Y = W * X
/// W is (rows, cols) quantized INT4 with group_size
pub fn gemv_int4(
    packed_weights: &[u8],
    scale_data: &[u8],
    group_size: usize,
    rows: usize,
    cols: usize,
    outliers: Option<&SparseOutlierTable>,
    x: &[f32],
    y: &mut [f32],
) -> Result<()> {
    if x.len() < cols || y.len() < rows {
        return Err(PoxError::Quantization("GEMV dimension mismatch".into()));
    }

    let mut dequant_row = vec![0.0f32; cols];
    let row_packed_bytes = (cols + 1) / 2;
    let num_groups_per_row = (cols + group_size - 1) / group_size;
    let row_scale_bytes = num_groups_per_row * 4;

    for r in 0..rows {
        let weight_start = r * row_packed_bytes;
        let weight_end = (weight_start + row_packed_bytes).min(packed_weights.len());
        let scale_start = r * row_scale_bytes;
        let scale_end = (scale_start + row_scale_bytes).min(scale_data.len());

        dequantize_int4_group(
            &packed_weights[weight_start..weight_end],
            &scale_data[scale_start..scale_end],
            group_size,
            cols,
            &mut dequant_row,
        )?;

        // Apply outliers if any for this row
        if let Some(table) = outliers {
            let row_offset = (r * cols) as u32;
            for (&idx, &val) in table.indices.iter().zip(table.values.iter()) {
                if idx >= row_offset && idx < row_offset + (cols as u32) {
                    let col_idx = (idx - row_offset) as usize;
                    dequant_row[col_idx] = val.to_f32();
                }
            }
        }

        // Dot product with input vector x (SIMD accelerated where available)
        y[r] = dot_product_simd(&dequant_row, x);
    }

    Ok(())
}

/// Quantized Matrix-Vector Multiplication: Y = W * X for INT8
pub fn gemv_int8(
    q_weights: &[u8],
    scale_data: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<()> {
    if x.len() < cols || y.len() < rows || q_weights.len() < rows * cols {
        return Err(PoxError::Quantization(
            "GEMV INT8 dimension mismatch".into(),
        ));
    }
    let mut cursor = Cursor::new(scale_data);
    let scale = cursor.read_f32::<LittleEndian>()?;

    for r in 0..rows {
        let row_start = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            let q = q_weights[row_start + c] as i8;
            sum += (q as f32) * scale * x[c];
        }
        y[r] = sum;
    }
    Ok(())
}

/// Dot product of two f32 vectors with SIMD acceleration (NEON on aarch64, scalar fallback)
#[inline]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut sum0 = unsafe { vdupq_n_f32(0.0) };
        let mut sum1 = unsafe { vdupq_n_f32(0.0) };
        let mut i = 0;

        while i + 8 <= len {
            unsafe {
                let va0 = vld1q_f32(a.as_ptr().add(i));
                let vb0 = vld1q_f32(b.as_ptr().add(i));
                sum0 = vfmaq_f32(sum0, va0, vb0);

                let va1 = vld1q_f32(a.as_ptr().add(i + 4));
                let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
                sum1 = vfmaq_f32(sum1, va1, vb1);
            }
            i += 8;
        }

        let combined = unsafe { vaddq_f32(sum0, sum1) };
        let mut total = unsafe { vaddvq_f32(combined) };

        // Remainder
        while i < len {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Unrolled loop for compiler auto-vectorization on x86_64
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let mut i = 0;

        while i + 4 <= len {
            sum0 += a[i] * b[i];
            sum1 += a[i + 1] * b[i + 1];
            sum2 += a[i + 2] * b[i + 2];
            sum3 += a[i + 3] * b[i + 3];
            i += 4;
        }

        let mut total = sum0 + sum1 + sum2 + sum3;
        while i < len {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }
}

/// Compute per-channel AWQ salience scaling factors from calibration activation scales
pub fn compute_awq_scales(act_scales: &[f32], cols: usize) -> Vec<f32> {
    if act_scales.is_empty() || act_scales.len() != cols {
        return vec![1.0f32; cols];
    }
    let max_scale = act_scales
        .iter()
        .fold(0.0f32, |acc, &s| acc.max(s.abs()))
        .max(1e-6);
    let mut protection = vec![1.0f32; cols];
    for c in 0..cols {
        let ratio = (act_scales[c].abs() / max_scale).powf(0.5);
        protection[c] = ratio.clamp(0.1, 2.0);
    }
    protection
}

/// Dequantize AWQ-quantized weights back to original unscaled numerical range
pub fn dequantize_awq(
    packed: &[u8],
    scales: &[u8],
    protection: &[f32],
    group_size: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<()> {
    let numel = rows * cols;
    dequantize_int4_group(packed, scales, group_size, numel, out)?;
    if !protection.is_empty() && protection.len() == cols {
        for r in 0..rows {
            let row_offset = r * cols;
            for c in 0..cols {
                let p = protection[c];
                if p.abs() > 1e-8 {
                    out[row_offset + c] /= p;
                }
            }
        }
    }
    Ok(())
}

/// Activation-Aware Weight Quantization (AWQ)
/// Protects salient channels by scaling them prior to group-wise INT4 quantization.
pub fn quantize_awq(
    weights: &[f32],
    rows: usize,
    cols: usize,
    act_scales: &[f32],
    group_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut scaled_weights = weights.to_vec();
    let protection = compute_awq_scales(act_scales, cols);
    for r in 0..rows {
        let row_offset = r * cols;
        for c in 0..cols {
            scaled_weights[row_offset + c] *= protection[c];
        }
    }
    quantize_int4_group(&scaled_weights, group_size)
}

/// Generalized Post-Training Quantization (GPTQ)
/// Column-by-column greedy quantization with second-order inverse Hessian error compensation.
pub fn quantize_gptq(
    weights: &[f32],
    rows: usize,
    cols: usize,
    inv_hessian: &[f32],
    group_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut w_mat = weights.to_vec();
    let numel = rows * cols;
    let num_groups = (numel + group_size - 1) / group_size;

    let has_valid_h = inv_hessian.len() >= cols * cols;

    // Compute group scales and offsets
    let mut scales = Vec::with_capacity(num_groups);
    let mut min_offsets = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(numel);
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &w in &w_mat[start..end] {
            if w < min_val {
                min_val = w;
            }
            if w > max_val {
                max_val = w;
            }
        }
        if (max_val - min_val).abs() < 1e-8 {
            max_val = min_val + 1e-4;
        }
        let scale = (max_val - min_val) / 15.0;
        scales.push(scale);
        min_offsets.push(min_val);
    }

    // Column-by-column update
    for j in 0..cols {
        let h_jj = if has_valid_h {
            inv_hessian[j * cols + j]
        } else {
            1.0
        };
        let h_jj_inv = if h_jj.abs() > 1e-12 { 1.0 / h_jj } else { 1.0 };

        let mut errors = vec![0.0f32; rows];
        for r in 0..rows {
            let flat_idx = r * cols + j;
            let group_idx = flat_idx / group_size;
            let scale = scales[group_idx];
            let min_offset = min_offsets[group_idx];

            let w = w_mat[flat_idx];
            let q = (((w - min_offset) / scale).round().clamp(0.0, 15.0)) as u8;
            let w_quant = min_offset + (q as f32) * scale;
            errors[r] = w - w_quant;
            w_mat[flat_idx] = w_quant;
        }

        // Compensate error to remaining columns
        if has_valid_h && j + 1 < cols {
            for k in (j + 1)..cols {
                let h_jk = inv_hessian[j * cols + k];
                let factor = h_jk * h_jj_inv;
                for r in 0..rows {
                    w_mat[r * cols + k] -= errors[r] * factor;
                }
            }
        }
    }

    // Pack into INT4 bitstream
    let mut packed_data = Vec::with_capacity((numel + 1) / 2);
    let mut scale_data = Vec::with_capacity(num_groups * 4);

    for g in 0..num_groups {
        let scale = scales[g];
        let min_offset = min_offsets[g];
        scale_data
            .write_u16::<LittleEndian>(f16::from_f32(scale).to_bits())
            .unwrap();
        scale_data
            .write_u16::<LittleEndian>(f16::from_f32(min_offset).to_bits())
            .unwrap();

        let start = g * group_size;
        let end = (start + group_size).min(numel);
        let group_slice = &w_mat[start..end];

        for i in (0..group_slice.len()).step_by(2) {
            let w0 = group_slice[i];
            let q0 = (((w0 - min_offset) / scale).round().clamp(0.0, 15.0)) as u8;
            let q1 = if i + 1 < group_slice.len() {
                let w1 = group_slice[i + 1];
                (((w1 - min_offset) / scale).round().clamp(0.0, 15.0)) as u8
            } else {
                0
            };
            packed_data.push(pack_i4(q0, q1));
        }
    }

    (packed_data, scale_data)
}
