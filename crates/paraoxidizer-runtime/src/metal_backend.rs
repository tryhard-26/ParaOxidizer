#[cfg(target_os = "macos")]
pub mod macos_metal {
    use metal::*;
    use paraoxidizer_core::error::Result;
    use std::sync::Arc;

    const SHADER_SOURCE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void gemv_int4_kernel(
        device const uchar* packed_weights [[buffer(0)]],
        device const half* scale_data     [[buffer(1)]],
        device const float* x             [[buffer(2)]],
        device float* y                   [[buffer(3)]],
        constant uint& rows               [[buffer(4)]],
        constant uint& cols               [[buffer(5)]],
        constant uint& group_size         [[buffer(6)]],
        uint row_idx                      [[thread_position_in_grid]]
    ) {
        if (row_idx >= rows) return;

        uint row_packed_bytes = (cols + 1) / 2;
        uint num_groups_per_row = (cols + group_size - 1) / group_size;
        uint row_scale_words = num_groups_per_row * 2; // scale and min_offset as f16

        uint weight_start = row_idx * row_packed_bytes;
        uint scale_start = row_idx * row_scale_words;

        float sum = 0.0f;

        for (uint c = 0; c < cols; c += 2) {
            uint group_idx = c / group_size;
            float scale = float(scale_data[scale_start + group_idx * 2]);
            float min_offset = float(scale_data[scale_start + group_idx * 2 + 1]);

            uchar byte = packed_weights[weight_start + (c / 2)];
            float q0 = float(byte & 0x0F);
            float q1 = float((byte >> 4) & 0x0F);

            float w0 = min_offset + q0 * scale;
            sum += w0 * x[c];

            if (c + 1 < cols) {
                uint group_idx1 = (c + 1) / group_size;
                float scale1 = float(scale_data[scale_start + group_idx1 * 2]);
                float min_offset1 = float(scale_data[scale_start + group_idx1 * 2 + 1]);
                float w1 = min_offset1 + q1 * scale1;
                sum += w1 * x[c + 1];
            }
        }

        y[row_idx] = sum;
    }
    "#;

    pub struct MetalBackend {
        pub device: Device,
        pub command_queue: CommandQueue,
        pub pipeline_state: ComputePipelineState,
    }

    impl MetalBackend {
        pub fn new() -> Option<Arc<Self>> {
            let device = Device::system_default()?;
            let command_queue = device.new_command_queue();

            let options = CompileOptions::new();
            let library = device.new_library_with_source(SHADER_SOURCE, &options).ok()?;
            let kernel_fn = library.get_function("gemv_int4_kernel", None).ok()?;
            let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel_fn).ok()?;

            Some(Arc::new(Self {
                device,
                command_queue,
                pipeline_state,
            }))
        }

        pub fn gemv_int4(
            &self,
            packed_weights: &[u8],
            scale_data: &[u8],
            rows: usize,
            cols: usize,
            group_size: usize,
            x: &[f32],
            y: &mut [f32],
        ) -> Result<()> {
            if rows == 0 || cols == 0 {
                return Ok(());
            }

            // Apple Silicon unified memory buffers (MTLResourceStorageModeShared)
            let buf_weights = self.device.new_buffer_with_data(
                packed_weights.as_ptr() as *const _,
                packed_weights.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_scales = self.device.new_buffer_with_data(
                scale_data.as_ptr() as *const _,
                scale_data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_x = self.device.new_buffer_with_data(
                x.as_ptr() as *const _,
                (cols * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buf_y = self.device.new_buffer(
                (rows * std::mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let rows_u32 = rows as u32;
            let cols_u32 = cols as u32;
            let group_u32 = group_size as u32;

            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.pipeline_state);
            encoder.set_buffer(0, Some(&buf_weights), 0);
            encoder.set_buffer(1, Some(&buf_scales), 0);
            encoder.set_buffer(2, Some(&buf_x), 0);
            encoder.set_buffer(3, Some(&buf_y), 0);
            encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &rows_u32 as *const _ as *const _);
            encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &cols_u32 as *const _ as *const _);
            encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &group_u32 as *const _ as *const _);

            let thread_group_count = MTLSize::new(rows as u64, 1, 1);
            let thread_group_size = MTLSize::new(self.pipeline_state.max_total_threads_per_threadgroup().min(64), 1, 1);

            encoder.dispatch_threads(thread_group_count, thread_group_size);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Copy results back from unified memory buffer
            let out_ptr = buf_y.contents() as *const f32;
            unsafe {
                std::ptr::copy_nonoverlapping(out_ptr, y.as_mut_ptr(), rows);
            }

            Ok(())
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub mod non_macos {
    use paraoxidizer_core::error::{PoxError, Result};
    use std::sync::Arc;

    pub struct MetalBackend;

    impl MetalBackend {
        pub fn new() -> Option<Arc<Self>> {
            None
        }

        pub fn gemv_int4(
            &self,
            _packed_weights: &[u8],
            _scale_data: &[u8],
            _rows: usize,
            _cols: usize,
            _group_size: usize,
            _x: &[f32],
            _y: &mut [f32],
        ) -> Result<()> {
            Err(PoxError::Runtime("Metal backend is only supported on macOS".into()))
        }
    }
}

#[cfg(target_os = "macos")]
pub use macos_metal::MetalBackend;

#[cfg(not(target_os = "macos"))]
pub use non_macos::MetalBackend;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use paraoxidizer_quant::kernels::quantize_int4_group;

    #[test]
    fn test_metal_gemv_int4() {
        if let Some(metal) = MetalBackend::new() {
            let rows = 32;
            let cols = 64;
            let weights: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.05).sin()).collect();
            let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).cos()).collect();
            let mut y = vec![0.0f32; rows];

            let (packed, scales) = quantize_int4_group(&weights, 32);
            let res = metal.gemv_int4(&packed, &scales, rows, cols, 32, &x, &mut y);
            assert!(res.is_ok());
            // Ensure non-zero computation on Metal GPU
            let sum_abs: f32 = y.iter().map(|v| v.abs()).sum();
            assert!(sum_abs > 0.01, "Metal GEMV computed valid non-zero outputs: {}", sum_abs);
        }
    }
}

