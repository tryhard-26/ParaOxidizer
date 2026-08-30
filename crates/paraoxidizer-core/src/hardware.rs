use serde::{Deserialize, Serialize};
use sysinfo::System;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub cpu_brand: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub simd_neon: bool,
    pub simd_avx2: bool,
    pub simd_avx512: bool,
    pub has_apple_silicon_gpu: bool,
    pub has_cuda_gpu: bool,
    pub unified_memory: bool,
}

impl HardwareInfo {
    pub fn probe() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let physical_cores = sys.physical_core_count().unwrap_or(1);
        let logical_cores = sys.cpus().len();
        let total_ram_mb = sys.total_memory() / (1024 * 1024);
        let available_ram_mb = sys.available_memory() / (1024 * 1024);

        let arch = std::env::consts::ARCH.to_string();
        let os = std::env::consts::OS.to_string();

        let simd_neon = cfg!(target_arch = "aarch64");
        #[cfg(target_arch = "x86_64")]
        let (simd_avx2, simd_avx512) = (
            is_x86_feature_detected!("avx2"),
            is_x86_feature_detected!("avx512f"),
        );
        #[cfg(not(target_arch = "x86_64"))]
        let (simd_avx2, simd_avx512) = (false, false);

        let is_apple_silicon = arch == "aarch64" && os == "macos";
        let has_apple_silicon_gpu = is_apple_silicon;
        let unified_memory = is_apple_silicon;

        // Simple check for CUDA presence via nvml or common paths
        let has_cuda_gpu = std::path::Path::new("/usr/local/cuda").exists()
            || std::env::var("CUDA_PATH").is_ok()
            || std::env::var("CUDA_HOME").is_ok();

        Self {
            os,
            arch,
            cpu_brand,
            physical_cores,
            logical_cores,
            total_ram_mb,
            available_ram_mb,
            simd_neon,
            simd_avx2,
            simd_avx512,
            has_apple_silicon_gpu,
            has_cuda_gpu,
            unified_memory,
        }
    }

    pub fn recommended_format(&self) -> &'static str {
        if self.has_apple_silicon_gpu {
            "INT4 (group 128) with FP16 sensitive layers (Metal / NEON optimized)"
        } else if self.simd_avx512 {
            "INT4 (group 64) with AVX-512 VNNI vectorization"
        } else if self.simd_avx2 {
            "INT4 (group 128) with AVX2 FMA acceleration"
        } else {
            "INT8 symmetric (portable scalar fallback)"
        }
    }

    pub fn recommended_runtime_backend(&self) -> &'static str {
        if self.has_apple_silicon_gpu {
            "Accelerate / NEON unified-memory CPU runtime"
        } else if self.has_cuda_gpu {
            "CUDA runtime"
        } else if self.simd_avx512 || self.simd_avx2 {
            "x86 SIMD CPU runtime"
        } else {
            "Scalar CPU runtime"
        }
    }
}
