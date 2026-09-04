use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use paraoxidizer_core::{
    arch::ModelConfig,
    error::{PoxError, Result},
    tensor::TensorMeta,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::{Cursor, Read, Write},
    path::Path,
};

pub const POX_MAGIC: &[u8; 4] = b"POX\x01";
pub const POX_VERSION: u32 = 1;
pub const HEADER_SIZE: u64 = 128;
pub const DATA_ALIGNMENT: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoxHeader {
    pub magic: [u8; 4],
    pub format_version: u32,
    pub flags: u32,
    pub metadata_offset: u64,
    pub metadata_len: u64,
    pub tensor_index_offset: u64,
    pub tensor_index_len: u64,
    pub quant_plan_offset: u64,
    pub quant_plan_len: u64,
    pub data_offset: u64,
    pub data_len: u64,
    pub manifest_offset: u64,
    pub manifest_len: u64,
    pub signature_offset: u64,
    pub signature_len: u64,
}

impl PoxHeader {
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != POX_MAGIC {
            return Err(PoxError::InvalidMagic {
                expected: *POX_MAGIC,
                found: magic,
            });
        }
        let format_version = reader.read_u32::<LittleEndian>()?;
        if format_version != POX_VERSION {
            return Err(PoxError::UnsupportedVersion(format_version));
        }
        let flags = reader.read_u32::<LittleEndian>()?;
        let _reserved = reader.read_u32::<LittleEndian>()?; // 4-byte padding to 8-byte boundary

        let metadata_offset = reader.read_u64::<LittleEndian>()?;
        let metadata_len = reader.read_u64::<LittleEndian>()?;
        let tensor_index_offset = reader.read_u64::<LittleEndian>()?;
        let tensor_index_len = reader.read_u64::<LittleEndian>()?;
        let quant_plan_offset = reader.read_u64::<LittleEndian>()?;
        let quant_plan_len = reader.read_u64::<LittleEndian>()?;
        let data_offset = reader.read_u64::<LittleEndian>()?;
        let data_len = reader.read_u64::<LittleEndian>()?;
        let manifest_offset = reader.read_u64::<LittleEndian>()?;
        let manifest_len = reader.read_u64::<LittleEndian>()?;
        let signature_offset = reader.read_u64::<LittleEndian>()?;
        let signature_len = reader.read_u64::<LittleEndian>()?;

        // Read remainder of 128-byte header (128 - 112 = 16 bytes)
        let mut padding = [0u8; 16];
        reader.read_exact(&mut padding)?;

        Ok(Self {
            magic,
            format_version,
            flags,
            metadata_offset,
            metadata_len,
            tensor_index_offset,
            tensor_index_len,
            quant_plan_offset,
            quant_plan_len,
            data_offset,
            data_len,
            manifest_offset,
            manifest_len,
            signature_offset,
            signature_len,
        })
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u32::<LittleEndian>(self.flags)?;
        writer.write_u32::<LittleEndian>(0)?; // reserved 4-byte padding

        writer.write_u64::<LittleEndian>(self.metadata_offset)?;
        writer.write_u64::<LittleEndian>(self.metadata_len)?;
        writer.write_u64::<LittleEndian>(self.tensor_index_offset)?;
        writer.write_u64::<LittleEndian>(self.tensor_index_len)?;
        writer.write_u64::<LittleEndian>(self.quant_plan_offset)?;
        writer.write_u64::<LittleEndian>(self.quant_plan_len)?;
        writer.write_u64::<LittleEndian>(self.data_offset)?;
        writer.write_u64::<LittleEndian>(self.data_len)?;
        writer.write_u64::<LittleEndian>(self.manifest_offset)?;
        writer.write_u64::<LittleEndian>(self.manifest_len)?;
        writer.write_u64::<LittleEndian>(self.signature_offset)?;
        writer.write_u64::<LittleEndian>(self.signature_len)?;

        // Pad to 128 bytes total
        writer.write_all(&[0u8; 16])?;
        Ok(())
    }
}

/// Metadata stored in .pox container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxMetadata {
    pub model_config: ModelConfig,
    pub total_parameters: u64,
    pub quantized_by: String,
    pub timestamp_utc: u64,
    pub original_format: String,
    pub base_model_name: String,
}

/// Quantization plan summary stored in .pox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxQuantPlanRecord {
    pub default_precision: String,
    pub group_size: usize,
    pub outlier_strategy: String,
    pub layer_assignments: HashMap<String, String>,
}

/// Cryptographic and provenance manifest stored in .pox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxManifest {
    pub run_id: String,
    pub source_model_sha256: String,
    pub calibration_sha256: Option<String>,
    pub compiler_version: String,
    pub target_arch: String,
    pub tensor_hashes: HashMap<String, String>,
    pub artifact_sha256: String,
}

/// Signed verification block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoxSignatureBlock {
    pub public_key_hex: String,
    pub signature_hex: String,
}

/// Memory-mapped .pox container for zero-copy high-performance inference
pub struct PoxFile {
    mmap: Mmap,
    pub header: PoxHeader,
    pub metadata: PoxMetadata,
    pub tensors: Vec<TensorMeta>,
    pub tensor_map: HashMap<String, usize>,
    pub quant_plan: PoxQuantPlanRecord,
    pub manifest: PoxManifest,
    pub signature: Option<PoxSignatureBlock>,
}

impl PoxFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Create an anonymous memory map or use the slice directly
        // For portability in tests, we write to a cursor or temp file if needed, or wrap slice
        if bytes.len() < HEADER_SIZE as usize {
            return Err(PoxError::Format("File size smaller than header".into()));
        }
        let mut cursor = Cursor::new(bytes);
        let header = PoxHeader::read_from(&mut cursor)?;

        let meta_start = header.metadata_offset as usize;
        let meta_end = meta_start + header.metadata_len as usize;
        if meta_end > bytes.len() {
            return Err(PoxError::Security("Metadata out of bounds".into()));
        }
        let metadata: PoxMetadata = serde_json::from_slice(&bytes[meta_start..meta_end])?;

        let index_start = header.tensor_index_offset as usize;
        let index_end = index_start + header.tensor_index_len as usize;
        if index_end > bytes.len() {
            return Err(PoxError::Security("Tensor index out of bounds".into()));
        }
        let tensors: Vec<TensorMeta> = serde_json::from_slice(&bytes[index_start..index_end])?;
        let mut tensor_map = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            tensor_map.insert(t.name.clone(), i);
        }

        let plan_start = header.quant_plan_offset as usize;
        let plan_end = plan_start + header.quant_plan_len as usize;
        if plan_end > bytes.len() {
            return Err(PoxError::Security("Quant plan out of bounds".into()));
        }
        let quant_plan: PoxQuantPlanRecord = serde_json::from_slice(&bytes[plan_start..plan_end])?;

        let manifest_start = header.manifest_offset as usize;
        let manifest_end = manifest_start + header.manifest_len as usize;
        if manifest_end > bytes.len() {
            return Err(PoxError::Security("Manifest out of bounds".into()));
        }
        let manifest: PoxManifest = serde_json::from_slice(&bytes[manifest_start..manifest_end])?;

        let signature = if header.signature_len > 0 {
            let sig_start = header.signature_offset as usize;
            let sig_end = sig_start + header.signature_len as usize;
            if sig_end > bytes.len() {
                return Err(PoxError::Security("Signature out of bounds".into()));
            }
            Some(serde_json::from_slice(&bytes[sig_start..sig_end])?)
        } else {
            None
        };

        // Create an anonymous Mmap backed by temp file or empty
        // To keep PoxFile holding an Mmap, let's open a temp file or memfd
        let mut temp = tempfile::tempfile()?;
        temp.write_all(bytes)?;
        let mmap = unsafe { Mmap::map(&temp)? };

        Ok(Self {
            mmap,
            header,
            metadata,
            tensors,
            tensor_map,
            quant_plan,
            manifest,
            signature,
        })
    }

    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        let bytes = &mmap[..];
        if bytes.len() < HEADER_SIZE as usize {
            return Err(PoxError::Format("File size smaller than header".into()));
        }
        let mut cursor = Cursor::new(bytes);
        let header = PoxHeader::read_from(&mut cursor)?;

        let meta_start = header.metadata_offset as usize;
        let meta_end = meta_start + header.metadata_len as usize;
        if meta_end > bytes.len() {
            return Err(PoxError::Security("Metadata out of bounds".into()));
        }
        let metadata: PoxMetadata = serde_json::from_slice(&bytes[meta_start..meta_end])?;

        let index_start = header.tensor_index_offset as usize;
        let index_end = index_start + header.tensor_index_len as usize;
        if index_end > bytes.len() {
            return Err(PoxError::Security("Tensor index out of bounds".into()));
        }
        let tensors: Vec<TensorMeta> = serde_json::from_slice(&bytes[index_start..index_end])?;
        let mut tensor_map = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            tensor_map.insert(t.name.clone(), i);
        }

        let plan_start = header.quant_plan_offset as usize;
        let plan_end = plan_start + header.quant_plan_len as usize;
        if plan_end > bytes.len() {
            return Err(PoxError::Security("Quant plan out of bounds".into()));
        }
        let quant_plan: PoxQuantPlanRecord = serde_json::from_slice(&bytes[plan_start..plan_end])?;

        let manifest_start = header.manifest_offset as usize;
        let manifest_end = manifest_start + header.manifest_len as usize;
        if manifest_end > bytes.len() {
            return Err(PoxError::Security("Manifest out of bounds".into()));
        }
        let manifest: PoxManifest = serde_json::from_slice(&bytes[manifest_start..manifest_end])?;

        let signature = if header.signature_len > 0 {
            let sig_start = header.signature_offset as usize;
            let sig_end = sig_start + header.signature_len as usize;
            if sig_end > bytes.len() {
                return Err(PoxError::Security("Signature out of bounds".into()));
            }
            Some(serde_json::from_slice(&bytes[sig_start..sig_end])?)
        } else {
            None
        };

        Ok(Self {
            mmap,
            header,
            metadata,
            tensors,
            tensor_map,
            quant_plan,
            manifest,
            signature,
        })
    }

    /// Access raw quantized payload for a given tensor name
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let idx = *self.tensor_map.get(name)?;
        let meta = &self.tensors[idx];
        let start = meta.data_offset as usize;
        let end = start + meta.data_len as usize;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Access raw scale and zero-point buffer for a given tensor name
    pub fn get_scale_data(&self, name: &str) -> Option<&[u8]> {
        let idx = *self.tensor_map.get(name)?;
        let meta = &self.tensors[idx];
        if meta.scale_len == 0 {
            return None;
        }
        let start = meta.scale_offset as usize;
        let end = start + meta.scale_len as usize;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Access raw outlier buffer for a given tensor name
    pub fn get_outlier_data(&self, name: &str) -> Option<&[u8]> {
        let idx = *self.tensor_map.get(name)?;
        let meta = &self.tensors[idx];
        if meta.outlier_len == 0 {
            return None;
        }
        let start = meta.outlier_offset as usize;
        let end = start + meta.outlier_len as usize;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Verify complete internal SHA-256 integrity of all tensors
    pub fn verify_integrity(&self) -> Result<()> {
        for meta in &self.tensors {
            let data = self.get_tensor_data(&meta.name).ok_or_else(|| {
                PoxError::Format(format!("Missing data for tensor {}", meta.name))
            })?;
            let mut hasher = Sha256::new();
            hasher.update(data);
            if meta.scale_len > 0 {
                if let Some(scales) = self.get_scale_data(&meta.name) {
                    hasher.update(scales);
                }
            }
            if meta.outlier_len > 0 {
                if let Some(outliers) = self.get_outlier_data(&meta.name) {
                    hasher.update(outliers);
                }
            }
            let hash = hex::encode(hasher.finalize());
            if hash != meta.sha256 {
                return Err(PoxError::IntegrityHashMismatch {
                    tensor: meta.name.clone(),
                    expected: meta.sha256.clone(),
                    calculated: hash,
                });
            }
        }
        Ok(())
    }
}

/// Builder and serializer for .pox container files
pub struct PoxWriter {
    metadata: PoxMetadata,
    quant_plan: PoxQuantPlanRecord,
    manifest: PoxManifest,
    signature: Option<PoxSignatureBlock>,
    tensors: Vec<TensorMeta>,
    payload_buffer: Vec<u8>,
}

impl PoxWriter {
    pub fn new(metadata: PoxMetadata, quant_plan: PoxQuantPlanRecord, run_id: String) -> Self {
        Self {
            metadata,
            quant_plan,
            manifest: PoxManifest {
                run_id,
                source_model_sha256: String::new(),
                calibration_sha256: None,
                compiler_version: env!("CARGO_PKG_VERSION").to_string(),
                target_arch: std::env::consts::ARCH.to_string(),
                tensor_hashes: HashMap::new(),
                artifact_sha256: String::new(),
            },
            signature: None,
            tensors: Vec::new(),
            payload_buffer: Vec::new(),
        }
    }

    pub fn set_source_hash(&mut self, hash: String) {
        self.manifest.source_model_sha256 = hash;
    }

    pub fn set_calibration_hash(&mut self, hash: Option<String>) {
        self.manifest.calibration_sha256 = hash;
    }

    pub fn set_signature(&mut self, pubkey_hex: String, signature_hex: String) {
        self.signature = Some(PoxSignatureBlock {
            public_key_hex: pubkey_hex,
            signature_hex,
        });
    }

    /// Add a tensor with data, optional scales, and optional outliers.
    /// Ensures 64-byte alignment of each chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: String,
        shape: paraoxidizer_core::tensor::Shape,
        dtype: paraoxidizer_core::tensor::DType,
        group_size: paraoxidizer_core::tensor::QuantGroupSize,
        data: &[u8],
        scales: Option<&[u8]>,
        outliers: Option<&[u8]>,
    ) {
        // Calculate hash of payload
        let mut hasher = Sha256::new();
        hasher.update(data);
        if let Some(s) = scales {
            hasher.update(s);
        }
        if let Some(o) = outliers {
            hasher.update(o);
        }
        let tensor_sha256 = hex::encode(hasher.finalize());
        self.manifest
            .tensor_hashes
            .insert(name.clone(), tensor_sha256.clone());

        // Align data offset to 64 bytes
        Self::pad_to_alignment(&mut self.payload_buffer, DATA_ALIGNMENT);
        let data_offset_in_payload = self.payload_buffer.len() as u64;
        self.payload_buffer.extend_from_slice(data);
        let data_len = data.len() as u64;

        let (scale_offset_in_payload, scale_len) = if let Some(s) = scales {
            Self::pad_to_alignment(&mut self.payload_buffer, DATA_ALIGNMENT);
            let off = self.payload_buffer.len() as u64;
            self.payload_buffer.extend_from_slice(s);
            (off, s.len() as u64)
        } else {
            (0, 0)
        };

        let (outlier_offset_in_payload, outlier_len) = if let Some(o) = outliers {
            Self::pad_to_alignment(&mut self.payload_buffer, DATA_ALIGNMENT);
            let off = self.payload_buffer.len() as u64;
            self.payload_buffer.extend_from_slice(o);
            (off, o.len() as u64)
        } else {
            (0, 0)
        };

        self.tensors.push(TensorMeta {
            name,
            shape,
            dtype,
            group_size,
            data_offset: data_offset_in_payload,
            data_len,
            scale_offset: scale_offset_in_payload,
            scale_len,
            outlier_offset: outlier_offset_in_payload,
            outlier_len,
            sha256: tensor_sha256,
        });
    }

    fn pad_to_alignment(buf: &mut Vec<u8>, align: usize) {
        let remainder = buf.len() % align;
        if remainder != 0 {
            let padding = align - remainder;
            buf.resize(buf.len() + padding, 0);
        }
    }

    /// Compile into binary .pox bytes
    pub fn write_to_bytes(mut self) -> Result<Vec<u8>> {
        let metadata_bytes = serde_json::to_vec(&self.metadata)?;
        let quant_plan_bytes = serde_json::to_vec(&self.quant_plan)?;

        // Layout:
        // Header: 64 bytes (HEADER_SIZE)
        // Metadata: at HEADER_SIZE
        // Quant Plan: after Metadata (aligned)
        // Tensor Data: after Quant Plan (aligned to 64 bytes)
        // Tensor Index: after Tensor Data (with updated absolute offsets)
        // Manifest: after Tensor Index
        // Signature: after Manifest

        let mut out = Vec::with_capacity(1024 + self.payload_buffer.len());
        // Placeholder header
        out.resize(HEADER_SIZE as usize, 0);

        // 1. Metadata
        let metadata_offset = out.len() as u64;
        let metadata_len = metadata_bytes.len() as u64;
        out.extend_from_slice(&metadata_bytes);
        Self::pad_to_alignment(&mut out, DATA_ALIGNMENT);

        // 2. Quant Plan
        let quant_plan_offset = out.len() as u64;
        let quant_plan_len = quant_plan_bytes.len() as u64;
        out.extend_from_slice(&quant_plan_bytes);
        Self::pad_to_alignment(&mut out, DATA_ALIGNMENT);

        // 3. Tensor Data Payload
        let data_offset = out.len() as u64;
        let data_len = self.payload_buffer.len() as u64;
        out.extend_from_slice(&self.payload_buffer);
        Self::pad_to_alignment(&mut out, DATA_ALIGNMENT);

        // Update tensor metadata offsets to be absolute within the file
        for t in &mut self.tensors {
            t.data_offset += data_offset;
            if t.scale_len > 0 {
                t.scale_offset += data_offset;
            }
            if t.outlier_len > 0 {
                t.outlier_offset += data_offset;
            }
        }

        // 4. Tensor Index
        let tensor_index_bytes = serde_json::to_vec(&self.tensors)?;
        let tensor_index_offset = out.len() as u64;
        let tensor_index_len = tensor_index_bytes.len() as u64;
        out.extend_from_slice(&tensor_index_bytes);
        Self::pad_to_alignment(&mut out, DATA_ALIGNMENT);

        // Compute artifact SHA-256 up to this point
        let mut artifact_hasher = Sha256::new();
        artifact_hasher.update(&out[HEADER_SIZE as usize..]);
        self.manifest.artifact_sha256 = hex::encode(artifact_hasher.finalize());

        // 5. Manifest
        let manifest_bytes = serde_json::to_vec(&self.manifest)?;
        let manifest_offset = out.len() as u64;
        let manifest_len = manifest_bytes.len() as u64;
        out.extend_from_slice(&manifest_bytes);
        Self::pad_to_alignment(&mut out, DATA_ALIGNMENT);

        // 6. Signature (if present)
        let (signature_offset, signature_len) = if let Some(sig) = &self.signature {
            let sig_bytes = serde_json::to_vec(sig)?;
            let off = out.len() as u64;
            let len = sig_bytes.len() as u64;
            out.extend_from_slice(&sig_bytes);
            (off, len)
        } else {
            (0, 0)
        };

        // Write final header
        let header = PoxHeader {
            magic: *POX_MAGIC,
            format_version: POX_VERSION,
            flags: 0,
            metadata_offset,
            metadata_len,
            tensor_index_offset,
            tensor_index_len,
            quant_plan_offset,
            quant_plan_len,
            data_offset,
            data_len,
            manifest_offset,
            manifest_len,
            signature_offset,
            signature_len,
        };

        let mut header_buf = Vec::with_capacity(HEADER_SIZE as usize);
        header.write_to(&mut header_buf)?;
        out[..HEADER_SIZE as usize].copy_from_slice(&header_buf[..HEADER_SIZE as usize]);

        Ok(out)
    }

    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<()> {
        let bytes = self.write_to_bytes()?;
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }
}
