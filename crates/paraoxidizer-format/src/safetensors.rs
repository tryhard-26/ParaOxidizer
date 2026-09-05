#![allow(clippy::all, unknown_lints)]

use half::{bf16, f16};
use memmap2::Mmap;
use paraoxidizer_core::{
    error::{PoxError, Result},
    tensor::{DType, Shape},
};
use safetensors::tensor::{Dtype as StDtype, SafeTensors};
use std::{collections::HashMap, fs::File, path::Path};

pub struct SafeTensorsModel {
    _mmap: Mmap,
    tensors: HashMap<String, (Shape, DType, Vec<f32>)>,
}

impl SafeTensorsModel {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| PoxError::Format(format!("Failed to parse SafeTensors: {e}")))?;

        let mut tensors = HashMap::new();

        for (name, view) in st.tensors() {
            let shape = Shape::new(view.shape().to_vec());
            let raw_data = view.data();
            let (dtype, floats) = match view.dtype() {
                StDtype::F32 => {
                    let floats: Vec<f32> = raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    (DType::F32, floats)
                }
                StDtype::F16 => {
                    let floats: Vec<f32> = raw_data
                        .chunks_exact(2)
                        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    (DType::F16, floats)
                }
                StDtype::BF16 => {
                    let floats: Vec<f32> = raw_data
                        .chunks_exact(2)
                        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect();
                    (DType::BF16, floats)
                }
                StDtype::U8 => {
                    let floats: Vec<f32> = raw_data.iter().map(|&b| b as f32).collect();
                    (DType::F32, floats)
                }
                StDtype::I8 => {
                    let floats: Vec<f32> = raw_data.iter().map(|&b| (b as i8) as f32).collect();
                    (DType::F32, floats)
                }
                StDtype::I32 => {
                    let floats: Vec<f32> = raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect();
                    (DType::F32, floats)
                }
                StDtype::I64 => {
                    let floats: Vec<f32> = raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect();
                    (DType::F32, floats)
                }
                StDtype::BOOL => {
                    let floats: Vec<f32> = raw_data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect();
                    (DType::F32, floats)
                }
                other => {
                    return Err(PoxError::Format(format!(
                        "Unsupported SafeTensors dtype: {:?}",
                        other
                    )));
                }
            };

            tensors.insert(name.to_string(), (shape, dtype, floats));
        }

        Ok(Self {
            _mmap: mmap,
            tensors,
        })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn get_tensor(&self, name: &str) -> Option<&(Shape, DType, Vec<f32>)> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> &HashMap<String, (Shape, DType, Vec<f32>)> {
        &self.tensors
    }
}
