use byteorder::{LittleEndian, ReadBytesExt};
use paraoxidizer_core::error::{PoxError, Result};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub kv_pairs: HashMap<String, String>,
}

pub struct GgufReader {
    pub metadata: GgufMetadata,
}

impl GgufReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != GGUF_MAGIC {
            return Err(PoxError::InvalidMagic {
                expected: *GGUF_MAGIC,
                found: magic,
            });
        }

        let version = reader.read_u32::<LittleEndian>()?;
        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let kv_count = reader.read_u64::<LittleEndian>()?;

        let mut kv_pairs = HashMap::new();
        // Parse basic string keys and values if available
        for _ in 0..kv_count.min(256) {
            if let Ok(key) = Self::read_gguf_string(&mut reader) {
                let val_type = reader.read_u32::<LittleEndian>().unwrap_or(999);
                match val_type {
                    8 => {
                        // String value
                        if let Ok(val) = Self::read_gguf_string(&mut reader) {
                            kv_pairs.insert(key, val);
                        }
                    }
                    4 => {
                        // u32
                        if let Ok(val) = reader.read_u32::<LittleEndian>() {
                            kv_pairs.insert(key, val.to_string());
                        }
                    }
                    5 => {
                        // i32
                        if let Ok(val) = reader.read_i32::<LittleEndian>() {
                            kv_pairs.insert(key, val.to_string());
                        }
                    }
                    6 => {
                        // f32
                        if let Ok(val) = reader.read_f32::<LittleEndian>() {
                            kv_pairs.insert(key, val.to_string());
                        }
                    }
                    _ => {
                        // Skip remaining or break
                        break;
                    }
                }
            } else {
                break;
            }
        }

        Ok(Self {
            metadata: GgufMetadata {
                version,
                tensor_count,
                kv_count,
                kv_pairs,
            },
        })
    }

    fn read_gguf_string<R: Read>(reader: &mut R) -> Result<String> {
        let len = reader.read_u64::<LittleEndian>()? as usize;
        if len > 65536 {
            return Err(PoxError::Format("GGUF string too long".into()));
        }
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| PoxError::Format(e.to_string()))
    }
}
