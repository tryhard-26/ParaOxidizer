//! Model file formats for ParaOxidizer (.pox container, SafeTensors, GGUF, and .poxcal).

#![allow(unknown_lints)]
#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if,
    clippy::chunks_exact_to_as_chunks
)]

pub mod gguf;
pub mod hf;
pub mod pox;
pub mod poxcal;
pub mod safetensors;

pub use gguf::{GgufMetadata, GgufReader};
pub use hf::{HfConfigJson, HfModel, HfShardedIndex};
pub use pox::{
    PoxFile, PoxHeader, PoxManifest, PoxMetadata, PoxQuantPlanRecord, PoxSignatureBlock, PoxWriter,
    DATA_ALIGNMENT, HEADER_SIZE, POX_MAGIC, POX_VERSION,
};
pub use poxcal::{LayerActivationStats, PoxCalArtifact, POXCAL_MAGIC, POXCAL_VERSION};
pub use safetensors::SafeTensorsModel;
