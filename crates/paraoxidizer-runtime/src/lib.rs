//! Lightweight quantized Transformer inference engine, KV cache, tokenizer, and sampler.

#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::collapsible_if
)]

pub mod engine;
pub mod eval;
pub mod metal_backend;
pub mod paged_cache;
pub mod sampler;
pub mod tokenizer;

pub use engine::{KvCache, PoxEngine};
pub use eval::{
    compute_kl_divergence, compute_nll, compute_perplexity, compute_top1_agreement,
    compute_topk_agreement, log_sum_exp,
};
pub use metal_backend::MetalBackend;
pub use paged_cache::{BlockManager, BlockTable, PagedKvCache};
pub use sampler::{Sampler, SamplerConfig};
pub use tokenizer::PoxTokenizer;
