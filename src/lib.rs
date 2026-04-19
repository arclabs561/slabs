#![warn(missing_docs)]
//! # code-chunker
//!
//! AST-aware code chunking and late chunking for RAG pipelines.
//!
//! ## Two primitives
//!
//! ### `CodeChunker` — split source code at AST boundaries
//!
//! Tree-sitter walks the parse tree and produces chunks aligned to
//! function, class, impl, and module boundaries. When a node fits the
//! configured size budget it is kept intact; oversize nodes are split
//! recursively at structural separators. Supports Rust, Python,
//! TypeScript/JavaScript, and Go (behind the `code` feature).
//!
//! ### `LateChunkingPooler` — pool token embeddings into chunk embeddings
//!
//! Late chunking (Günther et al. 2024, arXiv:2409.04701) embeds the full
//! document first so every token attends to the rest of the document,
//! then mean-pools token embeddings inside each chunk's byte span. The
//! result is a per-chunk embedding that carries document-wide context —
//! pronouns, anaphora, and acronym definitions are no longer lost at
//! chunk boundaries.
//!
//! `LateChunkingPooler` is span-only: bring your own boundaries from any
//! source — `CodeChunker`, `text-splitter`, regex, or hand-built `Slab`s.
//!
//! ## What this crate does not do
//!
//! - **General-purpose text chunking.** Use [`text-splitter`](https://crates.io/crates/text-splitter)
//!   for fixed/sentence/recursive prose splitting; it's the de-facto Rust
//!   standard with broader Unicode and tokenizer support.
//! - **Format conversion (PDF, HTML, DOCX).** Input is `&str`. Use
//!   [`deformat`](https://crates.io/crates/deformat) or
//!   [`pdf-extract`](https://crates.io/crates/pdf-extract) upstream.
//! - **Embedding generation.** `LateChunkingPooler` consumes
//!   pre-computed token embeddings; bring your own long-context model
//!   (Jina v2/v3, nomic-embed-text, candle, ort).
//! - **Vector store integration.** [`Slab`] is the boundary; enable the
//!   `serde` feature and wire to qdrant-client, lancedb, sqlx, etc. yourself.
//!
//! ## Quick start (code chunking)
//!
//! ```ignore
//! use code_chunker::{Chunker, CodeChunker, CodeLanguage};
//!
//! let chunker = CodeChunker::new(CodeLanguage::Rust, 1500, 0);
//! let slabs = chunker.chunk(source_code);
//! ```
//!
//! ## Quick start (late chunking)
//!
//! ```ignore
//! use code_chunker::{LateChunkingPooler, Slab};
//!
//! // Bring your own chunk boundaries (text-splitter, CodeChunker, ...).
//! let chunks: Vec<Slab> = my_chunker(&document);
//!
//! // Embed the full document with a long-context model.
//! let token_embeddings: Vec<Vec<f32>> = my_model.embed_tokens(&document);
//!
//! // Pool token embeddings into per-chunk embeddings.
//! let pooler = LateChunkingPooler::new(384);
//! let chunk_embeddings = pooler.pool(&token_embeddings, &chunks, document.len());
//! ```

mod error;
mod late;
mod sizer;
mod slab;

#[cfg(feature = "code")]
mod code;
#[cfg(feature = "code")]
mod recursive;

pub use error::{Error, Result};
pub use late::LateChunkingPooler;
pub use sizer::{ByteSizer, ChunkSizer};
pub use slab::{compute_char_offsets, Slab};

#[cfg(feature = "code")]
pub use code::{CodeChunker, CodeLanguage};

/// A chunking strategy: text in, [`Slab`]s out.
///
/// Implementors override [`chunk_bytes`](Chunker::chunk_bytes); the default
/// [`chunk`](Chunker::chunk) method adds Unicode character offsets.
///
/// This crate only ships one public chunker — [`CodeChunker`] — but the
/// trait is public so users can wrap external chunkers (text-splitter,
/// regex, custom logic) and feed the output into [`LateChunkingPooler`].
pub trait Chunker: Send + Sync {
    /// Core chunking implementation returning [`Slab`]s with byte offsets only.
    ///
    /// Implementors override this method. Users should call [`chunk`](Chunker::chunk)
    /// instead, which adds character offsets automatically.
    fn chunk_bytes(&self, text: &str) -> Vec<Slab>;

    /// Split text into chunks with both byte and character offsets.
    ///
    /// This calls [`chunk_bytes`](Chunker::chunk_bytes) and then computes
    /// Unicode character offsets on every slab. Users get correct `char_start`
    /// and `char_end` without manual work.
    fn chunk(&self, text: &str) -> Vec<Slab> {
        let mut slabs = self.chunk_bytes(text);
        compute_char_offsets(text, &mut slabs);
        slabs
    }

    /// Estimate the number of chunks for a given text length.
    ///
    /// Useful for pre-allocation. May be approximate.
    fn estimate_chunks(&self, text_len: usize) -> usize {
        // Conservative default
        (text_len / 500).max(1)
    }
}
