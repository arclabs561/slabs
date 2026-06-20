#![warn(missing_docs)]
//! # slabs
//!
//! Retrieval spans and late pooling.
//!
//! `slabs` centers the [`Slab`] type: a text span with byte and character
//! offsets in the source document. Use slabs between document extraction,
//! annotation, embedding, and indexing.
//!
//! ## Core types
//!
//! ### `Slab`: a retrieval span
//!
//! [`Slab`] stores text plus byte and character offsets. It does not decide
//! how text should be split; it records boundaries from any source.
//!
//! ### `LateChunkingPooler`: pool token embeddings into chunk embeddings
//!
//! Late chunking (Günther et al. 2024, arXiv:2409.04701) embeds the full
//! document first so every token attends to the rest of the document,
//! then mean-pools token embeddings inside each chunk's byte span. The
//! result is a per-slab embedding that carries document-wide context:
//! pronouns, anaphora, and acronym definitions are no longer lost at
//! chunk boundaries.
//!
//! `LateChunkingPooler` is span-only: bring your own boundaries from any
//! source: `text-splitter`, parser output, regex, or hand-built `Slab`s.
//!
//! ## What slabs does not do
//!
//! - **General-purpose text chunking.** Use [`text-splitter`](https://crates.io/crates/text-splitter)
//!   for fixed/sentence/recursive prose splitting and code splitting.
//! - **Format conversion (PDF, HTML, DOCX).** Input is `&str`. Use
//!   [`deformat`](https://crates.io/crates/deformat) or
//!   [`pdf-extract`](https://crates.io/crates/pdf-extract) upstream.
//! - **Embedding generation.** `LateChunkingPooler` consumes
//!   pre-computed token embeddings; bring your own long-context model
//!   (Jina v2/v3, nomic-embed-text, candle, ort).
//! - **Vector store integration.** [`Slab`] is the boundary; enable the
//!   `serde` feature and wire to qdrant-client, lancedb, sqlx, etc. yourself.
//!
//! ## Quick start (retrieval spans)
//!
//! ```ignore
//! use slabs::Slab;
//!
//! let slab = Slab::new("Ada designed the engine.", 0, 24, 0)
//!     .with_char_offsets(0, 24);
//! ```
//!
//! ## Quick start (late pooling)
//!
//! ```ignore
//! use slabs::{LateChunkingPooler, Slab};
//!
//! // Bring your own spans (text-splitter, deformat, anno, parser output, ...).
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
mod slab;

pub use error::{Error, Result};
pub use late::LateChunkingPooler;
pub use slab::{compute_char_offsets, slabs_from_byte_ranges, slabs_from_char_ranges, Slab};

/// A chunking strategy: text in, [`Slab`]s out.
///
/// Implementors override [`chunk_bytes`](Chunker::chunk_bytes); the default
/// [`chunk`](Chunker::chunk) method adds Unicode character offsets.
///
/// Slabs does not ship boundary finders. The trait is public so users can
/// wrap external chunkers (`text-splitter`, regex, parser output, custom
/// logic) and feed the output into [`LateChunkingPooler`].
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
