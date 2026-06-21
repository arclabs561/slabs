#![warn(missing_docs)]
//! # slabs
//!
//! Retrieval spans and late pooling.
//!
//! `slabs` centers the [`Slab`] type: a text span with byte and character
//! offsets in the exact source string used to create it. Use slabs between
//! document extraction, annotation, embedding, and indexing.
//!
//! ## Core types
//!
//! ### `Slab`: a retrieval span
//!
//! [`Slab`] stores text plus byte and character offsets. It does not decide
//! how text should be split; it records boundaries from any source.
//!
//! Offsets are source-string relative. If text is normalized, extracted, or
//! otherwise transformed before slab construction, the offsets refer to that
//! transformed string, not to an earlier document representation.
//!
//! ### `LateChunkingPooler`: pool token embeddings into span embeddings
//!
//! Late chunking (Günther et al. 2024, arXiv:2409.04701) embeds the full
//! document first so every token attends to the rest of the document,
//! then mean-pools token embeddings inside each slab's span and L2-normalizes
//! the result. The output is a fixed-width vector for each slab.
//!
//! `LateChunkingPooler` is span-only: bring your own boundaries from any
//! source: `text-splitter`, parser output, regex, or hand-built `Slab`s.
//!
//! ## Scope
//!
//! - Boundary finding is upstream.
//! - Format conversion is upstream; input is already `&str`.
//! - Embedding generation is upstream; [`LateChunkingPooler`] consumes token
//!   vectors.
//! - Storage is downstream; enable the `serde` feature when spans need to cross
//!   a storage or service boundary.
//! - Cross-file analysis is out of scope; a slab refers to one source string.
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
//! let spans: Vec<Slab> = boundary_source(&document);
//!
//! // Embed the full document with a long-context model.
//! let token_embeddings: Vec<Vec<f32>> = my_model.embed_tokens(&document);
//!
//! // Pool token embeddings into per-span embeddings.
//! let pooler = LateChunkingPooler::new(384);
//! let span_embeddings = pooler.pool(&token_embeddings, &spans, document.len());
//! ```

mod error;
mod late;
mod slab;

pub use error::{Error, Result};
pub use late::LateChunkingPooler;
pub use slab::{compute_char_offsets, slabs_from_byte_ranges, slabs_from_char_ranges, Slab};

/// A source of already-chosen [`Slab`] boundaries.
///
/// Implementors choose or receive text boundaries elsewhere, then return
/// slabs for those boundaries. This trait exists for adapters around
/// `text-splitter`, parser output, regex matches, extraction spans, or
/// product-specific boundary logic.
pub trait SlabSource: Send + Sync {
    /// Return [`Slab`]s with byte offsets only.
    ///
    /// Implementors override this method. Users should call
    /// [`slabs`](SlabSource::slabs) instead, which adds character offsets
    /// automatically.
    fn slab_bytes(&self, text: &str) -> Vec<Slab>;

    /// Return slabs with both byte and character offsets.
    ///
    /// Offsets are relative to the exact `text` argument passed here.
    fn slabs(&self, text: &str) -> Vec<Slab> {
        let mut slabs = self.slab_bytes(text);
        compute_char_offsets(text, &mut slabs);
        slabs
    }

    /// Estimate the number of slabs for a given text length.
    ///
    /// Useful for pre-allocation. May be approximate.
    fn estimate_slabs(&self, text_len: usize) -> usize {
        (text_len / 500).max(1)
    }
}

/// Compatibility adapter trait: text in, [`Slab`]s out.
///
/// Implementors override [`chunk_bytes`](Chunker::chunk_bytes); the default
/// [`chunk`](Chunker::chunk) method adds Unicode character offsets.
///
/// Slabs does not ship boundary finders. The trait is public so users can
/// wrap external chunkers (`text-splitter`, regex, parser output, custom
/// logic) and feed the output into [`LateChunkingPooler`].
///
/// Prefer [`SlabSource`] for new adapters. `Chunker` remains available for
/// existing code that already uses chunking vocabulary at the boundary source.
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
        (text_len / 500).max(1)
    }
}

impl<T: Chunker + ?Sized> SlabSource for T {
    fn slab_bytes(&self, text: &str) -> Vec<Slab> {
        self.chunk_bytes(text)
    }

    fn slabs(&self, text: &str) -> Vec<Slab> {
        self.chunk(text)
    }

    fn estimate_slabs(&self, text_len: usize) -> usize {
        self.estimate_chunks(text_len)
    }
}
