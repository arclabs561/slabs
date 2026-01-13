//! # slabs
//!
//! Text chunking for retrieval-augmented generation (RAG) pipelines.
//!
//! ## The Problem
//!
//! Language models have context windows. Documents don't fit. You need to split
//! them into pieces ("chunks") small enough to embed and retrieve, but large
//! enough to preserve meaning.
//!
//! This sounds trivial—just split every N characters, right? But consider:
//!
//! - A sentence split mid-word is garbage
//! - A paragraph split mid-argument loses coherence
//! - A code block split mid-function is useless
//! - Overlap is needed for context continuity, but how much?
//!
//! The right chunking strategy depends on your content and retrieval needs.
//!
//! ## Chunking Strategies
//!
//! ### Fixed Size (Baseline)
//!
//! The simplest approach: split every N characters with M overlap.
//!
//! ```text
//! Document: "The quick brown fox jumps over the lazy dog."
//! Size: 20, Overlap: 5
//!
//! Chunk 0: "The quick brown fox "  [0..20]
//! Chunk 1: " fox jumps over the "  [15..35]  <- overlap preserves "fox"
//! Chunk 2: " the lazy dog."        [30..44]
//! ```
//!
//! **When to use**: Homogeneous content (logs, code), baseline comparisons.
//! **Weakness**: Ignores linguistic boundaries—splits mid-sentence.
//!
//! ### Sentence-Based
//!
//! Split on sentence boundaries, group N sentences per chunk.
//!
//! The key insight: sentence boundaries are surprisingly hard to detect.
//! "Dr. Smith went to Washington D.C. on Jan. 15th." has 1 sentence, not 4.
//! We use Unicode segmentation (UAX #29) which handles most edge cases.
//!
//! **When to use**: Prose, articles, documentation.
//! **Weakness**: Very short or very long sentences cause imbalanced chunks.
//!
//! ### Recursive (LangChain-style)
//!
//! Try splitting on paragraph breaks first. If chunks are still too large,
//! split on sentence breaks. If still too large, split on words. Last resort:
//! split on characters.
//!
//! ```text
//! Separators: ["\n\n", "\n", ". ", " ", ""]
//!
//! 1. Try splitting on "\n\n" (paragraphs)
//! 2. Any chunk > max_size? Split that chunk on "\n" (lines)
//! 3. Still > max_size? Split on ". " (sentences)
//! 4. Still > max_size? Split on " " (words)
//! 5. Still > max_size? Split on "" (characters)
//! ```
//!
//! **When to use**: General-purpose, mixed content.
//! **Weakness**: Separator hierarchy is heuristic, not semantic.
//!
//! ### Semantic (Embedding-Based)
//!
//! Embed each sentence, compute similarity between adjacent sentences,
//! split where similarity drops below a threshold.
//!
//! ```text
//! Sentences:  [S1, S2, S3, S4, S5, S6]
//! Embeddings: [E1, E2, E3, E4, E5, E6]
//! Similarities: [sim(1,2)=0.9, sim(2,3)=0.8, sim(3,4)=0.3, sim(4,5)=0.85, sim(5,6)=0.7]
//!                                              ↑
//!                                         Topic shift!
//!
//! Chunks: [S1, S2, S3] | [S4, S5, S6]
//! ```
//!
//! **When to use**: When topic coherence matters more than size uniformity.
//! **Weakness**: Requires embedding model, slower, threshold is a hyperparameter.
//!
//! ## Quick Start
//!
//! ```rust
//! use slabs::{Chunker, FixedChunker, SentenceChunker, RecursiveChunker};
//!
//! let text = "The quick brown fox jumps over the lazy dog. \
//!             Pack my box with five dozen liquor jugs.";
//!
//! // Fixed size
//! let chunker = FixedChunker::new(50, 10);
//! let slabs = chunker.chunk(text);
//!
//! // Sentence-based (2 sentences per chunk)
//! let chunker = SentenceChunker::new(2);
//! let slabs = chunker.chunk(text);
//!
//! // Recursive with custom separators
//! let chunker = RecursiveChunker::new(100, &["\n\n", "\n", ". ", " "]);
//! let slabs = chunker.chunk(text);
//! ```
//!
//! ## Semantic Chunking (requires `semantic` feature)
//!
//! ```rust,ignore
//! use slabs::{Chunker, SemanticChunker};
//!
//! let chunker = SemanticChunker::new(0.5)?; // threshold
//! let slabs = chunker.chunk(long_document);
//! ```
//!
//! ## Performance Considerations
//!
//! | Strategy | Speed | Quality | Memory |
//! |----------|-------|---------|--------|
//! | Fixed | O(n) | Low | O(1) |
//! | Sentence | O(n) | Medium | O(n) |
//! | Recursive | O(n log n) | Medium | O(n) |
//! | Semantic | O(n × d) | High | O(n × d) |
//!
//! Where n = document length, d = embedding dimension.
//!
//! For most RAG applications, **Recursive** is the sweet spot.
//! Use **Semantic** when retrieval quality justifies the cost.

mod capacity;
mod error;
mod fixed;
mod recursive;
mod sentence;
mod slab;

#[cfg(feature = "semantic")]
mod semantic;

pub use capacity::{ChunkCapacity, ChunkCapacityError};
pub use error::{Error, Result};
pub use fixed::FixedChunker;
pub use recursive::RecursiveChunker;
pub use sentence::SentenceChunker;
pub use slab::Slab;

#[cfg(feature = "semantic")]
pub use semantic::SemanticChunker;

/// A text chunking strategy.
///
/// All chunkers implement this trait, enabling polymorphic usage:
///
/// ```rust
/// use slabs::{Chunker, FixedChunker, SentenceChunker};
///
/// fn chunk_document(chunker: &dyn Chunker, text: &str) -> Vec<slabs::Slab> {
///     chunker.chunk(text)
/// }
///
/// let fixed = FixedChunker::new(100, 20);
/// let sentence = SentenceChunker::new(3);
///
/// let text = "Hello world. This is a test.";
/// let slabs1 = chunk_document(&fixed, text);
/// let slabs2 = chunk_document(&sentence, text);
/// ```
pub trait Chunker: Send + Sync {
    /// Split text into chunks.
    ///
    /// Each chunk is a [`Slab`] containing the text and its byte offsets
    /// in the original document.
    fn chunk(&self, text: &str) -> Vec<Slab>;

    /// Estimate the number of chunks for a given text length.
    ///
    /// Useful for pre-allocation. May be approximate.
    fn estimate_chunks(&self, text_len: usize) -> usize {
        // Conservative default
        (text_len / 500).max(1)
    }
}
