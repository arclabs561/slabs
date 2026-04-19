//! Chunk-size measurement strategy.

/// Measures the size of a chunk for size-budget comparisons.
///
/// [`CodeChunker`](crate::CodeChunker) uses a `ChunkSizer` to decide whether
/// a node fits within `max_chunk_size` and whether to merge atomic chunks.
/// Default: byte length via [`ByteSizer`]. Plug in a tokenizer-backed sizer
/// to size chunks in tokens — match your embedding model's actual context
/// limit instead of approximating with bytes.
///
/// `max_chunk_size` is interpreted in whatever unit the sizer returns —
/// bytes for the default `ByteSizer`, tokens for a tokenizer-backed sizer.
pub trait ChunkSizer: Send + Sync {
    /// Return the size of `text` in whatever unit this sizer measures.
    fn size(&self, text: &str) -> usize;
}

/// Default sizer: returns the byte length of the chunk text.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteSizer;

impl ChunkSizer for ByteSizer {
    fn size(&self, text: &str) -> usize {
        text.len()
    }
}
