//! Fixed-size chunking with overlap.
//!
//! The simplest chunking strategy: split every N bytes with M bytes of overlap.
//!
//! ## How It Works
//!
//! ```text
//! size = 10, overlap = 3
//!
//! Document: "abcdefghijklmnopqrstuvwxyz"
//!
//! Chunk 0: "abcdefghij"   [0..10]
//! Chunk 1: "hijklmnopq"   [7..17]   <- starts at 10 - 3 = 7
//! Chunk 2: "opqrstuvwx"   [14..24]  <- starts at 17 - 3 = 14
//! Chunk 3: "vwxyz"        [21..26]  <- final chunk may be shorter
//! ```
//!
//! ## Why Overlap?
//!
//! Without overlap, information at chunk boundaries is lost. If a key sentence
//! spans two chunks, neither chunk captures it fully:
//!
//! ```text
//! "The answer is 42"
//!         ↓
//! No overlap:  ["The answer i", "s 42"]  <- broken!
//! With overlap: ["The answer is", "answer is 42"] <- both have context
//! ```
//!
//! ## Trade-offs
//!
//! | Overlap | Storage | Retrieval | Risk |
//! |---------|---------|-----------|------|
//! | 0% | Minimal | Poor at boundaries | Info loss |
//! | 10-20% | Low | Good | Sweet spot |
//! | 50%+ | High | Redundant | Wasted compute |
//!
//! A common heuristic: 10-20% overlap (e.g., size=500, overlap=50-100).

use crate::{Chunker, Slab};

/// Fixed-size chunker with configurable overlap.
///
/// ## Example
///
/// ```rust
/// use slabs::{Chunker, FixedChunker};
///
/// let chunker = FixedChunker::new(100, 20);
/// let text = "A".repeat(250);
/// let slabs = chunker.chunk(&text);
///
/// // 250 bytes with step=80: starts at 0, 80, 160 = 3 chunks
/// // (240 would start a 4th but 240+100 > 250, and remainder < step)
/// assert!(slabs.len() >= 3);
/// assert_eq!(slabs[0].len(), 100);
/// assert_eq!(slabs[1].start, 80); // 100 - 20 overlap
/// ```
#[derive(Debug, Clone)]
pub struct FixedChunker {
    size: usize,
    overlap: usize,
}

impl FixedChunker {
    /// Create a new fixed-size chunker.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum chunk size in bytes
    /// * `overlap` - Bytes to overlap between adjacent chunks
    ///
    /// # Panics
    ///
    /// Panics if `size == 0` or `overlap >= size`.
    #[must_use]
    pub fn new(size: usize, overlap: usize) -> Self {
        assert!(size > 0, "chunk size must be > 0");
        assert!(overlap < size, "overlap must be < size");
        Self { size, overlap }
    }

    /// Create a chunker with no overlap.
    #[must_use]
    pub fn no_overlap(size: usize) -> Self {
        Self::new(size, 0)
    }

    /// The step size between chunk starts.
    #[must_use]
    fn step(&self) -> usize {
        self.size - self.overlap
    }
}

impl Chunker for FixedChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        let step = self.step();
        let mut slabs = Vec::with_capacity(self.estimate_chunks(text.len()));
        let mut start = 0;
        let mut index = 0;

        while start < text.len() {
            // Find end, clamped to text length
            let end = (start + self.size).min(text.len());

            // Ensure we're at a char boundary
            // Replaces text.floor_char_boundary(end) for MSRV < 1.80 compatibility
            let mut end = end;
            while !text.is_char_boundary(end) {
                end -= 1;
            }

            if end > start {
                slabs.push(Slab::new(&text[start..end], start, end, index));
                index += 1;
            }

            // Move to next chunk
            let next_start = start + step;
            if next_start >= text.len() || next_start <= start {
                break;
            }

            // Ensure next start is at a char boundary
            // Replaces text.ceil_char_boundary(next_start) for MSRV < 1.80 compatibility
            start = next_start;
            while start < text.len() && !text.is_char_boundary(start) {
                start += 1;
            }
        }

        slabs
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        if text_len == 0 {
            return 0;
        }
        let step = self.step();
        text_len.div_ceil(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let chunker = FixedChunker::new(10, 2);
        let text = "abcdefghijklmnopqrstuvwxyz";
        let slabs = chunker.chunk(text);

        assert_eq!(slabs[0].text, "abcdefghij");
        assert_eq!(slabs[0].start, 0);
        assert_eq!(slabs[0].end, 10);

        assert_eq!(slabs[1].start, 8); // 10 - 2 overlap
    }

    #[test]
    fn test_empty_text() {
        let chunker = FixedChunker::new(10, 2);
        let slabs = chunker.chunk("");
        assert!(slabs.is_empty());
    }

    #[test]
    fn test_text_smaller_than_chunk() {
        let chunker = FixedChunker::new(100, 20);
        let slabs = chunker.chunk("small");
        assert_eq!(slabs.len(), 1);
        assert_eq!(slabs[0].text, "small");
    }

    #[test]
    fn test_unicode_boundaries() {
        let chunker = FixedChunker::new(5, 1);
        let text = "a日本語b"; // 'a' + 3 multibyte chars + 'b'
        let slabs = chunker.chunk(text);

        // Should not panic on multibyte boundaries
        for slab in &slabs {
            assert!(slab.text.is_char_boundary(0));
        }
    }

    #[test]
    #[should_panic]
    fn test_zero_size_panics() {
        let _ = FixedChunker::new(0, 0);
    }

    #[test]
    #[should_panic]
    fn test_overlap_exceeds_size_panics() {
        let _ = FixedChunker::new(10, 10);
    }
}
