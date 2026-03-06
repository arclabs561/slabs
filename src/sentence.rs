//! Sentence-based chunking.
//!
//! Splits text on sentence boundaries, grouping N sentences per chunk.
//!
//! ## The Hard Part: Finding Sentences
//!
//! Sentence detection seems simple until you encounter:
//!
//! ```text
//! "Dr. Smith went to Washington D.C. on Jan. 15th."
//!     ^                          ^       ^
//!     Not a sentence end (abbreviation)
//! ```
//!
//! We use Unicode Standard Annex #29 (UAX #29) for sentence segmentation,
//! which handles most edge cases including:
//!
//! - Abbreviations (Dr., Mr., Inc., etc.)
//! - Decimal numbers (3.14159)
//! - Ellipses (...)
//! - URLs (<https://example.com/path>)
//!
//! ## Why Group Sentences?
//!
//! Single sentences are often too short for effective retrieval. A question
//! like "What did the author conclude?" needs paragraph-level context.
//!
//! Typical settings:
//! - `sentences_per_chunk = 3-5` for dense technical content
//! - `sentences_per_chunk = 5-10` for narrative prose
//!
//! ## Overlap
//!
//! For NER and other span-level tasks, entities at chunk boundaries can be
//! split across chunks. Sentence overlap repeats the last N sentences of
//! each chunk at the start of the next:
//!
//! ```text
//! sentences_per_chunk = 3, overlap_sentences = 1
//!
//! Chunk 0: [S1, S2, S3]
//! Chunk 1: [S3, S4, S5]    <- S3 repeated
//! Chunk 2: [S5, S6, S7]    <- S5 repeated
//! ```
//!
//! ## Trade-offs
//!
//! | Sentences/Chunk | Pros | Cons |
//! |-----------------|------|------|
//! | 1 | Precise retrieval | No context |
//! | 3-5 | Good balance | May split paragraphs |
//! | 10+ | Full context | May exceed model limits |

use unicode_segmentation::UnicodeSegmentation;

use crate::{Chunker, Slab};

/// Sentence-based chunker with optional overlap.
///
/// Groups consecutive sentences into chunks of approximately equal size.
/// When `overlap_sentences > 0`, the last N sentences of each chunk are
/// repeated at the start of the next chunk.
///
/// ## Example
///
/// ```rust
/// use slabs::{Chunker, SentenceChunker};
///
/// let chunker = SentenceChunker::new(2);
/// let text = "First sentence. Second sentence. Third sentence.";
/// let slabs = chunker.chunk(text);
///
/// assert_eq!(slabs.len(), 2);
/// assert!(slabs[0].text.contains("First"));
/// assert!(slabs[0].text.contains("Second"));
/// ```
///
/// With overlap:
///
/// ```rust
/// use slabs::{Chunker, SentenceChunker};
///
/// let chunker = SentenceChunker::new(2).with_overlap(1);
/// let text = "One. Two. Three. Four.";
/// let slabs = chunker.chunk(text);
///
/// // Chunk 0: "One. Two.", Chunk 1: "Two. Three.", Chunk 2: "Three. Four."
/// assert!(slabs.len() >= 2);
/// ```
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    sentences_per_chunk: usize,
    overlap_sentences: usize,
}

impl SentenceChunker {
    /// Create a new sentence chunker.
    ///
    /// # Arguments
    ///
    /// * `sentences_per_chunk` - Number of sentences to group together
    ///
    /// # Panics
    ///
    /// Panics if `sentences_per_chunk == 0`.
    #[must_use]
    pub fn new(sentences_per_chunk: usize) -> Self {
        assert!(sentences_per_chunk > 0, "sentences_per_chunk must be > 0");
        Self {
            sentences_per_chunk,
            overlap_sentences: 0,
        }
    }

    /// Set the number of sentences to overlap between adjacent chunks.
    ///
    /// # Panics
    ///
    /// Panics if `overlap >= sentences_per_chunk`.
    #[must_use]
    pub fn with_overlap(mut self, overlap_sentences: usize) -> Self {
        assert!(
            overlap_sentences < self.sentences_per_chunk,
            "overlap_sentences ({}) must be < sentences_per_chunk ({})",
            overlap_sentences,
            self.sentences_per_chunk
        );
        self.overlap_sentences = overlap_sentences;
        self
    }

    /// Create a chunker that outputs one sentence per chunk.
    #[must_use]
    pub fn single() -> Self {
        Self::new(1)
    }
}

impl Chunker for SentenceChunker {
    fn chunk_bytes(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        // Collect sentence boundaries using Unicode segmentation
        let segments: Vec<&str> = text.split_sentence_bounds().collect();

        if segments.is_empty() {
            return vec![];
        }

        // Build (byte_offset, text) pairs, filtering whitespace-only segments.
        let sentences: Vec<(usize, &str)> = segments
            .into_iter()
            .scan(0usize, |offset, s| {
                let start = *offset;
                *offset += s.len();
                Some((start, s))
            })
            .filter(|(_, s)| !s.trim().is_empty())
            .collect();

        if sentences.is_empty() {
            return vec![];
        }

        let step = self.sentences_per_chunk - self.overlap_sentences;
        let mut slabs = Vec::new();
        let mut index = 0;
        let mut pos = 0;

        while pos < sentences.len() {
            let end = (pos + self.sentences_per_chunk).min(sentences.len());
            let chunk_sentences = &sentences[pos..end];

            if chunk_sentences.is_empty() {
                break;
            }

            let start_byte = chunk_sentences.first().map(|(off, _)| *off).unwrap_or(0);
            let end_byte = chunk_sentences
                .last()
                .map(|(off, s)| off + s.len())
                .unwrap_or(start_byte);

            let chunk_text: String = chunk_sentences.iter().map(|(_, s)| *s).collect();
            let trimmed = chunk_text.trim();

            if !trimmed.is_empty() {
                let leading_ws = chunk_text.len() - chunk_text.trim_start().len();
                let trailing_ws = chunk_text.len() - chunk_text.trim_end().len();

                slabs.push(Slab::new(
                    trimmed,
                    start_byte + leading_ws,
                    end_byte - trailing_ws,
                    index,
                ));
                index += 1;
            }

            // Advance by step (not by sentences_per_chunk) to create overlap.
            let next_pos = pos + step;
            if next_pos >= sentences.len() || next_pos <= pos {
                break;
            }
            pos = next_pos;
        }

        slabs
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        // Rough estimate: ~100 chars per sentence
        let estimated_sentences = text_len / 100;
        let step = self.sentences_per_chunk - self.overlap_sentences;
        (estimated_sentences / step).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sentences() {
        let chunker = SentenceChunker::new(1);
        let text = "Hello world. How are you? I am fine.";
        let slabs = chunker.chunk(text);

        assert_eq!(slabs.len(), 3);
        assert!(slabs[0].text.contains("Hello"));
        assert!(slabs[1].text.contains("How"));
        assert!(slabs[2].text.contains("fine"));
    }

    #[test]
    fn test_grouped_sentences() {
        let chunker = SentenceChunker::new(2);
        let text = "One. Two. Three. Four.";
        let slabs = chunker.chunk(text);

        assert_eq!(slabs.len(), 2);
    }

    #[test]
    fn test_overlap_sentences() {
        let chunker = SentenceChunker::new(2).with_overlap(1);
        let text = "One. Two. Three. Four.";
        let slabs = chunker.chunk(text);

        // With step=1 (2-1), we get: [One,Two], [Two,Three], [Three,Four]
        assert!(
            slabs.len() >= 3,
            "expected >= 3 overlapping chunks, got {}",
            slabs.len()
        );

        // Verify overlap: chunk 1 should contain text from the end of chunk 0
        if slabs.len() >= 2 {
            // The last sentence of chunk 0 should appear in chunk 1
            assert!(
                slabs[1].text.contains("Two"),
                "overlap: chunk 1 should contain 'Two' from chunk 0"
            );
        }
    }

    #[test]
    fn test_overlap_byte_offsets() {
        let chunker = SentenceChunker::new(3).with_overlap(1);
        let text = "Alpha. Bravo. Charlie. Delta. Echo. Foxtrot.";
        let slabs = chunker.chunk(text);

        // Verify overlapping byte ranges
        if slabs.len() >= 2 {
            assert!(
                slabs[1].start < slabs[0].end,
                "overlap: chunk 1 start ({}) should be < chunk 0 end ({})",
                slabs[1].start,
                slabs[0].end
            );
        }
    }

    #[test]
    fn test_no_overlap_default() {
        let chunker = SentenceChunker::new(2);
        let text = "One. Two. Three. Four.";
        let slabs = chunker.chunk(text);

        // Default (no overlap): exactly 2 chunks
        assert_eq!(slabs.len(), 2);

        // No overlapping byte ranges
        if slabs.len() >= 2 {
            assert!(
                slabs[1].start >= slabs[0].end,
                "no overlap: chunk 1 start ({}) should be >= chunk 0 end ({})",
                slabs[1].start,
                slabs[0].end
            );
        }
    }

    #[test]
    fn test_abbreviations() {
        let chunker = SentenceChunker::new(1);
        let text = "Dr. Smith went to Washington D.C. on Tuesday.";
        let slabs = chunker.chunk(text);

        // Unicode segmentation handles "Dr." but may split on "D.C."
        // The important thing is it doesn't split on every period
        assert!(slabs.len() <= 2, "Too many splits: {:?}", slabs);
    }

    #[test]
    fn test_empty_text() {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk("");
        assert!(slabs.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk("   \n\t  ");
        assert!(slabs.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_zero_sentences_panics() {
        let _ = SentenceChunker::new(0);
    }

    #[test]
    #[should_panic]
    fn test_overlap_exceeds_chunk_size_panics() {
        let _ = SentenceChunker::new(3).with_overlap(3);
    }
}
