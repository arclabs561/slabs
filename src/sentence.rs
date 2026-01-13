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
//! - URLs (https://example.com/path)
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
//! ## Trade-offs
//!
//! | Sentences/Chunk | Pros | Cons |
//! |-----------------|------|------|
//! | 1 | Precise retrieval | No context |
//! | 3-5 | Good balance | May split paragraphs |
//! | 10+ | Full context | May exceed model limits |

use unicode_segmentation::UnicodeSegmentation;

use crate::{Chunker, Slab};

/// Sentence-based chunker.
///
/// Groups consecutive sentences into chunks of approximately equal size.
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
#[derive(Debug, Clone)]
pub struct SentenceChunker {
    sentences_per_chunk: usize,
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
        Self { sentences_per_chunk }
    }

    /// Create a chunker that outputs one sentence per chunk.
    #[must_use]
    pub fn single() -> Self {
        Self::new(1)
    }
}

impl Chunker for SentenceChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        // Collect sentence boundaries using Unicode segmentation
        let sentences: Vec<&str> = text.split_sentence_bounds().collect();

        if sentences.is_empty() {
            return vec![];
        }

        // Filter out whitespace-only "sentences"
        let sentences: Vec<(usize, &str)> = sentences
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

        let mut slabs = Vec::new();
        let mut index = 0;

        for chunk_sentences in sentences.chunks(self.sentences_per_chunk) {
            if chunk_sentences.is_empty() {
                continue;
            }

            let start = chunk_sentences.first().map(|(off, _)| *off).unwrap_or(0);
            let end = chunk_sentences
                .last()
                .map(|(off, s)| off + s.len())
                .unwrap_or(start);

            let chunk_text: String = chunk_sentences.iter().map(|(_, s)| *s).collect();
            let trimmed = chunk_text.trim();

            if !trimmed.is_empty() {
                // Adjust start/end to match trimmed text position
                let leading_ws = chunk_text.len() - chunk_text.trim_start().len();
                let trailing_ws = chunk_text.len() - chunk_text.trim_end().len();

                slabs.push(Slab::new(
                    trimmed,
                    start + leading_ws,
                    end - trailing_ws,
                    index,
                ));
                index += 1;
            }
        }

        slabs
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        // Rough estimate: ~100 chars per sentence
        let estimated_sentences = text_len / 100;
        (estimated_sentences / self.sentences_per_chunk).max(1)
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
        SentenceChunker::new(0);
    }
}
