//! Recursive character splitting (LangChain-style).
//!
//! Tries progressively finer separators until chunks fit within the size limit.
//!
//! ## The Algorithm
//!
//! Given separators `["\n\n", "\n", ". ", " "]` and max_size `100`:
//!
//! ```text
//! 1. Split on "\n\n" (paragraphs)
//! 2. For each chunk > 100 bytes:
//!    - Split that chunk on "\n" (lines)
//! 3. For each chunk still > 100 bytes:
//!    - Split that chunk on ". " (sentences)
//! 4. For each chunk still > 100 bytes:
//!    - Split that chunk on " " (words)
//! 5. If still > 100 bytes:
//!    - Force split at 100 bytes (rare)
//! ```
//!
//! ## Why Recursive?
//!
//! Different content types need different separators:
//!
//! - **Markdown**: Headings (`#`), paragraphs (`\n\n`), lists (`\n-`)
//! - **Code**: Functions, classes, blank lines
//! - **Prose**: Paragraphs, sentences, words
//!
//! The recursive approach preserves structure at the highest level possible.
//! A paragraph boundary is better than a sentence boundary, which is better
//! than a word boundary.
//!
//! ## Default Separators
//!
//! For general text, this hierarchy works well:
//!
//! ```text
//! ["\n\n", "\n", ". ", " "]
//! ```
//!
//! For Markdown:
//!
//! ```text
//! ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
//! ```
//!
//! For code:
//!
//! ```text
//! ["\nfn ", "\nimpl ", "\n\n", "\n", " "]
//! ```

use crate::{Chunker, Slab};

/// Recursive character splitter.
///
/// Splits text using a hierarchy of separators, trying the coarsest first.
///
/// ## Example
///
/// ```rust
/// use slabs::{Chunker, RecursiveChunker};
///
/// let chunker = RecursiveChunker::new(50, &["\n\n", "\n", ". ", " "]);
/// let text = "Paragraph one.\n\nParagraph two is longer and might need splitting.";
/// let slabs = chunker.chunk(text);
/// ```
#[derive(Debug, Clone)]
pub struct RecursiveChunker {
    max_size: usize,
    overlap: usize,
    separators: Vec<String>,
}

impl RecursiveChunker {
    /// Create a new recursive chunker.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum chunk size in bytes
    /// * `separators` - Hierarchy of separators, coarsest first
    ///
    /// # Panics
    ///
    /// Panics if `max_size == 0` or `separators` is empty.
    #[must_use]
    pub fn new(max_size: usize, separators: &[&str]) -> Self {
        assert!(max_size > 0, "max_size must be > 0");
        assert!(!separators.is_empty(), "separators must not be empty");

        Self {
            max_size,
            overlap: 0,
            separators: separators.iter().map(|&s| s.to_string()).collect(),
        }
    }

    /// Set overlap size.
    #[must_use]
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Create a chunker with default separators for prose.
    #[must_use]
    pub fn prose(max_size: usize) -> Self {
        Self::new(max_size, &["\n\n", "\n", ". ", " "])
    }

    /// Create a chunker with default separators for Markdown.
    #[must_use]
    pub fn markdown(max_size: usize) -> Self {
        Self::new(max_size, &["\n## ", "\n### ", "\n\n", "\n", ". ", " "])
    }

    /// Recursively split a chunk using the remaining separators.
    fn split_recursive(&self, text: &str, sep_index: usize) -> Vec<String> {
        if text.len() <= self.max_size || sep_index >= self.separators.len() {
            // Base case: fits or no more separators
            if text.len() <= self.max_size {
                return vec![text.to_string()];
            }
            // Force split as last resort
            return self.force_split(text);
        }

        let sep = &self.separators[sep_index];
        let parts: Vec<&str> = text.split(sep).collect();

        if parts.len() == 1 {
            // Separator not found, try next one
            return self.split_recursive(text, sep_index + 1);
        }

        let mut result = Vec::new();
        let mut current = String::new();

        for (i, part) in parts.iter().enumerate() {
            let with_sep = if i < parts.len() - 1 {
                format!("{part}{sep}")
            } else {
                (*part).to_string()
            };

            if current.is_empty() {
                current = with_sep;
            } else if current.len() + with_sep.len() <= self.max_size {
                current.push_str(&with_sep);
            } else {
                // Current chunk is full, process it
                if current.len() <= self.max_size {
                    result.push(current);
                } else {
                    // Too big, recurse with finer separator
                    result.extend(self.split_recursive(&current, sep_index + 1));
                }
                current = with_sep;
            }
        }

        // Don't forget the last chunk
        if !current.is_empty() {
            if current.len() <= self.max_size {
                result.push(current);
            } else {
                result.extend(self.split_recursive(&current, sep_index + 1));
            }
        }

        result
    }

    /// Force split at byte boundaries when no separator works.
    fn force_split(&self, text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + self.max_size).min(text.len());
            // Ensure we're at a char boundary
            // Replaces text.floor_char_boundary(end) for MSRV < 1.80 compatibility
            let mut end = end;
            while !text.is_char_boundary(end) {
                end -= 1;
            }

            if end > start {
                result.push(text[start..end].to_string());
            }

            start = end;
        }

        result
    }
}

impl Chunker for RecursiveChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        let chunks = self.split_recursive(text, 0);

        // Convert to Slabs with offsets by walking the concatenation.
        // Invariant: `split_recursive` returns chunks that re-concatenate to `text`.
        let mut slabs = Vec::with_capacity(chunks.len());
        let mut cursor = 0usize;

        for (index, chunk) in chunks.into_iter().enumerate() {
            let start = cursor;
            let end = start + chunk.len();
            cursor = end;

            // Apply overlap by expanding the start backwards, but keep the final
            // chunk size bounded by `max_size`. (Overlap is "up to" `self.overlap`.)
            let mut start_with_overlap = start.saturating_sub(self.overlap);
            if end.saturating_sub(start_with_overlap) > self.max_size {
                start_with_overlap = end.saturating_sub(self.max_size);
            }

            // Ensure UTF-8 char boundary for slicing.
            while start_with_overlap > 0 && !text.is_char_boundary(start_with_overlap) {
                start_with_overlap -= 1;
            }

            slabs.push(Slab::new(
                text[start_with_overlap..end].to_string(),
                start_with_overlap,
                end,
                index,
            ));
        }

        slabs
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        let step = self.max_size.saturating_sub(self.overlap).max(1);
        (text_len / step).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraph_split() {
        let chunker = RecursiveChunker::prose(50);
        let text =
            "Short.\n\nThis is a longer paragraph that might need splitting into smaller pieces.";
        let slabs = chunker.chunk(text);

        assert!(slabs.len() >= 2);
        assert!(slabs[0].text.contains("Short"));
    }

    #[test]
    fn test_respects_max_size() {
        let chunker = RecursiveChunker::prose(20);
        let text = "The quick brown fox jumps over the lazy dog.";
        let slabs = chunker.chunk(text);

        for slab in &slabs {
            assert!(slab.len() <= 20, "Chunk too large: {} bytes", slab.len());
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = RecursiveChunker::prose(100);
        let slabs = chunker.chunk("");
        assert!(slabs.is_empty());
    }

    #[test]
    fn test_small_text_single_chunk() {
        let chunker = RecursiveChunker::prose(100);
        let slabs = chunker.chunk("Small text.");
        assert_eq!(slabs.len(), 1);
    }

    #[test]
    fn test_markdown_headers() {
        let chunker = RecursiveChunker::markdown(100);
        let text = "# Title\n\nIntro.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.";
        let slabs = chunker.chunk(text);

        // Should respect section boundaries
        assert!(slabs.len() >= 1);
    }

    #[test]
    #[should_panic]
    fn test_zero_size_panics() {
        let _ = RecursiveChunker::prose(0);
    }

    #[test]
    #[should_panic]
    fn test_empty_separators_panics() {
        let _ = RecursiveChunker::new(100, &[]);
    }
}
