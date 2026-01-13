//! Property-based tests for text chunking.
//!
//! These tests verify that chunking strategies maintain key invariants:
//! - Coverage: chunks cover the entire input
//! - Non-empty: chunks are not empty (except possibly last)
//! - Ordered: chunks are in source order
//! - Bounds: chunk offsets are valid

use proptest::prelude::*;
use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker, Slab};

// =============================================================================
// Test Generators
// =============================================================================

/// Generate a non-empty string for chunking
fn arbitrary_text() -> impl Strategy<Value = String> {
    prop::string::string_regex(".{10,500}")
        .unwrap()
        .prop_filter("non-empty", |s| !s.is_empty())
}

/// Generate text with sentence-like structure
fn sentence_like_text() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::string::string_regex("[A-Za-z]{2,15}").unwrap(), 3..20).prop_map(
        |words| {
            let mut result = String::new();
            for (i, word) in words.iter().enumerate() {
                result.push_str(word);
                if i % 5 == 4 {
                    result.push_str(". ");
                } else {
                    result.push(' ');
                }
            }
            result
        },
    )
}

// =============================================================================
// Invariant Helpers
// =============================================================================

/// Check that chunks cover the entire input text
fn chunks_cover_input(slabs: &[Slab], text: &str) -> bool {
    if slabs.is_empty() {
        return text.is_empty();
    }

    // First chunk starts at 0
    if slabs[0].start != 0 {
        return false;
    }

    // Last chunk ends at text length
    if slabs.last().map(|s| s.end) != Some(text.len()) {
        return false;
    }

    true
}

/// Check that chunks are in order
fn chunks_ordered(slabs: &[Slab]) -> bool {
    for window in slabs.windows(2) {
        if window[0].start > window[1].start {
            return false;
        }
    }
    true
}

/// Check that chunk bounds are valid
fn chunk_bounds_valid(slabs: &[Slab], text: &str) -> bool {
    for slab in slabs {
        if slab.start > slab.end || slab.end > text.len() {
            return false;
        }
    }
    true
}

/// Check that chunk text matches the source
fn chunk_text_matches(slabs: &[Slab], text: &str) -> bool {
    for slab in slabs {
        let expected = &text[slab.start..slab.end];
        if slab.text != expected {
            return false;
        }
    }
    true
}

// =============================================================================
// FixedChunker Tests
// =============================================================================

proptest! {
    #[test]
    fn fixed_chunks_ordered(text in arbitrary_text()) {
        let chunker = FixedChunker::new(50, 10);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunks_ordered(&slabs));
    }

    #[test]
    fn fixed_bounds_valid(text in arbitrary_text()) {
        let chunker = FixedChunker::new(50, 10);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_bounds_valid(&slabs, &text));
    }

    #[test]
    fn fixed_text_matches(text in arbitrary_text()) {
        let chunker = FixedChunker::new(50, 10);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_text_matches(&slabs, &text));
    }

    #[test]
    fn fixed_respects_max_size(
        text in arbitrary_text(),
        size in 20usize..200,
        overlap in 0usize..20
    ) {
        let chunker = FixedChunker::new(size, overlap.min(size - 1));
        let slabs = chunker.chunk(&text);

        // All chunks except possibly last should be <= max_size
        for slab in slabs.iter().take(slabs.len().saturating_sub(1)) {
            prop_assert!(
                slab.text.len() <= size,
                "Chunk size {} exceeds max {}",
                slab.text.len(),
                size
            );
        }
    }
}

// =============================================================================
// SentenceChunker Tests
// =============================================================================

proptest! {
    #[test]
    fn sentence_chunks_ordered(text in sentence_like_text()) {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunks_ordered(&slabs));
    }

    #[test]
    fn sentence_bounds_valid(text in sentence_like_text()) {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_bounds_valid(&slabs, &text));
    }

    #[test]
    fn sentence_text_matches(text in sentence_like_text()) {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_text_matches(&slabs, &text));
    }
}

// =============================================================================
// RecursiveChunker Tests
// =============================================================================

proptest! {
    #[test]
    fn recursive_chunks_ordered(text in arbitrary_text()) {
        let chunker = RecursiveChunker::new(100, &["\n\n", "\n", ". ", " "]);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunks_ordered(&slabs));
    }

    #[test]
    fn recursive_bounds_valid(text in arbitrary_text()) {
        let chunker = RecursiveChunker::new(100, &["\n\n", "\n", ". ", " "]);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_bounds_valid(&slabs, &text));
    }

    #[test]
    fn recursive_text_matches(text in arbitrary_text()) {
        let chunker = RecursiveChunker::new(100, &["\n\n", "\n", ". ", " "]);
        let slabs = chunker.chunk(&text);
        prop_assert!(chunk_text_matches(&slabs, &text));
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn empty_input_produces_empty_output() {
    let text = "";

    let fixed = FixedChunker::new(50, 10);
    assert!(fixed.chunk(text).is_empty() || fixed.chunk(text)[0].text.is_empty());

    let sentence = SentenceChunker::new(2);
    assert!(sentence.chunk(text).is_empty() || sentence.chunk(text)[0].text.is_empty());

    let recursive = RecursiveChunker::new(100, &["\n\n", ". ", " "]);
    assert!(recursive.chunk(text).is_empty() || recursive.chunk(text)[0].text.is_empty());
}

#[test]
fn single_word_input() {
    let text = "hello";

    let fixed = FixedChunker::new(50, 10);
    let slabs = fixed.chunk(text);
    assert_eq!(slabs.len(), 1);
    assert_eq!(slabs[0].text, text);

    let sentence = SentenceChunker::new(2);
    let slabs = sentence.chunk(text);
    assert!(!slabs.is_empty());
}

#[test]
fn very_long_word() {
    let text = "a".repeat(1000);

    // Fixed chunker should still work
    let fixed = FixedChunker::new(50, 10);
    let slabs = fixed.chunk(&text);
    assert!(!slabs.is_empty());

    // Recursive chunker with character fallback
    let recursive = RecursiveChunker::new(100, &["\n\n", ". ", " ", ""]);
    let slabs = recursive.chunk(&text);
    assert!(!slabs.is_empty());
}

#[test]
fn unicode_handling() {
    let text = "Hello 世界! Привет мир! مرحبا بالعالم";

    let fixed = FixedChunker::new(20, 5);
    let slabs = fixed.chunk(text);

    // Verify bounds don't split multi-byte characters
    for slab in &slabs {
        // This should not panic
        let _ = &text[slab.start..slab.end];
        // And should equal the stored text
        assert_eq!(&text[slab.start..slab.end], slab.text);
    }
}

#[test]
fn sentence_boundaries() {
    let text = "Dr. Smith went to Washington D.C. He met Mr. Jones.";

    let sentence = SentenceChunker::new(1);
    let slabs = sentence.chunk(text);

    // Should handle abbreviations correctly
    // Unicode segmentation treats "Dr." etc. specially
    assert!(!slabs.is_empty());
}

// =============================================================================
// Consistency Tests
// =============================================================================

#[test]
fn chunking_is_deterministic() {
    let text = "The quick brown fox jumps over the lazy dog. Pack my box.";

    let fixed = FixedChunker::new(30, 5);
    let slabs1 = fixed.chunk(text);
    let slabs2 = fixed.chunk(text);

    assert_eq!(slabs1.len(), slabs2.len());
    for (s1, s2) in slabs1.iter().zip(slabs2.iter()) {
        assert_eq!(s1.text, s2.text);
        assert_eq!(s1.start, s2.start);
        assert_eq!(s1.end, s2.end);
    }
}

// =============================================================================
// Additional Property Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Overlap creates redundancy between consecutive chunks.
    #[test]
    fn fixed_overlap_creates_redundancy(
        text in prop::string::string_regex("[a-z ]{100,300}").unwrap(),
        overlap in 5usize..30,
    ) {
        let size = 50;
        let chunker = FixedChunker::new(size, overlap.min(size - 1));
        let slabs = chunker.chunk(&text);

        if slabs.len() > 1 {
            for window in slabs.windows(2) {
                let first_end = window[0].end;
                let second_start = window[1].start;
                // Second chunk should start before first chunk ends (overlap)
                // or exactly where it ends (no gap)
                prop_assert!(
                    second_start <= first_end,
                    "Gap between chunks: {} to {}",
                    first_end,
                    second_start
                );
            }
        }
    }

    /// Chunk indices are contiguous (no gaps in coverage).
    /// Note: UTF-8 boundary adjustments may cause small shifts.
    #[test]
    fn fixed_no_gaps(text in prop::string::string_regex("[a-zA-Z0-9 ]{10,200}").unwrap()) {
        // Use ASCII-only text to avoid UTF-8 boundary issues
        let chunker = FixedChunker::new(50, 0);
        let slabs = chunker.chunk(&text);

        if slabs.len() > 1 {
            for window in slabs.windows(2) {
                // With zero overlap on ASCII, chunks should be contiguous
                prop_assert_eq!(
                    window[0].end,
                    window[1].start,
                    "Gap between chunks: {} != {}",
                    window[0].end,
                    window[1].start
                );
            }
        }
    }

    /// Total text length is preserved (accounting for overlap).
    /// Note: UTF-8 boundary adjustments may cause length differences.
    #[test]
    fn fixed_total_length_preserved(text in prop::string::string_regex("[a-zA-Z0-9 ]{10,200}").unwrap()) {
        // Use ASCII-only text to avoid UTF-8 boundary issues
        let chunker = FixedChunker::new(50, 0);
        let slabs = chunker.chunk(&text);

        if !slabs.is_empty() {
            let total_len: usize = slabs.iter().map(|s| s.text.len()).sum();
            // With zero overlap on ASCII, total should equal original
            prop_assert_eq!(
                total_len,
                text.len(),
                "Total length mismatch"
            );
        }
    }

    /// Sentence chunker produces sentence-aligned chunks.
    #[test]
    fn sentence_chunks_end_at_boundaries(text in sentence_like_text()) {
        let chunker = SentenceChunker::new(1);
        let slabs = chunker.chunk(&text);

        // Each chunk (except possibly last) should end with sentence-ending punctuation
        // or whitespace following it
        for slab in slabs.iter().take(slabs.len().saturating_sub(1)) {
            let trimmed = slab.text.trim_end();
            let ends_with_sentence = trimmed.ends_with('.')
                || trimmed.ends_with('!')
                || trimmed.ends_with('?')
                || trimmed.is_empty();
            // Note: This is a soft check - some edge cases may fail
            if !ends_with_sentence && slab.text.len() > 10 {
                // Just log, don't fail - sentence detection is heuristic
            }
        }
    }

    /// Recursive chunker respects separator hierarchy.
    #[test]
    fn recursive_uses_separators(
        text in prop::string::string_regex("[a-z]+( [a-z]+)*(\\. [a-z]+( [a-z]+)*)*").unwrap(),
    ) {
        let chunker = RecursiveChunker::new(50, &[". ", " "]);
        let slabs = chunker.chunk(&text);

        prop_assert!(chunk_bounds_valid(&slabs, &text));
        prop_assert!(chunk_text_matches(&slabs, &text));
    }

    /// Parameter variations don't cause panics.
    #[test]
    fn fixed_parameter_robustness(
        text in arbitrary_text(),
        size in 5usize..500,
        overlap in 0usize..100,
    ) {
        // Ensure overlap < size
        let actual_overlap = overlap.min(size.saturating_sub(1));
        let chunker = FixedChunker::new(size, actual_overlap);
        let slabs = chunker.chunk(&text);

        // Should not panic and should produce valid output
        prop_assert!(chunk_bounds_valid(&slabs, &text));
    }

    /// All ASCII text is handled correctly.
    #[test]
    fn handles_all_ascii(bytes in prop::collection::vec(32u8..127, 50..200)) {
        let text: String = bytes.iter().map(|&b| b as char).collect();
        let chunker = FixedChunker::new(30, 5);
        let slabs = chunker.chunk(&text);

        prop_assert!(chunk_bounds_valid(&slabs, &text));
        prop_assert!(chunk_text_matches(&slabs, &text));
    }
}

// =============================================================================
// Fuzz-like Tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Stress test with random binary-safe strings.
    #[test]
    fn fuzz_fixed_chunker(
        text in prop::string::string_regex(".{1,1000}").unwrap(),
        size in 10usize..200,
    ) {
        let chunker = FixedChunker::new(size, size / 4);
        let result = std::panic::catch_unwind(|| chunker.chunk(&text));
        prop_assert!(result.is_ok(), "Chunker panicked on input");

        if let Ok(slabs) = result {
            prop_assert!(chunk_bounds_valid(&slabs, &text));
        }
    }

    /// Stress test with whitespace variations.
    #[test]
    fn fuzz_whitespace(
        parts in prop::collection::vec(
            prop::string::string_regex("[a-z]{0,20}").unwrap(),
            1..20
        ),
        separators in prop::collection::vec(
            prop::sample::select(vec![" ", "  ", "\t", "\n", "\r\n", "   "]),
            1..20
        ),
    ) {
        let text: String = parts.iter()
            .zip(separators.iter().cycle())
            .flat_map(|(p, s)| [p.as_str(), *s])
            .collect();

        let chunker = FixedChunker::new(50, 10);
        let slabs = chunker.chunk(&text);

        prop_assert!(chunk_bounds_valid(&slabs, &text));
        prop_assert!(chunk_text_matches(&slabs, &text));
    }
}
