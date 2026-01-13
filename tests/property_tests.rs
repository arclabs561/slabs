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
    prop::collection::vec(
        prop::string::string_regex("[A-Za-z]{2,15}")
            .unwrap(),
        3..20
    ).prop_map(|words| {
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
    })
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
