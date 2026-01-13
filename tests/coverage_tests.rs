#![allow(dead_code, clippy::needless_range_loop, clippy::manual_assert)]
//! Coverage and overlap tests for text chunking.
//!
//! These tests verify that chunks properly cover input text and
//! handle overlaps correctly.

use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker, Slab};

// =============================================================================
// Coverage: Chunks should cover the entire input
// =============================================================================

/// Check if reconstructing text from chunks equals original.
fn reconstructs_to_original(slabs: &[Slab], text: &str) -> bool {
    if slabs.is_empty() {
        return text.is_empty();
    }

    // Build coverage map
    let mut covered = vec![false; text.len()];
    for slab in slabs {
        for i in slab.start..slab.end {
            covered[i] = true;
        }
    }

    // All bytes should be covered
    covered.iter().all(|&c| c)
}

#[test]
fn fixed_chunker_full_coverage() {
    let texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        &"A".repeat(1000),
        "Short",
        " Leading and trailing spaces ",
        "Multiple\n\nParagraphs\n\nHere",
    ];

    for text in &texts {
        let chunker = FixedChunker::new(50, 10);
        let slabs = chunker.chunk(text);

        assert!(
            reconstructs_to_original(&slabs, text),
            "Fixed chunker failed coverage for: {:?}",
            &text[..text.len().min(50)]
        );
    }
}

#[test]
fn recursive_chunker_valid_chunks() {
    let texts = [
        "Hello, world!",
        "First paragraph.\n\nSecond paragraph.\n\nThird.",
        "Sentence one. Sentence two. Sentence three.",
        "Word by word by word by word.",
        &"NoSeparatorsAtAll".repeat(10),
    ];

    for text in &texts {
        let chunker = RecursiveChunker::new(100, &["\n\n", "\n", ". ", " "]);
        let slabs = chunker.chunk(text);

        // Verify all chunks have valid bounds and text matches
        for slab in &slabs {
            assert!(slab.start <= slab.end, "Invalid bounds");
            assert!(slab.end <= text.len(), "End exceeds text length");
            assert_eq!(&text[slab.start..slab.end], slab.text, "Text mismatch");
        }
    }
}

#[test]
fn sentence_chunker_valid_chunks() {
    let texts = [
        "Hello. World.",
        "Dr. Smith went home. He was tired.",
        "First! Second? Third.",
        "No sentence ending here",
    ];

    for text in &texts {
        let chunker = SentenceChunker::new(2);
        let slabs = chunker.chunk(text);

        // Verify all chunks have valid bounds and text matches
        for slab in &slabs {
            assert!(slab.start <= slab.end, "Invalid bounds");
            assert!(slab.end <= text.len(), "End exceeds text length");
            assert_eq!(&text[slab.start..slab.end], slab.text, "Text mismatch");
        }
    }
}

// =============================================================================
// Overlap tests for FixedChunker
// =============================================================================

#[test]
fn fixed_chunker_overlap_property() {
    let text = "The quick brown fox jumps over the lazy dog. Pack my box.";

    // Test with different overlaps
    for overlap in [0, 5, 10, 20] {
        let chunker = FixedChunker::new(30, overlap);
        let slabs = chunker.chunk(text);

        if slabs.len() > 1 {
            for window in slabs.windows(2) {
                let first = &window[0];
                let second = &window[1];

                // Either chunks don't overlap, or they overlap by at most `overlap` chars
                if second.start < first.end {
                    let actual_overlap = first.end - second.start;
                    assert!(
                        actual_overlap <= overlap,
                        "Overlap {} exceeds requested {} for chunks [{},{}] and [{},{}]",
                        actual_overlap,
                        overlap,
                        first.start,
                        first.end,
                        second.start,
                        second.end
                    );
                }
            }
        }
    }
}

#[test]
fn fixed_chunker_no_overlap_means_contiguous() {
    let text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    let chunker = FixedChunker::new(5, 0);
    let slabs = chunker.chunk(text);

    // With zero overlap, chunks should be contiguous or adjacent
    for window in slabs.windows(2) {
        let gap = window[1].start.saturating_sub(window[0].end);
        assert!(gap == 0, "Gap of {} between chunks with zero overlap", gap);
    }
}

// =============================================================================
// Size bounds
// =============================================================================

#[test]
fn fixed_chunker_respects_size() {
    let text = "A".repeat(500);

    for size in [20, 50, 100, 200] {
        let chunker = FixedChunker::new(size, 5);
        let slabs = chunker.chunk(&text);

        for (i, slab) in slabs.iter().enumerate() {
            // Last chunk may be smaller but all others should be <= size
            if i < slabs.len() - 1 {
                assert!(
                    slab.text.len() <= size,
                    "Chunk {} has size {} > max {}",
                    i,
                    slab.text.len(),
                    size
                );
            }
        }
    }
}

#[test]
fn recursive_chunker_respects_size() {
    let text = "First paragraph with lots of words. More words here.\n\n\
                Second paragraph also has words. Even more words.\n\n\
                Third paragraph continues. And more sentences.";

    for size in [50, 100, 200] {
        let chunker = RecursiveChunker::new(size, &["\n\n", ". ", " "]);
        let slabs = chunker.chunk(text);

        for (i, slab) in slabs.iter().enumerate() {
            // With proper separators, chunks should mostly respect size
            // (may exceed if no separator found)
            if slab.text.len() > size * 2 {
                panic!(
                    "Chunk {} size {} greatly exceeds target {} for text starting: {:?}",
                    i,
                    slab.text.len(),
                    size,
                    &slab.text[..slab.text.len().min(30)]
                );
            }
        }
    }
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn chunker_handles_only_whitespace() {
    let text = "   \n\n\t\t  ";

    let fixed = FixedChunker::new(50, 10);
    let slabs = fixed.chunk(text);
    assert!(chunk_bounds_valid(&slabs, text));

    let recursive = RecursiveChunker::new(100, &["\n\n", " "]);
    let slabs = recursive.chunk(text);
    assert!(chunk_bounds_valid(&slabs, text));
}

#[test]
fn chunker_handles_newlines() {
    let text = "Line 1\nLine 2\nLine 3";

    let recursive = RecursiveChunker::new(50, &["\n"]);
    let slabs = recursive.chunk(text);

    assert!(!slabs.is_empty());
    assert!(chunk_bounds_valid(&slabs, text));
}

#[test]
fn chunker_handles_very_small_max_size() {
    let text = "Hello World";

    // Even with tiny max_size, should still work
    let chunker = FixedChunker::new(3, 1);
    let slabs = chunker.chunk(text);

    assert!(!slabs.is_empty());
    assert!(chunk_bounds_valid(&slabs, text));
}

#[test]
fn chunker_handles_size_equals_text_length() {
    let text = "Exactly fifty characters in this string, not more.";

    let chunker = FixedChunker::new(text.len(), 0);
    let slabs = chunker.chunk(text);

    // Should produce exactly one chunk
    assert_eq!(slabs.len(), 1);
    assert_eq!(slabs[0].text, text);
}

// =============================================================================
// Helpers
// =============================================================================

fn chunk_bounds_valid(slabs: &[Slab], text: &str) -> bool {
    for slab in slabs {
        if slab.start > slab.end || slab.end > text.len() {
            return false;
        }
        // Also verify text matches
        if slab.text != &text[slab.start..slab.end] {
            return false;
        }
    }
    true
}
