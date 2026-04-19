//! Integration tests for `ChunkSizer` plumbing and import-context injection.

#![cfg(feature = "code")]

use code_chunker::{ChunkSizer, Chunker, CodeChunker, CodeLanguage};

const RUST_SAMPLE: &str = r#"use std::collections::HashMap;
use std::sync::Arc;

pub struct Cache {
    inner: HashMap<String, Arc<str>>,
}

impl Cache {
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    pub fn get(&self, key: &str) -> Option<Arc<str>> {
        self.inner.get(key).cloned()
    }
}
"#;

/// A sizer that counts characters (Unicode scalar values) instead of bytes.
struct CharSizer;
impl ChunkSizer for CharSizer {
    fn size(&self, text: &str) -> usize {
        text.chars().count()
    }
}

/// A sizer that returns 1000× the byte length — forces every node to be
/// "oversize," so the chunker recursively splits everything down to leaves.
struct InflateSizer;
impl ChunkSizer for InflateSizer {
    fn size(&self, text: &str) -> usize {
        text.len() * 1000
    }
}

#[test]
fn default_sizer_is_bytes() {
    // 800-byte budget: the sample (~280 bytes) fits in a single chunk.
    let chunker = CodeChunker::new(CodeLanguage::Rust, 800, 0);
    let slabs = chunker.chunk(RUST_SAMPLE);
    assert_eq!(slabs.len(), 1, "small file should be one chunk");
}

#[test]
fn inflate_sizer_forces_more_chunks() {
    // Default byte sizer at budget 800: 1 chunk.
    let baseline = CodeChunker::new(CodeLanguage::Rust, 800, 0);
    let baseline_slabs = baseline.chunk(RUST_SAMPLE);

    // InflateSizer reports 1000× — every block becomes "oversize" so the
    // recursive splitter kicks in and produces more chunks.
    let inflated = CodeChunker::new(CodeLanguage::Rust, 800, 0).with_sizer(InflateSizer);
    let inflated_slabs = inflated.chunk(RUST_SAMPLE);

    assert!(
        inflated_slabs.len() > baseline_slabs.len(),
        "inflated sizer should produce more chunks; baseline={} inflated={}",
        baseline_slabs.len(),
        inflated_slabs.len()
    );
}

#[test]
fn char_sizer_works_for_unicode() {
    let utf8_sample = "fn é() { let 日本語 = 1; }";
    // Byte-sized: ~30 bytes. Char-sized: ~24 chars.
    // Both fit in budget 100, so this just confirms the plumbing compiles
    // and runs without panic.
    let chunker = CodeChunker::new(CodeLanguage::Rust, 100, 0).with_sizer(CharSizer);
    let _slabs = chunker.chunk(utf8_sample);
}

#[test]
fn with_imports_disabled_by_default() {
    let chunker = CodeChunker::new(CodeLanguage::Rust, 200, 0);
    let slabs = chunker.chunk(RUST_SAMPLE);
    // The impl-only chunk (no use statements) should NOT contain "use std".
    let impl_chunk = slabs
        .iter()
        .find(|s| s.text.contains("impl Cache"))
        .expect("expected an impl Cache chunk");
    assert!(
        !impl_chunk.text.contains("use std::collections::HashMap"),
        "import injection should be off by default"
    );
}

#[test]
fn with_imports_prepends_to_method_chunks() {
    // Tight budget so the `impl Cache { ... }` block lands in a chunk
    // separate from the file-head `use` declarations.
    let chunker = CodeChunker::new(CodeLanguage::Rust, 200, 0).with_imports(true);
    let slabs = chunker.chunk(RUST_SAMPLE);

    let impl_chunk = slabs
        .iter()
        .find(|s| s.text.contains("impl Cache"))
        .expect("expected an impl Cache chunk");

    assert!(
        impl_chunk.text.contains("use std::collections::HashMap"),
        "impl chunk should have HashMap import prepended; got:\n{}",
        impl_chunk.text
    );
    assert!(
        impl_chunk.text.contains("use std::sync::Arc"),
        "impl chunk should have Arc import prepended; got:\n{}",
        impl_chunk.text
    );
}

#[test]
fn with_imports_skips_chunks_already_covering_imports() {
    // A chunk that starts at byte 0 already includes the use declarations,
    // so prepending would duplicate them.
    let chunker = CodeChunker::new(CodeLanguage::Rust, 10_000, 0).with_imports(true);
    let slabs = chunker.chunk(RUST_SAMPLE);

    // Whole file fits in one chunk; that chunk starts at byte 0 (within
    // import range) so should not be re-prepended.
    assert_eq!(slabs.len(), 1);
    let single = &slabs[0];
    let occurrences = single.text.matches("use std::collections::HashMap").count();
    assert_eq!(
        occurrences, 1,
        "imports should not be duplicated when chunk already contains them"
    );
}
