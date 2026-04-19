//! Chunk a Rust source file at AST boundaries.
//!
//! Run with: `cargo run --example code_chunking --features code`

use code_chunker::{Chunker, CodeChunker, CodeLanguage};

fn main() {
    let source = r#"
use std::collections::HashMap;

pub struct Cache<K, V> {
    inner: HashMap<K, V>,
    capacity: usize,
}

impl<K: std::hash::Hash + Eq, V> Cache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.inner.len() >= self.capacity {
            return None;
        }
        self.inner.insert(key, value)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }
}

fn main() {
    let mut cache = Cache::new(10);
    cache.insert("hello", 1);
}
"#;

    // 800-byte chunks, no overlap. Functions/impls/structs are kept atomic when they fit.
    let chunker = CodeChunker::new(CodeLanguage::Rust, 800, 0);
    let slabs = chunker.chunk(source);

    println!(
        "{} chunk(s) produced from {} bytes:\n",
        slabs.len(),
        source.len()
    );
    for slab in &slabs {
        println!(
            "--- chunk {} (bytes {}..{}) ---",
            slab.index, slab.start, slab.end
        );
        println!("{}\n", slab.text);
    }
}
