//! Basic Text Chunking
//!
//! The minimal example: chunk text for embedding.
//!
//! ```bash
//! cargo run --example 01_basic_chunking
//! ```

use slabs::{Chunker, SentenceChunker};

fn main() {
    let document = "Machine learning models learn patterns from data. \
        They generalize these patterns to make predictions. \
        This is fundamentally different from traditional programming. \
        Deep learning extends this with multiple hidden layers. \
        Each layer learns increasingly abstract representations.";

    // Chunk by sentences (2 per chunk)
    let chunker = SentenceChunker::new(2);
    let chunks = chunker.chunk(document);

    println!("Document: {} chars", document.len());
    println!("Chunks: {}\n", chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        println!("[{}] {} chars: \"{}\"", i, chunk.text.len(), chunk.text);
    }

    // Each chunk is now small enough to embed (~100-200 tokens)
    // and large enough to preserve sentence context.
}
