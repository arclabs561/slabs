//! Pool fake token embeddings into per-chunk embeddings.
//!
//! In a real pipeline:
//! 1. Use any chunker (`CodeChunker`, `text-splitter`, regex, manual) to produce `Vec<Slab>`.
//! 2. Embed the FULL document with a long-context model (Jina v2/v3, nomic-embed-text)
//!    to get token-level embeddings of shape `[n_tokens, dim]`.
//! 3. Pool token embeddings inside each chunk's byte span.
//!
//! Pool semantics preserve document-wide context — pronouns, anaphora, acronym
//! definitions are no longer lost at chunk boundaries (Günther et al. 2024).
//!
//! Run with: `cargo run --example late_chunking`

use slabs::{LateChunkingPooler, Slab};

fn main() {
    let document =
        "Einstein developed relativity. He became famous. The theory transformed physics.";

    // Step 1: chunk boundaries from any source. Here, hand-rolled at sentence ends.
    // In practice, use text-splitter for prose, CodeChunker for code, etc.
    let chunks = vec![
        Slab::new("Einstein developed relativity.", 0, 30, 0),
        Slab::new(" He became famous.", 30, 48, 1),
        Slab::new(" The theory transformed physics.", 48, 80, 2),
    ];

    // Step 2: in a real pipeline, run the document through a long-context embedder
    // and capture the token-level output. Here we fake it: 16 tokens, 4 dims.
    let dim = 4;
    let n_tokens = 16;
    let token_embeddings: Vec<Vec<f32>> = (0..n_tokens)
        .map(|i| {
            let t = i as f32 / n_tokens as f32;
            vec![t, 1.0 - t, (t * 2.0).sin(), (t * 3.0).cos()]
        })
        .collect();

    // Step 3: pool. Linear approximation maps byte spans to token indices.
    // For exact mapping, use `pool_with_offsets` and pass tokenizer offsets.
    let pooler = LateChunkingPooler::new(dim);
    let chunk_embeddings = pooler.pool(&token_embeddings, &chunks, document.len());

    for (chunk, emb) in chunks.iter().zip(&chunk_embeddings) {
        println!("chunk {}: {:?}", chunk.index, chunk.text);
        println!("  embedding: {:?}\n", emb);
    }
}
