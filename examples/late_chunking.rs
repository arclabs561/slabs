//! Pool fake token embeddings into per-chunk embeddings.
//!
//! In a real pipeline:
//! 1. Use any boundary source (`text-splitter`, parser output, regex, manual) to produce `Vec<Slab>`.
//! 2. Embed the FULL document with a long-context model (Jina v2/v3, nomic-embed-text)
//!    to get token-level embeddings of shape `[n_tokens, dim]`.
//! 3. Pool token embeddings inside each chunk's byte span, preferably with
//!    exact tokenizer offsets.
//!
//! Pool semantics preserve document-wide context: pronouns, anaphora, acronym
//! definitions are no longer lost at chunk boundaries (Günther et al. 2024).
//!
//! Run with: `cargo run --example late_chunking`

use slabs::{LateChunkingPooler, Slab};

fn main() {
    let document =
        "Einstein developed relativity. He became famous. The theory transformed physics.";

    // Step 1: chunk boundaries from any source. Here, hand-rolled at sentence ends.
    // In practice, use text-splitter for prose or code, parser output, or
    // spans from an extraction pipeline.
    let chunks = vec![
        Slab::from_byte_range(document, 0..30, 0).unwrap(),
        Slab::from_byte_range(document, 30..48, 1).unwrap(),
        Slab::from_byte_range(document, 48..80, 2).unwrap(),
    ];
    assert_eq!(chunks.last().map(|chunk| chunk.end), Some(document.len()));

    // Step 2: in a real pipeline, run the document through a long-context embedder
    // and capture token-level output plus tokenizer offsets. Here we fake both.
    let dim = 4;
    let token_offsets = vec![
        (0, 8),   // Einstein
        (9, 18),  // developed
        (19, 30), // relativity.
        (31, 33), // He
        (34, 40), // became
        (41, 48), // famous.
        (49, 52), // The
        (53, 59), // theory
        (60, 71), // transformed
        (72, 80), // physics.
    ];
    let n_tokens = token_offsets.len();
    let token_embeddings: Vec<Vec<f32>> = (0..n_tokens)
        .map(|i| {
            let t = i as f32 / n_tokens as f32;
            vec![t, 1.0 - t, (t * 2.0).sin(), (t * 3.0).cos()]
        })
        .collect();

    // Step 3: pool. Use exact offsets when the tokenizer provides them.
    // `pool` is available as a fallback when only document length is known.
    let pooler = LateChunkingPooler::new(dim);
    let chunk_embeddings = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

    for (chunk, emb) in chunks.iter().zip(&chunk_embeddings) {
        println!("chunk {}: {:?}", chunk.index, chunk.text);
        println!("  embedding: {:?}\n", emb);
    }
}
