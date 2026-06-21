//! Pool full-document token embeddings over externally chosen spans.
//!
//! In a real pipeline:
//! 1. Choose boundaries with `text-splitter`, parser output, regex, or an extraction pipeline.
//! 2. Store those boundaries as `Slab`s.
//! 3. Embed the full document with a long-context model and keep token offsets.
//! 4. Pool token embeddings inside each slab's byte span.
//!
//! Pool semantics preserve document-wide context for spans that contain
//! pronouns, anaphora, or acronym references (Günther et al. 2024).
//!
//! Run with: `cargo run --example late_chunking`

use slabs::{LateChunkingPooler, Slab};
use text_splitter::TextSplitter;

fn main() {
    let document =
        "Einstein developed relativity. He became famous. The theory transformed physics.";

    // Step 1: boundaries come from another tool. `slabs` records the spans;
    // it does not decide where text should be split.
    let splitter = TextSplitter::new(32);
    let spans: Vec<Slab> = splitter
        .chunk_indices(document)
        .enumerate()
        .map(|(index, (start, chunk))| {
            Slab::from_byte_range(document, start..start + chunk.len(), index).unwrap()
        })
        .collect();

    // Step 2: in a real pipeline, run the document through a long-context embedder
    // and capture token-level output plus tokenizer offsets. Here we use small,
    // interpretable vectors. Dimensions are:
    // [Einstein/relativity context, pronoun/anaphora, theory reference, physics].
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
    let token_embeddings = vec![
        vec![1.00, 0.00, 0.00, 0.10], // Einstein
        vec![0.75, 0.00, 0.00, 0.05], // developed
        vec![0.95, 0.00, 0.20, 0.25], // relativity
        vec![0.85, 0.90, 0.00, 0.05], // He, contextualized by full-document encoding
        vec![0.55, 0.20, 0.00, 0.05], // became
        vec![0.50, 0.10, 0.00, 0.05], // famous
        vec![0.80, 0.00, 0.95, 0.15], // The, contextualized as the theory
        vec![0.85, 0.00, 1.00, 0.20], // theory
        vec![0.30, 0.00, 0.55, 0.85], // transformed
        vec![0.25, 0.00, 0.40, 1.00], // physics
    ];

    // Step 3: pool. Use exact offsets when the tokenizer provides them.
    // `pool` is available as a fallback when only document length is known.
    let pooler = LateChunkingPooler::new(dim);
    let span_embeddings = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &spans);

    for (span, emb) in spans.iter().zip(&span_embeddings) {
        println!("span {} [{:?}]: {:?}", span.index, span.span(), span.text);
        println!("  pooled [einstein, pronoun, theory, physics]: {emb:.3?}\n");
    }
}
