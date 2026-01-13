//! Chunking Strategies Comparison
//!
//! Demonstrates the three core chunking strategies and their trade-offs.
//!
//! ```bash
//! cargo run --example chunking_strategies
//! ```

use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker};

fn main() {
    println!("Text Chunking Strategies");
    println!("========================\n");

    // Sample document with varied structure
    let document = r"Machine learning models learn patterns from data. They generalize these patterns to make predictions on new, unseen examples. This is fundamentally different from traditional programming, where humans write explicit rules.

The training process involves three key steps:

1. Forward pass: Input flows through the network, producing predictions.
2. Loss computation: Predictions are compared against ground truth.
3. Backpropagation: Gradients flow backward, updating weights.

Deep learning extends this with multiple hidden layers. Each layer learns increasingly abstract representations. Early layers detect edges; later layers recognize objects.

Dr. Geoffrey Hinton pioneered backpropagation in the 1980s. His work at the University of Toronto, along with collaborators like Yann LeCun and Yoshua Bengio, laid the foundation for modern AI. In 2024, they were recognized with the Nobel Prize.";

    println!("Document length: {} characters\n", document.len());

    // Strategy 1: Fixed-size chunks
    println!("1. Fixed-Size Chunking");
    println!("   -------------------");
    println!("   Splits every N chars with M overlap. Simple but ignores boundaries.\n");

    let fixed = FixedChunker::new(200, 20);
    let chunks = fixed.chunk(document);

    println!("   Chunks: {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let preview = chunk.text.chars().take(60).collect::<String>();
        println!(
            "   [{}] bytes {}-{}: \"{}...\"",
            i, chunk.start, chunk.end, preview
        );
    }

    // Show a boundary problem
    if let Some(chunk) = chunks.get(1) {
        if chunk.text.chars().next().is_some_and(char::is_alphabetic) {
            println!("\n   Note: Chunk 1 starts mid-word/sentence - this is the trade-off.");
        }
    }

    // Strategy 2: Sentence-based chunks
    println!("\n2. Sentence-Based Chunking");
    println!("   -----------------------");
    println!("   Groups N sentences. Respects linguistic boundaries.\n");

    let sentence = SentenceChunker::new(3); // 3 sentences per chunk
    let chunks = sentence.chunk(document);

    println!("   Chunks: {} (3 sentences each)", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let sentences: Vec<_> = chunk.text.split(". ").take(2).collect();
        println!("   [{}] \"{}...\"", i, sentences.join(". "));
    }
    println!("\n   Note: Handles abbreviations like \"Dr.\" correctly (UAX #29 segmentation).");

    // Strategy 3: Recursive chunking
    println!("\n3. Recursive Chunking");
    println!("   ------------------");
    println!("   Tries paragraph -> line -> sentence -> word splits progressively.\n");

    let recursive = RecursiveChunker::new(250, &["\n\n", "\n", ". ", " "]);
    let chunks = recursive.chunk(document);

    println!("   Chunks: {}", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        let preview = chunk.text.chars().take(60).collect::<String>();
        println!(
            "   [{}] ({} chars): \"{}...\"",
            i,
            chunk.text.len(),
            preview
        );
    }
    println!("\n   Note: Paragraphs stay intact when possible (<250 chars).");

    // Summary
    println!("\n--- Summary ---\n");
    println!("| Strategy    | Chunks | Preserves Boundaries | Best For           |");
    println!("|-------------|--------|---------------------|---------------------|");
    println!("| Fixed       | many   | No                  | Logs, code, baseline|");
    println!("| Sentence    | few    | Yes (linguistic)    | Prose, articles     |");
    println!("| Recursive   | medium | Yes (structural)    | Mixed content       |");

    println!("\nFor semantic chunking (topic-aware), use `SemanticChunker` with embeddings.");
    println!("For late chunking (context-preserving), wrap any chunker with `LateChunker`.");
}
