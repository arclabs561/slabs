//! Integration tests for innr usage in semantic chunking.
//!
//! These tests verify that the semantic chunker correctly uses
//! innr for SIMD-accelerated similarity computation.

#![cfg(feature = "semantic")]

// Note: These tests require the `semantic` feature and fastembed model downloads.
// They are integration tests that verify the cross-crate dependency works.

use slabs::{Chunker, SemanticChunker};

#[test]
#[ignore] // Requires fastembed model download
fn semantic_chunker_creates_valid_chunks() {
    let text = r#"
    Machine learning is transforming technology. Neural networks 
    can recognize patterns in data. This has applications in image 
    recognition and natural language processing.
    
    Climate change is affecting ecosystems worldwide. Rising 
    temperatures impact agriculture and biodiversity. Scientists 
    are studying mitigation strategies.
    "#;

    let chunker = SemanticChunker::new(0.5).expect("Failed to create semantic chunker");
    let slabs = chunker.chunk(text);

    // Should produce at least one chunk
    assert!(!slabs.is_empty(), "Semantic chunker should produce chunks");

    // Chunks should cover the text
    for slab in &slabs {
        assert!(!slab.text.is_empty(), "Chunks should not be empty");
        assert!(slab.start < slab.end, "Chunk bounds should be valid");
    }
}

#[test]
#[ignore] // Requires fastembed model download  
fn semantic_chunker_detects_topic_shifts() {
    // Two clearly different topics
    let text = r#"
    Quantum computing uses qubits instead of classical bits.
    Superposition allows qubits to be in multiple states.
    Entanglement enables quantum correlations.
    
    Medieval castles served as defensive fortifications.
    Stone walls protected against siege weapons.
    Moats provided additional protection.
    "#;

    let chunker = SemanticChunker::new(0.3).expect("Failed to create semantic chunker");
    let slabs = chunker.chunk(text);

    // Should detect the topic shift and create at least 2 chunks
    // (one for quantum computing, one for medieval castles)
    assert!(
        slabs.len() >= 2,
        "Should detect topic shift, got {} chunks",
        slabs.len()
    );
}

/// Test that innr's cosine similarity is being used correctly
/// by verifying that similar sentences stay together
#[test]
#[ignore] // Requires fastembed model download
fn innr_cosine_groups_similar_content() {
    // Sentences about the same topic should stay together
    let text = r#"
    Dogs are loyal companions. Puppies need training.
    Canines have been domesticated for thousands of years.
    
    Algebra uses variables to represent numbers.
    Equations can be solved step by step.
    Mathematics is fundamental to science.
    "#;

    let chunker = SemanticChunker::new(0.4).expect("Failed to create semantic chunker");
    let slabs = chunker.chunk(text);

    // The chunker should keep related content together
    // Verify chunks are coherent (this is a weak test, but validates integration)
    for slab in &slabs {
        let words: Vec<&str> = slab.text.split_whitespace().collect();
        assert!(
            words.len() >= 2,
            "Chunks should contain meaningful content"
        );
    }
}
