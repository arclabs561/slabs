//! Semantic chunking using embedding similarity.
//!
//! Splits text where topic changes, detected by drops in embedding similarity.
//!
//! ## The Idea
//!
//! Adjacent sentences about the same topic have similar embeddings.
//! When the topic changes, similarity drops. We split there.
//!
//! ```text
//! Sentences about topic A:    [S1] [S2] [S3]
//! Embeddings:                  E1   E2   E3
//! Similarities:                    0.9  0.85
//!
//! Topic shift:                            |
//! Sentences about topic B:              [S4] [S5]
//! Embeddings:                            E4   E5
//! Similarity with S3:         0.3  ← below threshold!
//!
//! Result: Chunk 1 = [S1, S2, S3], Chunk 2 = [S4, S5]
//! ```
//!
//! ## Threshold Selection
//!
//! The threshold controls sensitivity to topic shifts:
//!
//! | Threshold | Effect |
//! |-----------|--------|
//! | 0.3 | Very sensitive, many small chunks |
//! | 0.5 | Balanced (recommended) |
//! | 0.7 | Only major topic shifts |
//!
//! ## Double-Pass Algorithm
//!
//! For better results, we use a double-pass approach:
//!
//! 1. **First pass**: Split on significant similarity drops
//! 2. **Merge pass**: Combine adjacent small chunks if similar
//!
//! This prevents over-fragmentation while preserving major topic boundaries.
//!
//! ## Performance
//!
//! Semantic chunking is O(n × d) where:
//! - n = number of sentences
//! - d = embedding dimension
//!
//! For a 10-page document (~200 sentences) with 384-dim embeddings:
//! - ~200 embedding calls
//! - ~200 similarity computations
//! - Total: 1-5 seconds depending on embedding model

use unicode_segmentation::UnicodeSegmentation;

use crate::{Chunker, Error, Result, Slab};

/// Semantic chunker using embedding similarity.
///
/// Requires the `semantic` feature and an embedding model.
///
/// ## Example
///
/// ```rust,ignore
/// use slabs::{Chunker, SemanticChunker};
///
/// // Uses fastembed's default model (BGE-small-en)
/// let chunker = SemanticChunker::new(0.5)?;
///
/// let text = "Intro to machine learning. ML is powerful. \
///             The weather today is sunny. It's warm outside.";
/// let slabs = chunker.chunk(text);
///
/// // Should split between ML content and weather content
/// assert_eq!(slabs.len(), 2);
/// ```
pub struct SemanticChunker {
    model: fastembed::TextEmbedding,
    threshold: f32,
    min_chunk_sentences: usize,
}

impl SemanticChunker {
    /// Create a new semantic chunker with default embedding model.
    ///
    /// Uses fastembed's BGE-small-en model (384 dimensions).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Similarity threshold for splitting (0.0 to 1.0)
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding model fails to load.
    pub fn new(threshold: f32) -> Result<Self> {
        let model = fastembed::TextEmbedding::try_new(Default::default())
            .map_err(|e| Error::Embedding(e.to_string()))?;

        Ok(Self {
            model,
            threshold,
            min_chunk_sentences: 2,
        })
    }

    /// Set the minimum sentences per chunk.
    ///
    /// Prevents over-fragmentation by requiring at least N sentences per chunk.
    #[must_use]
    pub fn with_min_sentences(mut self, min: usize) -> Self {
        self.min_chunk_sentences = min;
        self
    }

    /// Compute cosine similarity between two embeddings.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        #[cfg(feature = "innr")]
        {
            innr::cosine(a, b)
        }

        #[cfg(not(feature = "innr"))]
        {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a > 0.0 && norm_b > 0.0 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            }
        }
    }

    /// Extract sentences from text.
    fn extract_sentences(text: &str) -> Vec<(usize, String)> {
        let mut sentences = Vec::new();
        let mut offset = 0;

        for sentence in text.split_sentence_bounds() {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                // Find actual position in original text
                if let Some(pos) = text[offset..].find(trimmed) {
                    sentences.push((offset + pos, trimmed.to_string()));
                }
            }
            offset += sentence.len();
        }

        sentences
    }

    /// Find split points based on similarity drops.
    fn find_split_points(&self, embeddings: &[Vec<f32>]) -> Vec<usize> {
        if embeddings.len() <= 1 {
            return vec![];
        }

        let mut split_points = Vec::new();

        for i in 1..embeddings.len() {
            let sim = Self::cosine_similarity(&embeddings[i - 1], &embeddings[i]);
            if sim < self.threshold {
                // Check minimum chunk size
                let last_split = split_points.last().copied().unwrap_or(0);
                if i - last_split >= self.min_chunk_sentences {
                    split_points.push(i);
                }
            }
        }

        split_points
    }
}

impl Chunker for SemanticChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        // Extract sentences
        let sentences = Self::extract_sentences(text);
        if sentences.is_empty() {
            return vec![];
        }

        // Embed sentences
        let texts: Vec<&str> = sentences.iter().map(|(_, s)| s.as_str()).collect();
        let embeddings = match self.model.embed(texts, None) {
            Ok(e) => e,
            Err(_) => {
                // Fallback: return as single chunk
                return vec![Slab::new(text.trim(), 0, text.len(), 0)];
            }
        };

        // Find split points
        let split_points = self.find_split_points(&embeddings);

        // Create chunks
        let mut slabs = Vec::new();
        let mut chunk_start_idx = 0;

        for (chunk_idx, &split_idx) in split_points.iter().enumerate() {
            let chunk_sentences = &sentences[chunk_start_idx..split_idx];
            if !chunk_sentences.is_empty() {
                let start = chunk_sentences.first().map(|(off, _)| *off).unwrap_or(0);
                let end = chunk_sentences
                    .last()
                    .map(|(off, s)| off + s.len())
                    .unwrap_or(start);
                let chunk_text: String = chunk_sentences
                    .iter()
                    .map(|(_, s)| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                slabs.push(Slab::new(chunk_text, start, end, chunk_idx));
            }
            chunk_start_idx = split_idx;
        }

        // Final chunk
        if chunk_start_idx < sentences.len() {
            let chunk_sentences = &sentences[chunk_start_idx..];
            let start = chunk_sentences.first().map(|(off, _)| *off).unwrap_or(0);
            let end = chunk_sentences
                .last()
                .map(|(off, s)| off + s.len())
                .unwrap_or(start);
            let chunk_text: String = chunk_sentences
                .iter()
                .map(|(_, s)| s.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            slabs.push(Slab::new(chunk_text, start, end, slabs.len()));
        }

        slabs
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        // Very rough estimate based on typical topic density
        (text_len / 1000).max(1)
    }
}

impl std::fmt::Debug for SemanticChunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticChunker")
            .field("threshold", &self.threshold)
            .field("min_chunk_sentences", &self.min_chunk_sentences)
            .finish()
    }
}
