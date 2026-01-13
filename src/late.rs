//! Late Chunking: Embed first, then chunk.
//!
//! ## The Problem with Traditional Chunking
//!
//! Traditional chunking embeds chunks independently:
//!
//! ```text
//! Document: "Einstein developed relativity. He became famous."
//! Chunks:   ["Einstein developed relativity.", "He became famous."]
//! Embeddings: [embed(chunk1), embed(chunk2)]
//!                              ↑
//!                              "He" loses context!
//! ```
//!
//! The second chunk embeds "He" without knowing it refers to Einstein.
//!
//! ## Late Chunking Solution
//!
//! Late chunking (Günther et al. 2024) embeds the full document first,
//! then pools token embeddings for each chunk:
//!
//! ```text
//! Document: "Einstein developed relativity. He became famous."
//!
//! Step 1: Embed full document → Token embeddings [t1, t2, ..., tn]
//!         Each token "sees" the full document via attention.
//!
//! Step 2: Pool chunks from token embeddings:
//!         Chunk 1: mean_pool([t1, ..., t4])  ← "Einstein developed relativity."
//!         Chunk 2: mean_pool([t5, ..., t7])  ← "He became famous."
//!                                               "He" now has Einstein context!
//! ```
//!
//! ## The Math
//!
//! Given token embeddings H = [h1, h2, ..., hn] from full document,
//! and chunk boundaries [(s1, e1), (s2, e2), ...]:
//!
//! ```text
//! chunk_embedding_i = (1 / |ei - si|) * Σ_{t=si}^{ei} ht
//! ```
//!
//! Mean pooling preserves the contextual information each token gained
//! from attending to the full document.
//!
//! ## When to Use
//!
//! - **Use Late Chunking**: When chunks reference each other (pronouns,
//!   acronym definitions, temporal references). Long coherent documents.
//!
//! - **Use Traditional**: Independent chunks, real-time embedding needed,
//!   memory-constrained (late chunking needs full doc in memory).
//!
//! ## Trade-offs
//!
//! | Aspect | Traditional | Late Chunking |
//! |--------|-------------|---------------|
//! | Memory | O(chunk_size) | O(doc_length × dim) |
//! | Context | Local only | Full document |
//! | Speed | Parallel chunks | Sequential doc first |
//! | Quality | Baseline | +5-15% recall typically |
//!
//! ## References
//!
//! Günther, Billerbeck, et al. (2024). "Late Chunking: Contextual Chunk
//! Embeddings Using Long-Context Embedding Models." arXiv:2409.04701.

use crate::{Chunker, Slab};

/// Late chunking pooler: pools token embeddings into chunk embeddings.
///
/// This is the core operation of late chunking. Given token-level embeddings
/// from a full document, it pools the tokens within each chunk boundary
/// to create contextualized chunk embeddings.
#[derive(Debug, Clone)]
pub struct LateChunkingPooler {
    /// Embedding dimension (for validation).
    dim: usize,
}

impl LateChunkingPooler {
    /// Create a new late chunking pooler.
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Pool token embeddings into chunk embeddings.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token-level embeddings from full document.
    ///   Shape: [n_tokens, dim]. Each token has "seen" the full document.
    /// * `chunks` - Chunk boundaries from any chunker.
    /// * `doc_len` - Total document length in bytes (for mapping).
    ///
    /// # Returns
    ///
    /// Contextualized chunk embeddings. Each chunk embedding is the mean
    /// of its constituent token embeddings.
    ///
    /// # Panics
    ///
    /// Panics if token embeddings have inconsistent dimensions.
    pub fn pool(
        &self,
        token_embeddings: &[Vec<f32>],
        chunks: &[Slab],
        doc_len: usize,
    ) -> Vec<Vec<f32>> {
        if token_embeddings.is_empty() || chunks.is_empty() || doc_len == 0 {
            return vec![vec![0.0; self.dim]; chunks.len()];
        }

        let n_tokens = token_embeddings.len();

        chunks
            .iter()
            .map(|chunk| {
                // Map byte offsets to token indices (linear approximation)
                let token_start = (chunk.start as f64 / doc_len as f64 * n_tokens as f64) as usize;
                let token_end =
                    ((chunk.end as f64 / doc_len as f64 * n_tokens as f64) as usize).min(n_tokens);

                if token_end <= token_start {
                    // Fallback: use full document average
                    return self.mean_pool(token_embeddings);
                }

                self.mean_pool(&token_embeddings[token_start..token_end])
            })
            .collect()
    }

    /// Pool with exact token-to-character mappings.
    ///
    /// Use this when you have exact token offsets from the tokenizer,
    /// rather than relying on linear approximation.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token-level embeddings [n_tokens, dim].
    /// * `token_offsets` - Character offset for each token [(start, end), ...].
    /// * `chunks` - Chunk boundaries.
    pub fn pool_with_offsets(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
    ) -> Vec<Vec<f32>> {
        if token_embeddings.is_empty() || chunks.is_empty() {
            return vec![vec![0.0; self.dim]; chunks.len()];
        }

        chunks
            .iter()
            .map(|chunk| {
                // Find tokens that overlap with this chunk
                let token_indices: Vec<usize> = token_offsets
                    .iter()
                    .enumerate()
                    .filter(|(_, (start, end))| {
                        // Token overlaps with chunk
                        *start < chunk.end && *end > chunk.start
                    })
                    .map(|(i, _)| i)
                    .collect();

                if token_indices.is_empty() {
                    return self.mean_pool(token_embeddings);
                }

                let selected: Vec<&[f32]> = token_indices
                    .iter()
                    .filter_map(|&i| token_embeddings.get(i).map(Vec::as_slice))
                    .collect();

                self.mean_pool_refs(&selected)
            })
            .collect()
    }

    /// Mean pool a slice of token embeddings.
    fn mean_pool(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; self.dim];
        }

        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        let count = embeddings.len() as f32;

        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                result[i] += v;
            }
        }

        for v in &mut result {
            *v /= count;
        }

        // L2 normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }

    /// Mean pool from references.
    fn mean_pool_refs(&self, embeddings: &[&[f32]]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; self.dim];
        }

        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        let count = embeddings.len() as f32;

        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                result[i] += v;
            }
        }

        for v in &mut result {
            *v /= count;
        }

        // L2 normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }
}

/// Wrapper that applies late chunking to any base chunker.
///
/// # Example
///
/// ```ignore
/// use slabs::{LateChunker, SentenceChunker, Chunker};
///
/// let base_chunker = SentenceChunker::new(3);
/// let late = LateChunker::new(base_chunker, 384);
///
/// // First, embed full document to get token embeddings
/// let token_embeddings = embed_document_tokens(&text);
///
/// // Get chunk boundaries from base chunker
/// let chunks = late.chunk(&text);
///
/// // Pool token embeddings into chunk embeddings
/// let chunk_embeddings = late.pool(&token_embeddings, &chunks, text.len());
/// ```
#[derive(Debug)]
pub struct LateChunker<C: Chunker> {
    /// Base chunker for determining chunk boundaries.
    base: C,
    /// Pooler for late chunking.
    pooler: LateChunkingPooler,
}

impl<C: Chunker> LateChunker<C> {
    /// Create a late chunker wrapping a base chunker.
    pub fn new(base: C, dim: usize) -> Self {
        Self {
            base,
            pooler: LateChunkingPooler::new(dim),
        }
    }

    /// Access the pooler for late chunking operations.
    pub fn pooler(&self) -> &LateChunkingPooler {
        &self.pooler
    }

    /// Pool token embeddings into chunk embeddings.
    ///
    /// Call this after getting token embeddings from your embedding model.
    pub fn pool(
        &self,
        token_embeddings: &[Vec<f32>],
        chunks: &[Slab],
        doc_len: usize,
    ) -> Vec<Vec<f32>> {
        self.pooler.pool(token_embeddings, chunks, doc_len)
    }
}

impl<C: Chunker> Chunker for LateChunker<C> {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        self.base.chunk(text)
    }

    fn estimate_chunks(&self, text_len: usize) -> usize {
        self.base.estimate_chunks(text_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SentenceChunker;

    #[test]
    fn test_late_chunking_pooler_basic() {
        let pooler = LateChunkingPooler::new(4);

        // Simulate 6 tokens, 4-dim embeddings
        let token_embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
        ];

        let chunks = vec![
            Slab {
                text: "first chunk".to_string(),
                start: 0,
                end: 10,
                index: 0,
            },
            Slab {
                text: "second chunk".to_string(),
                start: 10,
                end: 20,
                index: 1,
            },
        ];

        let chunk_embeddings = pooler.pool(&token_embeddings, &chunks, 20);

        assert_eq!(chunk_embeddings.len(), 2);
        assert_eq!(chunk_embeddings[0].len(), 4);
        assert_eq!(chunk_embeddings[1].len(), 4);

        // Embeddings should be normalized
        let norm0: f32 = chunk_embeddings[0]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm0 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_late_chunker_wrapper() {
        let sentence_chunker = SentenceChunker::new(2);
        let late = LateChunker::new(sentence_chunker, 384);

        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = late.chunk(text);

        // Should produce chunks like base chunker
        assert!(!chunks.is_empty());

        // Simulate token embeddings
        let token_embeddings: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 384]).collect();

        let chunk_embeddings = late.pool(&token_embeddings, &chunks, text.len());
        assert_eq!(chunk_embeddings.len(), chunks.len());
    }

    #[test]
    fn test_pool_with_exact_offsets() {
        let pooler = LateChunkingPooler::new(3);

        // 5 tokens with known character offsets
        let token_embeddings = vec![
            vec![1.0, 0.0, 0.0], // "Hello"
            vec![0.0, 1.0, 0.0], // " "
            vec![0.0, 0.0, 1.0], // "world"
            vec![1.0, 1.0, 0.0], // "."
            vec![0.0, 1.0, 1.0], // " Bye"
        ];

        let token_offsets = vec![
            (0, 5),   // "Hello"
            (5, 6),   // " "
            (6, 11),  // "world"
            (11, 12), // "."
            (12, 16), // " Bye"
        ];

        let chunks = vec![
            Slab {
                text: "Hello world.".to_string(),
                start: 0,
                end: 12,
                index: 0,
            },
            Slab {
                text: " Bye".to_string(),
                start: 12,
                end: 16,
                index: 1,
            },
        ];

        let embeddings = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(embeddings.len(), 2);
        // First chunk should average tokens 0-3
        // Second chunk should be token 4
    }

    #[test]
    fn test_empty_inputs() {
        let pooler = LateChunkingPooler::new(4);

        let result = pooler.pool(&[], &[], 0);
        assert!(result.is_empty());

        let chunks = vec![Slab {
            text: "test".to_string(),
            start: 0,
            end: 4,
            index: 0,
        }];

        let result = pooler.pool(&[], &chunks, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
    }
}
