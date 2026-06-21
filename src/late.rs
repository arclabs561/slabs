//! Late chunking: embed first, then pool spans.
//!
//! ## Independent chunk embedding
//!
//! Independent chunk embedding embeds chunks separately:
//!
//! ```text
//! Document: "Einstein developed relativity. He became famous."
//! Chunks:   ["Einstein developed relativity.", "He became famous."]
//! Embeddings: [embed(chunk1), embed(chunk2)]
//!                              ↑
//!                              "He" has no antecedent in this input.
//! ```
//!
//! The second chunk embeds "He" without knowing it refers to Einstein.
//!
//! ## Late pooling
//!
//! Late chunking (Günther et al. 2024) embeds the full document first,
//! then pools token embeddings for each span:
//!
//! ```text
//! Document: "Einstein developed relativity. He became famous."
//!
//! Step 1: Embed full document -> Token embeddings [t1, t2, ..., tn]
//!         Each token "sees" the full document via attention.
//!
//! Step 2: Pool spans from token embeddings:
//!         Span 1: mean_pool([t1, ..., t4])  <- "Einstein developed relativity."
//!         Span 2: mean_pool([t5, ..., t7])  <- "He became famous."
//!                                               "He" now has Einstein context!
//! ```
//!
//! ## Pooling rule
//!
//! Given token embeddings H = [h1, h2, ..., hn] from full document,
//! and span boundaries [(s1, e1), (s2, e2), ...]:
//!
//! ```text
//! span_embedding_i = (1 / |ei - si|) * Σ_{t=si}^{ei} ht
//! ```
//!
//! The returned vector is the L2-normalized mean vector.
//!
//! ## Scope
//!
//! Use this module when boundaries already exist and token embeddings come
//! from a full-document encoder. Boundary selection and embedding generation
//! are upstream concerns.
//!
//! ## Trade-offs
//!
//! | Aspect | Independent chunk embedding | Late pooling |
//! |--------|-------------|---------------|
//! | Memory | O(chunk_size) | O(doc_length × dim) |
//! | Context | Local only | Full document |
//! | Speed | Parallel chunks | Sequential doc first |
//!
//! ## References
//!
//! Günther, Billerbeck, et al. (2024). "Late Chunking: Contextual Chunk
//! Embeddings Using Long-Context Embedding Models." arXiv:2409.04701.

use crate::Slab;

/// Late chunking pooler: pools token embeddings into span embeddings.
///
/// Given token-level embeddings from a full document, it pools the tokens
/// within each [`Slab`] boundary and returns one L2-normalized vector per slab.
#[derive(Debug, Clone)]
pub struct LateChunkingPooler {
    /// Output dimension and expected token embedding dimension.
    dim: usize,
}

impl LateChunkingPooler {
    /// Create a new late chunking pooler.
    ///
    /// # Arguments
    ///
    /// * `dim` - output dimension and expected token embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Pool token embeddings into slab embeddings.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token-level embeddings from full document.
    ///   Shape: [n_tokens, dim]. Each token has "seen" the full document.
    /// * `chunks` - span boundaries from any source.
    /// * `doc_len` - Total document length in bytes (for mapping).
    ///
    /// # Returns
    ///
    /// One L2-normalized mean vector per slab. Each output vector has length
    /// `dim`.
    ///
    /// # Dimension contract
    ///
    /// Token vectors are expected to have `dim` components. Debug builds assert
    /// that contract. Release builds use the first `dim` components and treat
    /// missing components as zero.
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
                // Map byte offsets to token indices (linear approximation).
                let token_start = (chunk.start as f64 / doc_len as f64 * n_tokens as f64) as usize;
                let token_end =
                    ((chunk.end as f64 / doc_len as f64 * n_tokens as f64) as usize).min(n_tokens);

                if token_end <= token_start {
                    // Fallback: use full document average.
                    return self.mean_pool(token_embeddings);
                }

                self.mean_pool(&token_embeddings[token_start..token_end])
            })
            .collect()
    }

    /// Pool with exact token byte offsets.
    ///
    /// Use this when you have exact token offsets from the tokenizer,
    /// rather than relying on linear approximation.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token-level embeddings [n_tokens, dim].
    /// * `token_offsets` - Byte offset for each token [(start, end), ...].
    /// * `chunks` - span boundaries.
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
                // Find tokens that overlap with this slab.
                let token_indices: Vec<usize> = token_offsets
                    .iter()
                    .enumerate()
                    .filter(|(_, (start, end))| {
                        // Token overlaps with slab.
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

    /// Pool with exact token character offsets.
    ///
    /// Use this when a tokenizer reports character offsets instead of byte
    /// offsets. Each `Slab` should have `char_start` and `char_end` populated,
    /// for example by [`Slab::from_char_range`](crate::Slab::from_char_range)
    /// or [`crate::compute_char_offsets`]. A slab without character offsets
    /// falls back to the full-document average.
    pub fn pool_with_char_offsets(
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
                let Some(span) = chunk.char_span() else {
                    return self.mean_pool(token_embeddings);
                };

                let token_indices: Vec<usize> = token_offsets
                    .iter()
                    .enumerate()
                    .filter(|(_, (start, end))| *start < span.end && *end > span.start)
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

        let mut result = vec![0.0; self.dim];
        let count = embeddings.len() as f32;

        for emb in embeddings {
            debug_assert_eq!(
                emb.len(),
                self.dim,
                "token embedding dimension mismatch: expected {}, got {}",
                self.dim,
                emb.len()
            );
            for (i, &v) in emb.iter().take(self.dim).enumerate() {
                result[i] += v;
            }
        }

        for v in &mut result {
            *v /= count;
        }

        // L2 normalize.
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

        let mut result = vec![0.0; self.dim];
        let count = embeddings.len() as f32;

        for emb in embeddings {
            debug_assert_eq!(
                emb.len(),
                self.dim,
                "token embedding dimension mismatch: expected {}, got {}",
                self.dim,
                emb.len()
            );
            for (i, &v) in emb.iter().take(self.dim).enumerate() {
                result[i] += v;
            }
        }

        for v in &mut result {
            *v /= count;
        }

        // L2 normalize.
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let spans = vec![
            Slab::new("first chunk", 0, 10, 0),
            Slab::new("second chunk", 10, 20, 1),
        ];

        let span_embeddings = pooler.pool(&token_embeddings, &spans, 20);

        assert_eq!(span_embeddings.len(), 2);
        assert_eq!(span_embeddings[0].len(), 4);
        assert_eq!(span_embeddings[1].len(), 4);

        // Embeddings should be normalized
        let norm0: f32 = span_embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm0 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pool_with_exact_offsets() {
        let pooler = LateChunkingPooler::new(3);

        // 5 tokens with known byte offsets
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
            Slab::new("Hello world.", 0, 12, 0),
            Slab::new(" Bye", 12, 16, 1),
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

        let chunks = vec![Slab::new("test", 0, 4, 0)];

        let result = pooler.pool(&[], &chunks, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
    }

    #[test]
    fn pool_uses_configured_output_dimension() {
        let pooler = LateChunkingPooler::new(3);
        let chunks = vec![Slab::new("abc", 0, 3, 0)];
        let token_embeddings = vec![vec![2.0, 0.0, 0.0], vec![0.0, 2.0, 0.0]];

        let pooled = pooler.pool(&token_embeddings, &chunks, 3);

        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0].len(), 3);
        let norm = pooled[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn pool_with_offsets_uses_configured_output_dimension() {
        let pooler = LateChunkingPooler::new(3);
        let chunks = vec![Slab::new("abc", 0, 3, 0)];
        let token_embeddings = vec![vec![2.0, 0.0, 0.0]];
        let token_offsets = vec![(0, 3)];

        let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0].len(), 3);
    }

    #[test]
    fn pool_with_offsets_uses_byte_spans() {
        let pooler = LateChunkingPooler::new(2);
        let text = "éclair cake";
        let chunks = vec![Slab::from_byte_range(text, 0..7, 0).unwrap()];
        let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let token_offsets = vec![(0, 7), (8, 12)];

        let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled[0], vec![1.0, 0.0]);
    }

    #[test]
    fn pool_with_char_offsets_uses_character_spans() {
        let pooler = LateChunkingPooler::new(2);
        let text = "éclair cake";
        let chunks = vec![Slab::from_char_range(text, 0..6, 0).unwrap()];
        let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let token_offsets = vec![(0, 6), (7, 11)];

        let pooled = pooler.pool_with_char_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled[0], vec![1.0, 0.0]);
    }
}
