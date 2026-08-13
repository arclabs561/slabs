//! Span pooling for full-document token embeddings.
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
//! ## Span pooling
//!
//! Late chunking (Günther et al. 2024) embeds the full document first,
//! then pools token embeddings for each selected span:
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
//! | Aspect | Independent chunk embedding | Span pooling |
//! |--------|-------------|---------------|
//! | Memory | O(chunk_size) | O(doc_length × dim) |
//! | Context | Local only | Full document |
//! | Speed | Parallel chunks | Sequential doc first |
//!
//! ## References
//!
//! Günther, Billerbeck, et al. (2024). "Late Chunking: Contextual Chunk
//! Embeddings Using Long-Context Embedding Models." arXiv:2409.04701.

use std::ops::Range;

use crate::Slab;

/// Invalid input to exact span pooling.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PoolingError {
    /// The number of token embeddings did not match the number of token spans.
    #[error("token embedding count {embeddings} does not match token offset count {offsets}")]
    TokenCountMismatch {
        /// Number of token embedding vectors.
        embeddings: usize,
        /// Number of token offset pairs.
        offsets: usize,
    },
    /// A token embedding had a different dimension than the pooler.
    #[error("token embedding {token} has dimension {actual}; expected {expected}")]
    EmbeddingDimension {
        /// Index of the malformed token embedding.
        token: usize,
        /// Configured embedding dimension.
        expected: usize,
        /// Actual embedding dimension.
        actual: usize,
    },
    /// The pooler was configured with a zero-dimensional output.
    #[error("pooling dimension must be greater than zero")]
    ZeroDimension,
    /// A non-empty token offset was reversed, overlapping, or out of order.
    #[error("invalid token offset at index {token}: {start}..{end}")]
    InvalidTokenOffset {
        /// Index of the malformed token offset.
        token: usize,
        /// Start offset.
        start: usize,
        /// End offset.
        end: usize,
    },
    /// Slabs were not ordered by their selected coordinate.
    #[error("slab at index {slab} starts before the preceding slab")]
    UnsortedSlabs {
        /// Index of the first out-of-order slab.
        slab: usize,
    },
    /// A slab span was empty or reversed.
    #[error("invalid slab span at index {slab}: {start}..{end}")]
    InvalidSlabSpan {
        /// Index of the malformed slab.
        slab: usize,
        /// Start offset in the selected coordinate.
        start: usize,
        /// End offset in the selected coordinate.
        end: usize,
    },
    /// Character pooling requires character offsets on every slab.
    #[error("slab at index {slab} has no character offsets")]
    MissingCharacterOffsets {
        /// Index of the slab without character offsets.
        slab: usize,
    },
    /// No token overlaps a slab.
    #[error("no token overlaps slab at index {slab}")]
    NoTokenOverlap {
        /// Index of the slab without a token.
        slab: usize,
    },
}

/// Pools token embeddings into span embeddings.
///
/// Given token-level embeddings from a full document, it pools the tokens
/// within each [`Slab`] boundary and returns one L2-normalized vector per slab.
#[derive(Debug, Clone)]
pub struct SpanPooler {
    /// Output dimension and expected token embedding dimension.
    dim: usize,
}

/// Compatibility alias for the old pooler name.
///
/// Use [`SpanPooler`] in new code. The old name remains available because
/// existing callers may still use late-chunking vocabulary for the full
/// pipeline. This crate only owns the span-pooling primitive.
#[deprecated(
    since = "0.4.0",
    note = "use SpanPooler; slabs owns span pooling, not the full late-chunking pipeline"
)]
pub type LateChunkingPooler = SpanPooler;

impl SpanPooler {
    /// Create a new span pooler.
    ///
    /// # Arguments
    ///
    /// * `dim` - output dimension and expected token embedding dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Pool token embeddings into slab embeddings by approximate position.
    ///
    /// # Arguments
    ///
    /// * `token_embeddings` - Token-level embeddings from the full document.
    ///   Shape: [n_tokens, dim]. Each token has "seen" the full document.
    /// * `chunks` - Span boundaries from any source.
    /// * `doc_len` - Total document length in bytes.
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
    ///
    /// # Precision
    ///
    /// This method linearly maps byte offsets to token indices. Prefer
    /// [`pool_with_offsets`](SpanPooler::pool_with_offsets) or
    /// [`pool_with_char_offsets`](SpanPooler::pool_with_char_offsets) when a
    /// tokenizer reports exact offsets.
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
    ///
    /// Invalid input retains the legacy permissive behavior for compatibility.
    /// Prefer [`try_pool_with_offsets`](Self::try_pool_with_offsets) when input
    /// errors must not be hidden.
    pub fn pool_with_offsets(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
    ) -> Vec<Vec<f32>> {
        self.try_pool_with_offsets(token_embeddings, token_offsets, chunks)
            .unwrap_or_else(|_| {
                self.compatibility_pool_exact(token_embeddings, token_offsets, chunks, |chunk| {
                    Some(chunk.start..chunk.end)
                })
            })
    }

    /// Pool with exact token byte offsets, validating every input contract.
    ///
    /// Non-empty token offsets must be non-overlapping and ordered. Empty
    /// offsets are ignored, as tokenizers commonly use them for special tokens.
    /// Slabs may overlap, but must be ordered by byte start. Every slab must
    /// overlap at least one token.
    ///
    /// # Errors
    ///
    /// Returns an error for count or dimension mismatches, malformed ordering,
    /// invalid half-open token spans, or a slab with no overlapping token.
    pub fn try_pool_with_offsets(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
    ) -> std::result::Result<Vec<Vec<f32>>, PoolingError> {
        self.try_pool_exact(token_embeddings, token_offsets, chunks, |chunk| {
            Some(chunk.start..chunk.end)
        })
    }

    /// Pool with exact token character offsets.
    ///
    /// Use this when a tokenizer reports character offsets instead of byte
    /// offsets. Each `Slab` should have `char_start` and `char_end` populated,
    /// for example by [`Slab::from_char_range`](crate::Slab::from_char_range)
    /// or [`crate::compute_char_offsets`]. A slab without character offsets
    /// falls back to the full-document average.
    /// Other invalid input retains the legacy permissive behavior. Prefer
    /// [`try_pool_with_char_offsets`](Self::try_pool_with_char_offsets) when
    /// input errors must not be hidden.
    pub fn pool_with_char_offsets(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
    ) -> Vec<Vec<f32>> {
        self.try_pool_with_char_offsets(token_embeddings, token_offsets, chunks)
            .unwrap_or_else(|_| {
                self.compatibility_pool_exact(
                    token_embeddings,
                    token_offsets,
                    chunks,
                    Slab::char_span,
                )
            })
    }

    /// Pool with exact token character offsets, validating every input contract.
    ///
    /// Non-empty token offsets must be non-overlapping and ordered. Empty
    /// offsets are ignored, as tokenizers commonly use them for special tokens.
    /// Slabs may overlap, but must have character offsets and be ordered by
    /// character start. Every slab must overlap at least one token.
    ///
    /// # Errors
    ///
    /// Returns an error for count or dimension mismatches, malformed ordering,
    /// missing character offsets, invalid half-open token spans, or a slab with
    /// no overlapping token.
    pub fn try_pool_with_char_offsets(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
    ) -> std::result::Result<Vec<Vec<f32>>, PoolingError> {
        self.try_pool_exact(token_embeddings, token_offsets, chunks, Slab::char_span)
    }

    fn try_pool_exact<F>(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
        span_of: F,
    ) -> std::result::Result<Vec<Vec<f32>>, PoolingError>
    where
        F: Fn(&Slab) -> Option<Range<usize>>,
    {
        self.validate_tokens(token_embeddings, token_offsets)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let mut spans = Vec::with_capacity(chunks.len());
        let mut previous_start = None;
        for (slab, chunk) in chunks.iter().enumerate() {
            let span = span_of(chunk).ok_or(PoolingError::MissingCharacterOffsets { slab })?;
            if span.start >= span.end {
                return Err(PoolingError::InvalidSlabSpan {
                    slab,
                    start: span.start,
                    end: span.end,
                });
            }
            if previous_start.is_some_and(|start| span.start < start) {
                return Err(PoolingError::UnsortedSlabs { slab });
            }
            previous_start = Some(span.start);
            spans.push(span);
        }

        let mut first_candidate = 0usize;
        let mut pooled = Vec::with_capacity(spans.len());
        for (slab, span) in spans.into_iter().enumerate() {
            while first_candidate < token_offsets.len()
                && token_offsets[first_candidate].1 <= span.start
            {
                first_candidate += 1;
            }

            let mut sum = vec![0.0; self.dim];
            let mut count = 0usize;
            for (offset, embedding) in token_offsets[first_candidate..]
                .iter()
                .zip(&token_embeddings[first_candidate..])
            {
                if offset.0 == offset.1 {
                    continue;
                }
                if offset.0 >= span.end {
                    break;
                }
                if offset.1 > span.start {
                    for (total, value) in sum.iter_mut().zip(embedding) {
                        *total += value;
                    }
                    count += 1;
                }
            }
            if count == 0 {
                return Err(PoolingError::NoTokenOverlap { slab });
            }
            Self::normalize_mean(&mut sum, count);
            pooled.push(sum);
        }
        Ok(pooled)
    }

    fn validate_tokens(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
    ) -> std::result::Result<(), PoolingError> {
        if self.dim == 0 {
            return Err(PoolingError::ZeroDimension);
        }
        if token_embeddings.len() != token_offsets.len() {
            return Err(PoolingError::TokenCountMismatch {
                embeddings: token_embeddings.len(),
                offsets: token_offsets.len(),
            });
        }
        let mut previous_end = 0usize;
        for (token, ((start, end), embedding)) in
            token_offsets.iter().zip(token_embeddings).enumerate()
        {
            if start > end || (start != end && *start < previous_end) {
                return Err(PoolingError::InvalidTokenOffset {
                    token,
                    start: *start,
                    end: *end,
                });
            }
            if embedding.len() != self.dim {
                return Err(PoolingError::EmbeddingDimension {
                    token,
                    expected: self.dim,
                    actual: embedding.len(),
                });
            }
            if start != end {
                previous_end = *end;
            }
        }
        Ok(())
    }

    fn compatibility_pool_exact<F>(
        &self,
        token_embeddings: &[Vec<f32>],
        token_offsets: &[(usize, usize)],
        chunks: &[Slab],
        span_of: F,
    ) -> Vec<Vec<f32>>
    where
        F: Fn(&Slab) -> Option<Range<usize>>,
    {
        chunks
            .iter()
            .map(|chunk| {
                let Some(span) = span_of(chunk) else {
                    return self.mean_pool(token_embeddings);
                };
                let mut sum = vec![0.0; self.dim];
                let mut count = 0usize;
                let mut overlaps = false;
                for (token, offset) in token_offsets.iter().enumerate() {
                    if offset.0 < span.end && offset.1 > span.start {
                        overlaps = true;
                        if let Some(embedding) = token_embeddings.get(token) {
                            debug_assert_eq!(embedding.len(), self.dim);
                            for (total, value) in sum.iter_mut().zip(embedding) {
                                *total += value;
                            }
                            count += 1;
                        }
                    }
                }
                if !overlaps {
                    return self.mean_pool(token_embeddings);
                }
                if count > 0 {
                    Self::normalize_mean(&mut sum, count);
                }
                sum
            })
            .collect()
    }

    fn normalize_mean(result: &mut [f32], count: usize) {
        for value in result.iter_mut() {
            *value /= count as f32;
        }
        let norm = result.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for value in result {
                *value /= norm;
            }
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_pooler_basic() {
        let pooler = SpanPooler::new(4);

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
        let pooler = SpanPooler::new(3);

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
        let pooler = SpanPooler::new(4);

        let result = pooler.pool(&[], &[], 0);
        assert!(result.is_empty());

        let chunks = vec![Slab::new("test", 0, 4, 0)];

        let result = pooler.pool(&[], &chunks, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 4);
    }

    #[test]
    fn pool_uses_configured_output_dimension() {
        let pooler = SpanPooler::new(3);
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
        let pooler = SpanPooler::new(3);
        let chunks = vec![Slab::new("abc", 0, 3, 0)];
        let token_embeddings = vec![vec![2.0, 0.0, 0.0]];
        let token_offsets = vec![(0, 3)];

        let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0].len(), 3);
    }

    #[test]
    fn pool_with_offsets_uses_byte_spans() {
        let pooler = SpanPooler::new(2);
        let text = "éclair cake";
        let chunks = vec![Slab::from_byte_range(text, 0..7, 0).unwrap()];
        let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let token_offsets = vec![(0, 7), (8, 12)];

        let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled[0], vec![1.0, 0.0]);
    }

    #[test]
    fn pool_with_char_offsets_uses_character_spans() {
        let pooler = SpanPooler::new(2);
        let text = "éclair cake";
        let chunks = vec![Slab::from_char_range(text, 0..6, 0).unwrap()];
        let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let token_offsets = vec![(0, 6), (7, 11)];

        let pooled = pooler.pool_with_char_offsets(&token_embeddings, &token_offsets, &chunks);

        assert_eq!(pooled[0], vec![1.0, 0.0]);
    }
}
