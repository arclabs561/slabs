//! Model-based chunking strategies.
//!
//! Uses a token-classification model (like `chonky`) or embedding transitions
//! to identify semantic boundaries.

use crate::{Chunker, Slab};
use std::sync::Arc;

/// A trait for token classification models used in chunking.
/// This allows plugging in different backends (ORT, Candle, etc.).
pub trait TokenClassifier: Send + Sync {
    /// Predict split points for the given text.
    /// Returns a list of byte offsets where splits should occur.
    fn predict_splits(&self, text: &str) -> Vec<usize>;
}

/// A chunker that uses a machine learning model to predict boundaries.
pub struct ModelChunker {
    model: Arc<dyn TokenClassifier>,
}

impl ModelChunker {
    /// Create a new model-based chunker with a specific model backend.
    pub fn new(model: Arc<dyn TokenClassifier>) -> Self {
        Self { model }
    }
}

impl Chunker for ModelChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        if text.is_empty() {
            return vec![];
        }

        let split_points = self.model.predict_splits(text);

        let mut slabs = Vec::with_capacity(split_points.len() + 1);
        let mut start = 0;

        for (i, end) in split_points.into_iter().enumerate() {
            if end > start && end <= text.len() {
                slabs.push(Slab::new(&text[start..end], start, end, i));
                start = end;
            }
        }

        // Add final chunk
        if start < text.len() {
            slabs.push(Slab::new(&text[start..], start, text.len(), slabs.len()));
        }

        slabs
    }
}

// TODO: Implement concrete TokenClassifier for ONNX Runtime (using fastembed/ort)
// when the 'semantic' feature is enabled.
