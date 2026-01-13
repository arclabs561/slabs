//! Error types for slabs.

/// Errors that can occur during chunking.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid chunk size (must be > 0).
    #[error("invalid chunk size: {0} (must be > 0)")]
    InvalidChunkSize(usize),

    /// Overlap exceeds chunk size.
    #[error("overlap {overlap} exceeds chunk size {size}")]
    OverlapExceedsSize {
        /// The chunk size.
        size: usize,
        /// The overlap that exceeded the size.
        overlap: usize,
    },

    /// Semantic chunking requires the `semantic` feature.
    #[error("semantic chunking requires the 'semantic' feature")]
    SemanticFeatureRequired,

    /// Embedding model error.
    #[error("embedding error: {0}")]
    Embedding(String),
}

/// Result type for slabs operations.
pub type Result<T> = std::result::Result<T, Error>;
