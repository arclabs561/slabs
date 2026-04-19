//! Error types for code-chunker.

/// Errors that can occur during chunking.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Embedding model error.
    #[error("embedding error: {0}")]
    Embedding(String),
}

/// Result type for code-chunker operations.
pub type Result<T> = std::result::Result<T, Error>;
