//! Error types for slabs.

/// Errors that can occur during slab construction or adapter code.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A byte span was outside the source text or had `start > end`.
    #[error("invalid byte span {start}..{end} for source length {len}")]
    InvalidByteSpan {
        /// Start byte offset.
        start: usize,
        /// End byte offset.
        end: usize,
        /// Source length in bytes.
        len: usize,
    },

    /// A character span was outside the source text or had `start > end`.
    #[error("invalid character span {start}..{end} for source length {len}")]
    InvalidCharSpan {
        /// Start character offset.
        start: usize,
        /// End character offset.
        end: usize,
        /// Source length in characters.
        len: usize,
    },

    /// A byte span endpoint did not fall on a UTF-8 character boundary.
    #[error("byte offset {offset} is not a UTF-8 character boundary")]
    NonCharBoundary {
        /// Invalid byte offset.
        offset: usize,
    },

    /// Compatibility error for adapters that map upstream embedding failures
    /// into `slabs::Error`.
    #[error("embedding error: {0}")]
    Embedding(String),
}

/// Result type for slabs operations.
pub type Result<T> = std::result::Result<T, Error>;
