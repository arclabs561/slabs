//! Chunk capacity configuration.
//!
//! ## The Problem
//!
//! Fixed chunk sizes create a tension:
//!
//! - Too small: Fragments meaning, poor embeddings
//! - Too large: Dilutes semantics, retrieval precision drops
//! - "Just right": Depends on your content and use case
//!
//! But there's a subtler issue: rigid sizes force awkward splits.
//!
//! ```text
//! Target: 100 chars
//! Text: "Introduction. [98 chars]. The key insight is..."
//!
//! Rigid:  ["Introduction. [98 chars].", "The key insight is..."]
//!         ↑ Splits at semantic boundary but loses context
//!
//! Flexible (target=100, max=120):
//!         ["Introduction. [98 chars]. The key insight is..."]
//!         ↑ Stays at paragraph level, slightly over target
//! ```
//!
//! ## The Solution: Desired vs Max
//!
//! `ChunkCapacity` separates target size from hard limit:
//!
//! - `desired`: What we're aiming for. Chunks will be as close as possible.
//! - `max`: The absolute ceiling. Never exceeded.
//!
//! This lets chunkers stay at higher semantic levels (paragraphs > sentences > words)
//! when doing so would only slightly exceed the target.

use std::cmp::Ordering;

/// Configuration for chunk size with flexible target and hard limit.
///
/// # Examples
///
/// ```rust
/// use slabs::ChunkCapacity;
///
/// // Fixed size: desired == max
/// let cap = ChunkCapacity::new(512);
/// assert_eq!(cap.desired(), 512);
/// assert_eq!(cap.max(), 512);
///
/// // Flexible: aim for 512, allow up to 640
/// let cap = ChunkCapacity::new(512).with_max(640).unwrap();
/// assert_eq!(cap.desired(), 512);
/// assert_eq!(cap.max(), 640);
///
/// // Range syntax
/// let cap = ChunkCapacity::from(400..600);
/// assert_eq!(cap.desired(), 400);
/// assert_eq!(cap.max(), 599);  // exclusive end
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCapacity {
    desired: usize,
    max: usize,
}

impl ChunkCapacity {
    /// Create a capacity with the same desired and max size.
    ///
    /// Use this when you need strict size limits (e.g., embedding model max tokens).
    #[must_use]
    pub const fn new(size: usize) -> Self {
        Self {
            desired: size,
            max: size,
        }
    }

    /// The target chunk size.
    ///
    /// Chunks will be as close to this as possible while respecting semantic boundaries.
    #[must_use]
    pub const fn desired(&self) -> usize {
        self.desired
    }

    /// The maximum allowed chunk size.
    ///
    /// Chunks will never exceed this, even if it means breaking at a lower semantic level.
    #[must_use]
    pub const fn max(&self) -> usize {
        self.max
    }

    /// Set a maximum size larger than the desired size.
    ///
    /// This allows the chunker to stay at higher semantic levels
    /// when doing so would only slightly exceed the target.
    ///
    /// # Errors
    ///
    /// Returns an error if `max < desired`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use slabs::ChunkCapacity;
    ///
    /// let cap = ChunkCapacity::new(500).with_max(600).unwrap();
    /// assert_eq!(cap.desired(), 500);
    /// assert_eq!(cap.max(), 600);
    /// ```
    pub fn with_max(self, max: usize) -> Result<Self, ChunkCapacityError> {
        if max < self.desired {
            Err(ChunkCapacityError::MaxLessThanDesired {
                desired: self.desired,
                max,
            })
        } else {
            Ok(Self { max, ..self })
        }
    }

    /// Check if a chunk size fits within this capacity.
    ///
    /// Returns:
    /// - `Ordering::Less`: Chunk is smaller than desired, can add more
    /// - `Ordering::Equal`: Chunk is in the sweet spot (desired..=max)
    /// - `Ordering::Greater`: Chunk exceeds max, must split
    #[must_use]
    pub fn fits(&self, size: usize) -> Ordering {
        if size < self.desired {
            Ordering::Less
        } else if size > self.max {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    /// Check if adding `additional` bytes would exceed the capacity.
    ///
    /// Useful for incremental chunk building.
    #[must_use]
    pub fn would_overflow(&self, current: usize, additional: usize) -> bool {
        current.saturating_add(additional) > self.max
    }
}

impl Default for ChunkCapacity {
    fn default() -> Self {
        // Reasonable default for RAG: ~512 tokens, assuming ~4 chars/token
        Self::new(2048)
    }
}

impl From<usize> for ChunkCapacity {
    fn from(size: usize) -> Self {
        Self::new(size)
    }
}

impl From<std::ops::Range<usize>> for ChunkCapacity {
    fn from(range: std::ops::Range<usize>) -> Self {
        Self {
            desired: range.start,
            max: range.end.saturating_sub(1).max(range.start),
        }
    }
}

impl From<std::ops::RangeInclusive<usize>> for ChunkCapacity {
    fn from(range: std::ops::RangeInclusive<usize>) -> Self {
        Self {
            desired: *range.start(),
            max: *range.end(),
        }
    }
}

/// Error when configuring chunk capacity.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChunkCapacityError {
    /// Max size must be >= desired size.
    #[error("max ({max}) must be >= desired ({desired})")]
    MaxLessThanDesired {
        /// The desired chunk size.
        desired: usize,
        /// The max that was too small.
        max: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_capacity() {
        let cap = ChunkCapacity::new(100);
        assert_eq!(cap.fits(50), Ordering::Less);
        assert_eq!(cap.fits(100), Ordering::Equal);
        assert_eq!(cap.fits(150), Ordering::Greater);
    }

    #[test]
    fn test_flexible_capacity() {
        let cap = ChunkCapacity::new(100).with_max(120).unwrap();
        assert_eq!(cap.fits(50), Ordering::Less);
        assert_eq!(cap.fits(100), Ordering::Equal);
        assert_eq!(cap.fits(110), Ordering::Equal); // in range
        assert_eq!(cap.fits(120), Ordering::Equal);
        assert_eq!(cap.fits(121), Ordering::Greater);
    }

    #[test]
    fn test_range_conversion() {
        let cap = ChunkCapacity::from(100..200);
        assert_eq!(cap.desired(), 100);
        assert_eq!(cap.max(), 199); // exclusive
    }

    #[test]
    fn test_range_inclusive_conversion() {
        let cap = ChunkCapacity::from(100..=200);
        assert_eq!(cap.desired(), 100);
        assert_eq!(cap.max(), 200); // inclusive
    }

    #[test]
    fn test_would_overflow() {
        let cap = ChunkCapacity::new(100);
        assert!(!cap.would_overflow(50, 49));
        assert!(!cap.would_overflow(50, 50));
        assert!(cap.would_overflow(50, 51));
    }

    #[test]
    fn test_max_less_than_desired_error() {
        let result = ChunkCapacity::new(100).with_max(50);
        assert!(result.is_err());
    }
}
