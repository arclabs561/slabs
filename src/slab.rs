//! The Slab type: a text span with position metadata.

use std::ops::Range;

use crate::{Error, Result};

/// A text span with its position in the source string.
///
/// A slab is a self-contained span that can be embedded, indexed, and
/// retrieved while preserving where it came from in the text used to create it.
///
/// ## Offsets
///
/// Primary offsets (`start`/`end`) are byte offsets into the source string,
/// matching Rust's string slicing semantics:
///
/// ```rust
/// use slabs::Slab;
///
/// let text = "Hello, world!";
/// let slab = Slab::new("world", 7, 12, 0);
///
/// // The offsets let you recover the source position.
/// assert_eq!(&text[slab.start..slab.end], "world");
/// ```
///
/// Character offsets (`char_start`/`char_end`) are automatically populated
/// when using [`SlabSource::slabs`](crate::SlabSource::slabs) or the
/// range constructors. They count Unicode
/// scalar values (`char`s), useful for NLP systems that index by character
/// position. They are `None` when constructing with [`Slab::new`] or returning
/// byte-only spans from [`SlabSource::slab_bytes`](crate::SlabSource::slab_bytes).
///
/// ## Overlap Handling
///
/// When spans overlap, adjacent slabs share some text. The `index` field
/// identifies each slab's position in the sequence:
///
/// ```text
/// Original: "The quick brown fox"
/// Slab 0:   "The quick b"     [0..11]
/// Slab 1:   "ck brown fox"    [8..19]  <- overlaps with slab 0
///                ^
///            overlap region [8..11]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Slab {
    /// The span text.
    pub text: String,
    /// Byte offset where this span starts in the source string.
    pub start: usize,
    /// Byte offset where this span ends (exclusive) in the source string.
    pub end: usize,
    /// Character offset where this span starts (Unicode scalar values).
    /// `None` until [`with_char_offsets`](Slab::with_char_offsets) or
    /// [`compute_char_offsets`] is called.
    pub char_start: Option<usize>,
    /// Character offset where this span ends (exclusive, Unicode scalar values).
    pub char_end: Option<usize>,
    /// Zero-based index of this span in the sequence.
    pub index: usize,
}

impl Slab {
    /// Create a new slab (byte offsets only; char offsets unset).
    #[must_use]
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        debug_assert!(
            start <= end,
            "Slab start ({start}) must not exceed end ({end})"
        );
        Self {
            text: text.into(),
            start,
            end,
            char_start: None,
            char_end: None,
            index,
        }
    }

    /// Create a slab from a byte range in the source text.
    ///
    /// The range must be within the source and both endpoints must be UTF-8
    /// character boundaries. Character offsets are computed automatically.
    pub fn from_byte_range(source: &str, range: Range<usize>, index: usize) -> Result<Self> {
        validate_byte_range(source, range.clone())?;

        let char_start = byte_to_char_offset(source, range.start);
        let char_end = byte_to_char_offset(source, range.end);
        Ok(Self {
            text: source[range.clone()].to_string(),
            start: range.start,
            end: range.end,
            char_start: Some(char_start),
            char_end: Some(char_end),
            index,
        })
    }

    /// Create a slab from a character range in the source text.
    ///
    /// Character offsets count Unicode scalar values. The returned slab stores
    /// both byte and character offsets.
    pub fn from_char_range(source: &str, range: Range<usize>, index: usize) -> Result<Self> {
        let char_len = source.chars().count();
        if range.start > range.end || range.end > char_len {
            return Err(Error::InvalidCharSpan {
                start: range.start,
                end: range.end,
                len: char_len,
            });
        }

        let start = char_to_byte_offset(source, range.start);
        let end = char_to_byte_offset(source, range.end);
        Ok(Self {
            text: source[start..end].to_string(),
            start,
            end,
            char_start: Some(range.start),
            char_end: Some(range.end),
            index,
        })
    }

    /// Set character offsets on this slab.
    #[must_use]
    pub fn with_char_offsets(mut self, char_start: usize, char_end: usize) -> Self {
        self.char_start = Some(char_start);
        self.char_end = Some(char_end);
        self
    }

    /// The length of this span in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// The length of this span in characters (Unicode scalar values).
    #[must_use]
    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }

    /// Whether this span is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The byte span in the source string.
    #[must_use]
    pub fn span(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }

    /// The character span, if computed.
    #[must_use]
    pub fn char_span(&self) -> Option<std::ops::Range<usize>> {
        match (self.char_start, self.char_end) {
            (Some(s), Some(e)) => Some(s..e),
            _ => None,
        }
    }
}

/// Create slabs from byte ranges in the source text.
pub fn slabs_from_byte_ranges(source: &str, ranges: &[Range<usize>]) -> Result<Vec<Slab>> {
    ranges
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, range)| Slab::from_byte_range(source, range, index))
        .collect()
}

/// Create slabs from character ranges in the source text.
pub fn slabs_from_char_ranges(source: &str, ranges: &[Range<usize>]) -> Result<Vec<Slab>> {
    ranges
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, range)| Slab::from_char_range(source, range, index))
        .collect()
}

fn validate_byte_range(source: &str, range: Range<usize>) -> Result<()> {
    if range.start > range.end || range.end > source.len() {
        return Err(Error::InvalidByteSpan {
            start: range.start,
            end: range.end,
            len: source.len(),
        });
    }

    if !source.is_char_boundary(range.start) {
        return Err(Error::NonCharBoundary {
            offset: range.start,
        });
    }
    if !source.is_char_boundary(range.end) {
        return Err(Error::NonCharBoundary { offset: range.end });
    }

    Ok(())
}

fn byte_to_char_offset(source: &str, byte_offset: usize) -> usize {
    source[..byte_offset].chars().count()
}

fn char_to_byte_offset(source: &str, char_offset: usize) -> usize {
    source
        .char_indices()
        .nth(char_offset)
        .map(|(byte_offset, _)| byte_offset)
        .unwrap_or(source.len())
}

/// Compute character offsets for a batch of slabs from the same document.
///
/// Builds a byte-to-char mapping in a single O(n) pass over the source text,
/// then fills `char_start`/`char_end` on each slab. This is faster than
/// per-slab computation when there are many slabs.
///
/// # Example
///
/// ```rust
/// use slabs::{compute_char_offsets, Slab};
///
/// let text = "Hello 日本語 world";
/// let mut slabs = vec![
///     Slab::new("Hello ", 0, 6, 0),
///     Slab::new("日本語", 6, 15, 1),
/// ];
/// compute_char_offsets(text, &mut slabs);
///
/// assert_eq!(slabs[0].char_start, Some(0));
/// assert_eq!(slabs[1].char_start, Some(6));
/// assert_eq!(slabs[1].char_end, Some(9));
/// ```
pub fn compute_char_offsets(text: &str, slabs: &mut [Slab]) {
    if slabs.is_empty() {
        return;
    }

    // Build byte->char index in one pass.
    // byte_to_char[byte_offset] = char_offset for each char boundary.
    // For non-boundary bytes, the value is undefined (we only look up boundaries).
    let mut byte_to_char = vec![0usize; text.len() + 1];
    for (char_idx, (byte_idx, _)) in text.char_indices().enumerate() {
        byte_to_char[byte_idx] = char_idx;
    }
    // Sentinel: byte offset == text.len() maps to total char count.
    byte_to_char[text.len()] = text.chars().count();

    for slab in slabs.iter_mut() {
        slab.char_start = Some(byte_to_char[slab.start]);
        slab.char_end = Some(byte_to_char[slab.end]);
    }
}

impl std::fmt::Display for Slab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let (Some(cs), Some(ce)) = (self.char_start, self.char_end) {
            write!(
                f,
                "Slab {{ index: {}, bytes: {}..{}, chars: {}..{}, len: {} }}",
                self.index,
                self.start,
                self.end,
                cs,
                ce,
                self.len()
            )
        } else {
            write!(
                f,
                "Slab {{ index: {}, span: {}..{}, len: {} }}",
                self.index,
                self.start,
                self.end,
                self.len()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_byte_range_sets_character_offsets() {
        let text = "Hello 日本語 world";
        let slab = Slab::from_byte_range(text, 6..15, 0).unwrap();

        assert_eq!(slab.text, "日本語");
        assert_eq!(slab.span(), 6..15);
        assert_eq!(slab.char_span(), Some(6..9));
    }

    #[test]
    fn from_byte_range_rejects_non_character_boundary() {
        let err = Slab::from_byte_range("éclair", 1..3, 0).unwrap_err();

        assert!(matches!(err, Error::NonCharBoundary { offset: 1 }));
    }

    #[test]
    fn from_char_range_converts_to_byte_offsets() {
        let text = "Hello 日本語 world";
        let slab = Slab::from_char_range(text, 6..9, 0).unwrap();

        assert_eq!(slab.text, "日本語");
        assert_eq!(slab.span(), 6..15);
        assert_eq!(slab.char_span(), Some(6..9));
    }

    #[test]
    fn batch_helpers_assign_sequence_indices() {
        let text = "alpha beta gamma";
        let slabs = slabs_from_byte_ranges(text, &[0..5, 6..10, 11..16]).unwrap();

        assert_eq!(
            slabs.iter().map(|slab| slab.index).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(slabs[2].text, "gamma");
    }
}
