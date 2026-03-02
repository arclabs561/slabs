//! The Slab type: a chunk of text with position metadata.

/// A chunk of text with its position in the original document.
///
/// The name "slab" evokes a physical slice of material—concrete, wood, stone.
/// Each slab is a self-contained piece that can be embedded, indexed, and
/// retrieved independently.
///
/// ## Offsets
///
/// Primary offsets (`start`/`end`) are byte offsets into the original text,
/// matching Rust's string slicing semantics:
///
/// ```rust
/// use slabs::Slab;
///
/// let text = "Hello, world!";
/// let slab = Slab::new("world", 7, 12, 0);
///
/// // The offsets let you recover the original position
/// assert_eq!(&text[slab.start..slab.end], "world");
/// ```
///
/// Character offsets (`char_start`/`char_end`) are available after calling
/// [`Slab::with_char_offsets`] or [`compute_char_offsets`]. These count
/// Unicode scalar values (`char`s), useful for NLP systems that index
/// by character position.
///
/// ## Overlap Handling
///
/// When chunks overlap, adjacent slabs share some text. The `index` field
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
pub struct Slab {
    /// The chunk text.
    pub text: String,
    /// Byte offset where this chunk starts in the original document.
    pub start: usize,
    /// Byte offset where this chunk ends (exclusive) in the original document.
    pub end: usize,
    /// Character offset where this chunk starts (Unicode scalar values).
    /// `None` until [`with_char_offsets`](Slab::with_char_offsets) or
    /// [`compute_char_offsets`] is called.
    pub char_start: Option<usize>,
    /// Character offset where this chunk ends (exclusive, Unicode scalar values).
    pub char_end: Option<usize>,
    /// Zero-based index of this chunk in the sequence.
    pub index: usize,
}

impl Slab {
    /// Create a new slab (byte offsets only; char offsets unset).
    #[must_use]
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            char_start: None,
            char_end: None,
            index,
        }
    }

    /// Set character offsets on this slab.
    #[must_use]
    pub fn with_char_offsets(mut self, char_start: usize, char_end: usize) -> Self {
        self.char_start = Some(char_start);
        self.char_end = Some(char_end);
        self
    }

    /// The length of this chunk in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// The length of this chunk in characters (Unicode scalar values).
    #[must_use]
    pub fn char_len(&self) -> usize {
        self.text.chars().count()
    }

    /// Whether this chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The byte span of this chunk in the original document.
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

/// Compute character offsets for a batch of slabs from the same document.
///
/// Builds a byte-to-char mapping in a single O(n) pass over the source text,
/// then fills `char_start`/`char_end` on each slab. This is faster than
/// per-slab computation when there are many slabs.
///
/// # Example
///
/// ```rust
/// use slabs::{Chunker, FixedChunker, compute_char_offsets};
///
/// let text = "Hello 日本語 world";
/// let chunker = FixedChunker::new(8, 2);
/// let mut slabs = chunker.chunk(text);
/// compute_char_offsets(text, &mut slabs);
///
/// for slab in &slabs {
///     assert!(slab.char_start.is_some());
///     assert!(slab.char_end.is_some());
/// }
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
