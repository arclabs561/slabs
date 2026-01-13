//! The Slab type: a chunk of text with position metadata.

/// A chunk of text with its position in the original document.
///
/// The name "slab" evokes a physical slice of material—concrete, wood, stone.
/// Each slab is a self-contained piece that can be embedded, indexed, and
/// retrieved independently.
///
/// ## Byte Offsets
///
/// `start` and `end` are byte offsets into the original text, not character
/// indices. This matches Rust's string slicing semantics:
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
    /// Zero-based index of this chunk in the sequence.
    pub index: usize,
}

impl Slab {
    /// Create a new slab.
    #[must_use]
    pub fn new(text: impl Into<String>, start: usize, end: usize, index: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
            index,
        }
    }

    /// The length of this chunk in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
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
}

impl std::fmt::Display for Slab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
