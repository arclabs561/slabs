use crate::{Chunker, Slab};
use thiserror::Error;
use tree_sitter::{Language, Node, Parser};

/// Supported programming languages for code chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    /// Rust
    Rust,
    /// Python
    Python,
    /// TypeScript/JavaScript
    TypeScript,
    /// Go
    Go,
}

impl CodeLanguage {
    /// Get the tree-sitter language for this code language.
    pub fn get_language(&self) -> Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
        }
    }

    /// Guess language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "ts" | "tsx" | "js" | "jsx" => Some(Self::TypeScript),
            "go" => Some(Self::Go),
            _ => None,
        }
    }

    /// Check if a node type represents a cohesive block (function, class, etc.).
    pub fn is_block_node(&self, kind: &str) -> bool {
        match self {
            Self::Rust => matches!(
                kind,
                "function_item"
                    | "impl_item"
                    | "mod_item"
                    | "struct_item"
                    | "enum_item"
                    | "trait_item"
            ),
            Self::Python => matches!(kind, "function_definition" | "class_definition"),
            Self::TypeScript => matches!(
                kind,
                "function_declaration"
                    | "class_declaration"
                    | "method_definition"
                    | "interface_declaration"
                    | "enum_declaration"
            ),
            Self::Go => matches!(
                kind,
                "function_declaration" | "method_declaration" | "type_declaration"
            ),
        }
    }
}

/// Errors that can occur during code chunking.
#[derive(Debug, Error)]
pub enum CodeChunkerError {
    #[error("Tree-sitter language error: {0}")]
    LanguageError(#[from] tree_sitter::LanguageError),
    #[error("Failed to parse code")]
    ParseError,
}

/// A chunker that respects code structure using tree-sitter.
///
/// It attempts to keep functions, classes, and other code blocks intact.
pub struct CodeChunker {
    language: CodeLanguage,
    max_chunk_size: usize,
    chunk_overlap: usize,
}

impl CodeChunker {
    /// Create a new code chunker.
    pub fn new(language: CodeLanguage, max_chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            language,
            max_chunk_size,
            chunk_overlap,
        }
    }

    fn collect_leafs(&self, node: Node, code: &str, chunks: &mut Vec<Slab>) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let len = end_byte - start_byte;

        // If the node fits, we take it as a unit.
        // If it's a block node, we definitely want to try to keep it together.
        if len <= self.max_chunk_size {
            chunks.push(Slab::new(
                &code[start_byte..end_byte],
                start_byte,
                end_byte,
                0, // Index fixed later
            ));
            return;
        }

        // If it's too big, we MUST split it.
        // We iterate children to break it down.
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            let mut last_end = start_byte;

            loop {
                let child = cursor.node();
                let child_start = child.start_byte();

                // Gap before child
                if child_start > last_end {
                    let gap_text = &code[last_end..child_start];
                    if !gap_text.trim().is_empty() {
                        chunks.push(Slab::new(gap_text, last_end, child_start, 0));
                    }
                }

                // Process child
                self.collect_leafs(child, code, chunks);
                last_end = child.end_byte();

                if !cursor.goto_next_sibling() {
                    break;
                }
            }

            // Gap after last child
            if last_end < end_byte {
                let gap_text = &code[last_end..end_byte];
                if !gap_text.trim().is_empty() {
                    chunks.push(Slab::new(gap_text, last_end, end_byte, 0));
                }
            }
        } else {
            // Leaf node too big. Fall back to recursive text chunking.
            // This handles long string literals or comments.
            let leaf_text = &code[start_byte..end_byte];
            let recursive = crate::RecursiveChunker::new(
                self.max_chunk_size,
                &["\n\n", "\n", " ", ""], // Standard separators
            )
            .with_overlap(0); // No internal overlap for atomic parts (handled by merger)

            let sub_chunks = recursive.chunk(leaf_text);

            for sub in sub_chunks {
                // Adjust offsets relative to original code
                chunks.push(Slab::new(
                    sub.text,
                    start_byte + sub.start,
                    start_byte + sub.end,
                    0,
                ));
            }
        }
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        let mut parser = Parser::new();
        if parser.set_language(&self.language.get_language()).is_err() {
            return vec![];
        }

        let Some(tree) = parser.parse(text, None) else {
            return vec![];
        };

        let root = tree.root_node();
        let mut atomic_chunks = Vec::new();

        // 1. Decompose into atomic chunks (leaves or small blocks)
        self.collect_leafs(root, text, &mut atomic_chunks);

        // 2. Merge atomic chunks into maximal slabs
        let mut slabs = Vec::new();
        let mut current_text = String::new();
        let mut current_start = if atomic_chunks.is_empty() {
            0
        } else {
            atomic_chunks[0].start
        };
        let mut current_end = current_start;

        // Ensure atomic chunks are sorted
        atomic_chunks.sort_by_key(|c| c.start);

        for (i, chunk) in atomic_chunks.iter().enumerate() {
            // Calculate potential gap between current end and next chunk start
            // (collect_leafs should cover gaps, but just in case)
            let gap = if chunk.start > current_end {
                &text[current_end..chunk.start]
            } else {
                ""
            };

            let added_len = gap.len() + chunk.len();

            if !current_text.is_empty() && current_text.len() + added_len > self.max_chunk_size {
                // Emit current slab
                slabs.push(Slab::new(
                    current_text.clone(),
                    current_start,
                    current_end,
                    slabs.len(),
                ));
                current_text.clear();

                // Overlap Logic
                if self.chunk_overlap > 0 {
                    let mut overlap_size = 0;
                    let mut overlap_chunks = Vec::new();

                    // Walk backwards to find chunks that fit in overlap
                    for j in (0..i).rev() {
                        let prev_chunk = &atomic_chunks[j];

                        // Calculate gap after this prev_chunk
                        // If it's the last one before current (j = i-1), gap is `gap` (current_end..chunk.start)
                        // Wait, `gap` is between `current_end` and `chunk.start`.
                        // `current_end` aligns with `prev_chunk.end`.

                        let next_start = if j == i - 1 {
                            chunk.start
                        } else {
                            atomic_chunks[j + 1].start
                        };

                        let gap_len = next_start - prev_chunk.end;
                        let chunk_len = prev_chunk.len();

                        if overlap_size + chunk_len + gap_len > self.chunk_overlap {
                            if overlap_chunks.is_empty() {
                                overlap_chunks.push(j);
                            }
                            break;
                        }

                        overlap_chunks.push(j);
                        overlap_size += chunk_len + gap_len;
                    }

                    if !overlap_chunks.is_empty() {
                        overlap_chunks.reverse(); // Forward order
                        let first_idx = overlap_chunks[0];
                        let last_idx = *overlap_chunks.last().unwrap();

                        let first_chunk = &atomic_chunks[first_idx];
                        let last_chunk = &atomic_chunks[last_idx];

                        current_start = first_chunk.start;
                        // Include text up to end of last overlap chunk
                        // (Gaps between overlap chunks are included by slicing source text)
                        current_text = text[current_start..last_chunk.end].to_string();
                        current_end = last_chunk.end;
                    } else {
                        current_start = chunk.start;
                    }
                } else {
                    current_start = chunk.start;
                }
            }

            if current_text.is_empty() {
                current_start = chunk.start;
            } else {
                current_text.push_str(gap);
            }

            current_text.push_str(&chunk.text);
            current_end = chunk.end;
        }

        // Flush last chunk
        if !current_text.is_empty() {
            slabs.push(Slab::new(
                current_text,
                current_start,
                current_end,
                slabs.len(),
            ));
        }

        slabs
    }
}
