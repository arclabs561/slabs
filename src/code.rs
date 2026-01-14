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
            // Leaf node too big. Hard split?
            // For now, emit as one big chunk.
            chunks.push(Slab::new(
                &code[start_byte..end_byte],
                start_byte,
                end_byte,
                0,
            ));
        }
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        let mut parser = Parser::new();
        if let Err(_) = parser.set_language(&self.language.get_language()) {
            return vec![];
        }

        let tree = match parser.parse(text, None) {
            Some(t) => t,
            None => return vec![],
        };

        let root = tree.root_node();
        let mut atomic_chunks = Vec::new();

        // 1. Decompose into atomic chunks (leaves or small blocks)
        self.collect_leafs(root, text, &mut atomic_chunks);

        // 2. Merge atomic chunks into maximal slabs
        let mut slabs = Vec::new();
        let mut current_text = String::new();
        let mut current_start = if !atomic_chunks.is_empty() {
            atomic_chunks[0].start
        } else {
            0
        };
        let mut current_end = current_start;

        // Ensure atomic chunks are sorted
        atomic_chunks.sort_by_key(|c| c.start);

        for chunk in atomic_chunks {
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
                current_start = chunk.start;
                // If the gap was significant, we might have skipped it.
                // But generally gaps stick to the preceding or following chunk.
                // In this simple merge, we restart at chunk.start.
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
