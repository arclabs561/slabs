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
            Self::Rust => tree_sitter_rust::LANGUAGE,
            Self::Python => tree_sitter_python::LANGUAGE,
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
            Self::Go => tree_sitter_go::LANGUAGE,
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

    fn traverse(&self, node: Node, code: &str, chunks: &mut Vec<Slab>) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let len = end_byte - start_byte;
        let kind = node.kind();

        // If it's a block node and fits, take it.
        // Or if it's any node and fits, and is "significant enough"?
        // Simpler strategy:
        // 1. If it fits and is a block node, take it.
        // 2. If it doesn't fit, recurse.
        // 3. If it fits but is not a block node, it might be part of a sequence of statements.
        //    We need a buffer to accumulate small statements.

        // This is a recursive strategy similar to LangChain's RecursiveCharacterTextSplitter
        // but using AST structure.

        // For this first pass, let's implement a greedy recursive approach.

        if len <= self.max_chunk_size {
            // If we are at root, we might want to split if it's huge.
            // But if `node` fits, we accept it as a chunk IF it is a block or we are at a leaf.
            // Actually, if it fits, we should probably just take it?
            // But we might want to merge adjacent small nodes.

            // Let's defer accumulation to a higher level loop?
            // No, let's just collect ranges here and merge them later.

            chunks.push(Slab::new(
                &code[start_byte..end_byte],
                start_byte,
                end_byte,
                chunks.len(),
            ));
            return;
        }

        // If it's too big, traverse children.
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                self.traverse(cursor.node(), code, chunks);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        } else {
            // Leaf node too big. Split by lines or chars?
            // Fallback to simpler chunking for this segment.
            // For now, just emit it (violating max_size).
            chunks.push(Slab::new(
                &code[start_byte..end_byte],
                start_byte,
                end_byte,
                chunks.len(),
            ));
        }
    }
}

impl Chunker for CodeChunker {
    fn chunk(&self, text: &str) -> Vec<Slab> {
        // Init parser
        let mut parser = Parser::new();
        if let Err(_) = parser.set_language(&self.language.get_language()) {
            return vec![]; // Should handle error better? Trait returns Vec<Slab>
        }

        let tree = match parser.parse(text, None) {
            Some(t) => t,
            None => return vec![],
        };

        let root = tree.root_node();
        let mut raw_chunks = Vec::new();

        // Use a modified traversal that accumulates
        let mut accumulator: Vec<Node> = Vec::new();
        let mut current_size = 0;

        // We need a custom walker that flattens the tree into a stream of "atomic" nodes
        // (leafs or small blocks)

        self.traverse(root, text, &mut raw_chunks);

        // Basic consolidation
        // The recursive traversal above produces a mix of small and large chunks.
        // We should really be doing the accumulation logic *inside* the traversal or after.
        // The current `traverse` implementation just recursively splits until things fit.
        // This is actually decent for a start. It will break a file into functions.
        // But it won't merge small functions together.

        // Let's implement a merging pass.
        let mut merged_chunks = Vec::new();
        let mut current_chunk_start = 0;
        let mut current_chunk_end = 0;
        let mut current_chunk_text = String::new();

        // Sort raw chunks by start position just in case
        raw_chunks.sort_by_key(|c| c.start);

        for chunk in raw_chunks {
            // Check if adding this chunk exceeds max_size
            if !current_chunk_text.is_empty()
                && current_chunk_text.len() + chunk.len() > self.max_chunk_size
            {
                // Emit current
                merged_chunks.push(Slab::new(
                    current_chunk_text.clone(),
                    current_chunk_start,
                    current_chunk_end,
                    merged_chunks.len(),
                ));
                current_chunk_text.clear();
            }

            if current_chunk_text.is_empty() {
                current_chunk_start = chunk.start;
            }

            // If there's a gap between current_chunk_end and chunk.start (whitespace/comments),
            // we should probably include it if we are merging?
            // But `traverse` above might skip gaps if we only visit children.
            // Actually, `traverse` as written above covers the whole range if we assume children cover the whole range.
            // But children don't always cover the whole range (whitespace between them).

            // Better strategy: Use the byte range.
            if !current_chunk_text.is_empty() {
                let gap = &text[current_chunk_end..chunk.start];
                current_chunk_text.push_str(gap);
            }

            current_chunk_text.push_str(&chunk.text);
            current_chunk_end = chunk.end;
        }

        if !current_chunk_text.is_empty() {
            merged_chunks.push(Slab::new(
                current_chunk_text,
                current_chunk_start,
                current_chunk_end,
                merged_chunks.len(),
            ));
        }

        merged_chunks
    }
}
