use crate::sizer::{ByteSizer, ChunkSizer};
use crate::{Chunker, Slab};
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

    /// Check if a node type represents a top-level import.
    pub fn is_import_node(&self, kind: &str) -> bool {
        match self {
            Self::Rust => matches!(kind, "use_declaration" | "extern_crate_declaration"),
            Self::Python => matches!(kind, "import_statement" | "import_from_statement"),
            Self::TypeScript => matches!(kind, "import_statement"),
            Self::Go => matches!(kind, "import_declaration"),
        }
    }
}

/// A chunker that respects code structure using tree-sitter.
///
/// Functions, classes, and other AST blocks are kept intact when they fit
/// `max_chunk_size`; oversize nodes split recursively. The size unit
/// (bytes by default) is determined by the [`ChunkSizer`] — plug in a
/// tokenizer via [`with_sizer`](Self::with_sizer) to size in tokens.
/// Enable [`with_imports`](Self::with_imports) to prepend top-level
/// `use`/`import` statements to non-import chunks.
pub struct CodeChunker {
    language: CodeLanguage,
    max_chunk_size: usize,
    chunk_overlap: usize,
    sizer: Box<dyn ChunkSizer>,
    inject_imports: bool,
}

impl CodeChunker {
    /// Create a new code chunker.
    ///
    /// `max_chunk_size` is in bytes by default (via [`ByteSizer`]). To size
    /// chunks in tokens, attach a tokenizer-backed sizer with
    /// [`with_sizer`](Self::with_sizer).
    pub fn new(language: CodeLanguage, max_chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            language,
            max_chunk_size,
            chunk_overlap,
            sizer: Box::new(ByteSizer),
            inject_imports: false,
        }
    }

    /// Plug in a custom size metric (tokens, codepoints, etc.).
    ///
    /// `max_chunk_size` is then interpreted in whatever unit the sizer returns.
    #[must_use]
    pub fn with_sizer<S: ChunkSizer + 'static>(mut self, sizer: S) -> Self {
        self.sizer = Box::new(sizer);
        self
    }

    /// Prepend top-level `use`/`import` declarations to chunks that don't
    /// already contain them.
    ///
    /// Method-only chunks lose the surrounding imports that name the types
    /// they reference; this restores that context. Increases chunk size by
    /// the import block length on every non-import chunk — the resulting
    /// chunk may exceed `max_chunk_size`. The caller owns the budget
    /// tradeoff; widen `max_chunk_size` to accommodate imports if your
    /// embedding model has a strict context limit.
    ///
    /// When enabled, `slab.text` may contain prepended import text not
    /// present at the original `slab.start..slab.end` byte range.
    #[must_use]
    pub fn with_imports(mut self, inject: bool) -> Self {
        self.inject_imports = inject;
        self
    }

    /// Walk root children, collect import nodes, return (concatenated text, max end byte).
    fn collect_imports(&self, root: Node, code: &str) -> (String, usize) {
        let mut imports = String::new();
        let mut max_end = 0usize;
        let mut cursor = root.walk();
        if cursor.goto_first_child() {
            loop {
                let node = cursor.node();
                if self.language.is_import_node(node.kind()) {
                    let s = node.start_byte();
                    let e = node.end_byte();
                    imports.push_str(&code[s..e]);
                    imports.push('\n');
                    if e > max_end {
                        max_end = e;
                    }
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        (imports, max_end)
    }

    fn collect_leafs(&self, node: Node, code: &str, chunks: &mut Vec<Slab>) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let node_text = &code[start_byte..end_byte];

        // If the node fits the size budget, take it as a unit.
        // Block nodes (functions/classes) we especially want to keep together.
        if self.sizer.size(node_text) <= self.max_chunk_size {
            chunks.push(Slab::new(
                node_text, start_byte, end_byte, 0, // Index fixed later
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
            let recursive = crate::recursive::RecursiveChunker::new(
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
    fn chunk_bytes(&self, text: &str) -> Vec<Slab> {
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

            // Size check uses the sizer; for non-byte sizers we recompute
            // current_text's size each iteration (O(N*T) for token sizers,
            // acceptable for typical chunk sizes; tokenizer caching is the
            // user's job if needed).
            let projected = if current_text.is_empty() {
                0
            } else {
                self.sizer.size(&current_text) + self.sizer.size(gap) + self.sizer.size(&chunk.text)
            };

            if !current_text.is_empty() && projected > self.max_chunk_size {
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
                        let last_idx = overlap_chunks[overlap_chunks.len() - 1];

                        let first_chunk = &atomic_chunks[first_idx];
                        let last_chunk = &atomic_chunks[last_idx];

                        current_start = first_chunk.start;
                        // Include text up to end of last overlap chunk
                        // (Gaps between overlap chunks are included by slicing source text)
                        current_text = text[current_start..last_chunk.end].to_string();
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

        // 3. Optional: prepend top-level imports to chunks that don't already
        //    cover them. Injection may push a chunk past `max_chunk_size` —
        //    the caller opted in and owns the budget tradeoff.
        if self.inject_imports {
            let (imports, import_end) = self.collect_imports(root, text);
            if !imports.is_empty() {
                let header = format!("{}\n", imports.trim_end());
                for slab in slabs.iter_mut() {
                    if slab.start >= import_end {
                        slab.text = format!("{}{}", header, slab.text);
                    }
                }
            }
        }

        slabs
    }
}
