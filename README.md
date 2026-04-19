# slabs

[![crates.io](https://img.shields.io/crates/v/slabs.svg)](https://crates.io/crates/slabs)
[![Documentation](https://docs.rs/slabs/badge.svg)](https://docs.rs/slabs)
[![CI](https://github.com/arclabs561/slabs/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/slabs/actions/workflows/ci.yml)

AST-aware code chunking and late chunking for RAG.

Two primitives:

- **`CodeChunker`** — split source code at function/class/impl boundaries via tree-sitter. Rust, Python, TypeScript/JavaScript, Go. Optional import-context injection. Pluggable size metric (bytes by default; bring your own tokenizer).
- **`LateChunkingPooler`** — pool full-document token embeddings into per-chunk vectors (Günther et al. 2024). Bring your own boundaries from any source.

Dual-licensed under MIT or Apache-2.0.

## Install

```toml
[dependencies]
slabs = { version = "0.2", features = ["code"] }
```

Features:

| Feature | What it enables |
|---|---|
| `code` | `CodeChunker` via tree-sitter (Rust, Python, TypeScript, Go) |
| `serde` | `Serialize`/`Deserialize` on `Slab` for storage backends |

## Code chunking

Splits source files at AST-defined boundaries — keeping functions, classes,
and impl blocks atomic when they fit the size budget. Oversize nodes are split
recursively at structural separators; unparseable leaves fall back to recursive
text splitting.

```rust
use slabs::{Chunker, CodeChunker, CodeLanguage};

let chunker = CodeChunker::new(CodeLanguage::Rust, 1500, 0);
let slabs = chunker.chunk(source_code);

for slab in &slabs {
    println!("[{}..{}]\n{}\n", slab.start, slab.end, slab.text);
}
```

Language can also be inferred from a file extension:

```rust
use slabs::{CodeChunker, CodeLanguage};
let lang = CodeLanguage::from_extension("py").unwrap();
let chunker = CodeChunker::new(lang, 1500, 0);
```

### Import-context injection

Method chunks lose the surrounding `use`/`import` statements that name the
types they reference. `with_imports(true)` walks the AST once, collects every
top-level import node, and prepends them to each chunk that doesn't already
contain them. Retrievers see imports next to call sites instead of stranded
at the file head.

```rust
use slabs::{CodeChunker, CodeLanguage};

let chunker = CodeChunker::new(CodeLanguage::Rust, 1500, 0)
    .with_imports(true);
let slabs = chunker.chunk(source_code);
```

Per-language import nodes:

| Language | Nodes treated as imports |
|---|---|
| Rust | `use_declaration`, `extern_crate_declaration` |
| Python | `import_statement`, `import_from_statement` |
| TypeScript | `import_statement` |
| Go | `import_declaration` |

### Pluggable size metric

`CodeChunker` sizes chunks in bytes by default. To target a model's token
context limit, plug in your tokenizer through the `ChunkSizer` trait:

```rust
use slabs::{ChunkSizer, CodeChunker, CodeLanguage};

struct TiktokenSizer { /* your tokenizer */ }

impl ChunkSizer for TiktokenSizer {
    fn size(&self, text: &str) -> usize {
        // count tokens using your tokenizer
        # 0
    }
}

let chunker = CodeChunker::new(CodeLanguage::Rust, 8000, 0)
    .with_sizer(TiktokenSizer { /* ... */ });
```

The `max_chunk_size` argument is interpreted in whatever unit the sizer
returns — bytes for the default `ByteSizer`, tokens for a tokenizer-backed
sizer.

### AST node types kept atomic

| Language   | Block types                                              |
|------------|----------------------------------------------------------|
| Rust       | `function_item`, `impl_item`, `struct_item`, `enum_item`, `trait_item`, `mod_item` |
| Python     | `function_definition`, `class_definition`               |
| TypeScript | `function_declaration`, `class_declaration`, `method_definition`, `interface_declaration`, `enum_declaration` |
| Go         | `function_declaration`, `method_declaration`, `type_declaration` |

## Late chunking

Traditional chunking embeds chunks independently, so cross-chunk references —
"He became famous" loses the antecedent "Einstein" — degrade retrieval. Late
chunking embeds the full document first so every token attends to the rest of
the document, then pools token-level embeddings into per-chunk vectors. The
result preserves document-wide context.

`LateChunkingPooler` is a primitive: it takes pre-computed token embeddings
plus chunk boundaries and returns pooled chunk embeddings. Bring your own
boundaries from any source.

```rust
use slabs::{LateChunkingPooler, Slab};

// 1. Chunk boundaries from any source — text-splitter, CodeChunker, regex, manual.
let chunks: Vec<Slab> = my_chunker(&document);

// 2. Embed the FULL document with a long-context model
//    (Jina v2/v3, nomic-embed-text, etc.) to get [n_tokens, dim] embeddings.
let token_embeddings: Vec<Vec<f32>> = my_model.embed_tokens(&document);

// 3. Pool token embeddings inside each chunk's byte span.
let pooler = LateChunkingPooler::new(384); // dim
let chunk_embeddings = pooler.pool(&token_embeddings, &chunks, document.len());
```

If you have exact token offsets from the tokenizer, use `pool_with_offsets`
for precise boundary mapping instead of the default linear approximation.

Late chunking requires holding full-document token embeddings in memory and a
model whose context window covers the document.

## What slabs does not do

- **General-purpose text chunking.** Use [`text-splitter`](https://crates.io/crates/text-splitter)
  (1.2M+ downloads) for fixed/sentence/recursive prose splitting. It has
  broader Unicode handling, token-count sizing, and is the de-facto Rust
  standard. Wrap its output in `Slab` if you want to feed it to
  `LateChunkingPooler`.
- **Format conversion (PDF, HTML, DOCX).** Input is `&str`. Use
  [`deformat`](https://crates.io/crates/deformat) or
  [`pdf-extract`](https://crates.io/crates/pdf-extract) upstream.
- **Embedding generation.** `LateChunkingPooler` consumes pre-computed token
  embeddings. Bring your own model.
- **Vector store integration.** `Slab` is the boundary; enable the `serde`
  feature and wire to qdrant-client, lancedb, sqlx, etc. yourself.
- **Cross-file analysis (LSP, type resolution, dependency graphs).** Slabs
  operates on one document at a time. See `tree-sitter-stack-graphs` and
  `ast-grep` for code-graph tools.

## Examples

```sh
cargo run --example code_chunking --features code
cargo run --example late_chunking
```

## Migrating from 0.1

Removed in 0.2:

- `FixedChunker`, `SentenceChunker`, `RecursiveChunker`, `SemanticChunker` →
  use [`text-splitter`](https://crates.io/crates/text-splitter)
- `LateChunker<C>` wrapper → use `LateChunkingPooler` directly with
  `Vec<Slab>` from any source
- `ChunkCapacity` → was unused by any constructor; gone
- `slabs` CLI binary → use the chunking library APIs directly

Added in 0.2:

- `ChunkSizer` trait + `ByteSizer` default; `CodeChunker::with_sizer()`
- `CodeChunker::with_imports(true)` for import-context injection
- `serde` feature for `Slab` serialization
