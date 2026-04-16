# slabs

[![crates.io](https://img.shields.io/crates/v/slabs.svg)](https://crates.io/crates/slabs)
[![Documentation](https://docs.rs/slabs/badge.svg)](https://docs.rs/slabs)
[![CI](https://github.com/arclabs561/slabs/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/slabs/actions/workflows/ci.yml)

Text and code chunking for RAG pipelines.

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
slabs = { version = "0.1.4", features = ["code"] }
```

## Code chunking (tree-sitter AST-aware)

Splits source files at function, class, and block boundaries rather than at character counts.
Supports Rust, Python, TypeScript/JavaScript, and Go.

```rust
use slabs::{Chunker, CodeChunker, CodeLanguage};

let chunker = CodeChunker::new(CodeLanguage::Rust, 1500, 0);
let slabs = chunker.chunk(source_code);

for slab in &slabs {
    println!("[{}..{}]:\n{}\n", slab.start, slab.end, slab.text);
}
```

Language can also be inferred from a file extension:

```rust
let lang = CodeLanguage::from_extension("py").unwrap();
let chunker = CodeChunker::new(lang, 1500, 0);
```

AST node types that are kept intact when they fit within `max_chunk_size`:

| Language   | Block types                                              |
|------------|----------------------------------------------------------|
| Rust       | `function_item`, `impl_item`, `struct_item`, `enum_item`, `trait_item`, `mod_item` |
| Python     | `function_definition`, `class_definition`               |
| TypeScript | `function_declaration`, `class_declaration`, `method_definition`, `interface_declaration`, `enum_declaration` |
| Go         | `function_declaration`, `method_declaration`, `type_declaration` |

Nodes larger than `max_chunk_size` are split recursively. Leaf nodes that cannot be
parsed (long string literals, embedded DSLs) fall back to recursive text splitting.

## Late chunking (Günther et al. 2024)

Traditional chunking embeds each chunk in isolation, so cross-chunk references ("He became
famous" loses the antecedent "Einstein") degrade retrieval. Late chunking embeds the full
document first, then pools token-level embeddings into per-chunk vectors.

```rust
use slabs::{LateChunker, LateChunkingPooler, SentenceChunker, Chunker};

let base = SentenceChunker::new(3);
let late = LateChunker::new(base, 384); // 384 = embedding dim

// Get chunk boundaries
let chunks = late.chunk(&document);

// Embed the full document to get token embeddings (shape: [n_tokens, dim])
let token_embeddings = your_model.embed_tokens(&document);

// Pool into contextualized chunk embeddings
let chunk_embeddings = late.pool(&token_embeddings, &chunks, document.len());
```

If you have exact token offsets from the tokenizer, use `pool_with_offsets` for precise
boundary mapping instead of the default linear approximation.

Typical recall improvement: +5–15% over independent chunk embeddings on documents with
cross-sentence references. Requires holding full-document token embeddings in memory.

## Semantic chunking (embedding-based)

Splits at topic boundaries detected by embedding similarity drops between adjacent sentences.

```rust
use slabs::{Chunker, SemanticChunker};

let chunker = SemanticChunker::new(0.5)?; // similarity threshold
let slabs = chunker.chunk(long_document);
```

Requires the `semantic` feature (`fastembed`, `innr`, `textprep` dependencies).

## Prose chunking strategies

| Strategy  | When to use                          | Complexity      |
|-----------|--------------------------------------|-----------------|
| Fixed     | Homogeneous content, baselines       | O(n)            |
| Sentence  | Prose, articles, documentation       | O(n)            |
| Recursive | General-purpose, mixed content       | O(n log n)      |
| Semantic  | Topic coherence (`semantic` feature) | O(n × d)        |

```rust
use slabs::{Chunker, RecursiveChunker, SentenceChunker, FixedChunker};

// Recursive: tries paragraphs → lines → sentences → words → chars
let chunker = RecursiveChunker::prose(500);
let slabs = chunker.chunk(text);

// Sentence: 3 sentences per chunk, Unicode segmentation (UAX #29)
let chunker = SentenceChunker::new(3);
let slabs = chunker.chunk(text);

// Fixed: 200 chars, 40-char overlap
let chunker = FixedChunker::new(200, 40);
let slabs = chunker.chunk(text);
```

All chunkers return `Vec<Slab>` with byte and Unicode character offsets populated.

## Features

| Feature    | What it enables                                          |
|------------|----------------------------------------------------------|
| `code`     | `CodeChunker` via tree-sitter (Rust, Python, TypeScript, Go) |
| `semantic` | `SemanticChunker` (requires `fastembed`, `innr`, `textprep`) |
| `cli`      | `slabs` CLI binary                                       |
