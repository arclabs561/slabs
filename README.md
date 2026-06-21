# slabs

[![crates.io](https://img.shields.io/crates/v/slabs.svg)](https://crates.io/crates/slabs)
[![Documentation](https://docs.rs/slabs/badge.svg)](https://docs.rs/slabs)
[![CI](https://github.com/arclabs561/slabs/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/slabs/actions/workflows/ci.yml)

Retrieval spans and late pooling.

`slabs` provides `Slab`, a text span with byte and character offsets, plus
utilities for pooling token embeddings over those spans. Use it between
document extraction, annotation, embedding, and indexing.

- `Slab`: text plus byte and character offsets in the source string.
- `LateChunkingPooler`: pool full-document token embeddings into span vectors
  (Günther et al. 2024). Boundaries come from upstream code.

Dual-licensed under MIT or Apache-2.0.

## Install

```toml
[dependencies]
slabs = "0.3"
```

Features:

| Feature | What it enables |
|---|---|
| `serde` | `Serialize`/`Deserialize` on `Slab` for storage backends |

## Retrieval spans

A `Slab` is the unit that moves between extraction, annotation, embedding, and
indexing. It does not decide how text should be split; it records the chosen
span.

```rust
use slabs::Slab;

let document = "Ada designed the engine. She wrote notes.";
let slab = Slab::from_byte_range(document, 0..24, 0).unwrap();

assert_eq!(&document[slab.span()], slab.text);
assert_eq!(slab.char_span(), Some(0..24));
```

Boundary sources can be manual spans, `text-splitter`, `deformat` segments,
or `anno` RAG chunks.

Offsets are relative to the exact string used to construct the slab. If text is
normalized, extracted, or otherwise transformed before slab construction, the
offsets refer to that transformed string. Preserve a separate mapping when the
original document offsets are also required.

## Late pooling

Traditional chunking embeds chunks independently, so cross-chunk references
like "She wrote notes" lose the antecedent from an earlier sentence. Late
pooling embeds the full document first so every token attends to the rest of
the document, then pools token-level embeddings over each `Slab` span.

```rust
use slabs::{LateChunkingPooler, Slab};

let spans: Vec<Slab> = make_spans(&document);
let token_embeddings: Vec<Vec<f32>> = embed_full_document_tokens(&document);
let token_offsets: Vec<(usize, usize)> = tokenizer_offsets(&document);

let pooler = LateChunkingPooler::new(384);
let span_embeddings = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &spans);
```

Use `pool_with_offsets` when your tokenizer exposes exact byte offsets. Use
`pool_with_char_offsets` when it exposes character offsets and your `Slab`s
have `char_start`/`char_end`. Use `pool` only as an approximation when you
have token embeddings and document length but no offsets.

Each returned vector is the L2-normalized mean of the token vectors overlapping
the slab. Late pooling requires holding full-document token embeddings in memory
and a model whose context window covers the document.

## Scope

- Boundary finding is upstream. `slabs` records selected ranges and pools over
  them.
- Input text is already `&str`. Format conversion is upstream.
- Embedding generation is upstream. `LateChunkingPooler` consumes token vectors.
- Storage is downstream. Enable `serde` on `Slab` when spans need to cross a
  storage or service boundary.
- Cross-file code analysis is out of scope. A slab refers to one source string.

## Examples

See [examples/README.md](examples/README.md) for the runnable example map.

```sh
cargo run --example late_chunking
```

## Migrating from 0.2

Removed in 0.3:

- `CodeChunker`, `CodeLanguage`, `ChunkSizer`, and `ByteSizer`: use
  `text-splitter` for boundary finding, then construct `Slab`s from the
  resulting ranges.
- `code` feature: `slabs` no longer has tree-sitter dependencies.

Added in 0.3:

- `Slab::from_byte_range()` and `Slab::from_char_range()` constructors
- `slabs_from_byte_ranges()` and `slabs_from_char_ranges()` batch helpers
- `pool_with_char_offsets()` for tokenizers that report character spans

## Migrating from 0.1

Removed in 0.2:

- `FixedChunker`, `SentenceChunker`, `RecursiveChunker`, `SemanticChunker`:
  use [`text-splitter`](https://crates.io/crates/text-splitter)
- `LateChunker<C>` wrapper: use `LateChunkingPooler` directly with
  `Vec<Slab>` from any source
- `ChunkCapacity`: was unused by any constructor; gone
- `slabs` CLI binary: use the chunking library APIs directly

Added in 0.2:

- `serde` feature for `Slab` serialization
