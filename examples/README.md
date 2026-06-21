# slabs examples

Examples for late chunk embedding pooling.

## Running

```sh
cargo run --example late_chunking
```

Use `cargo test --examples` to compile the examples.

## Task map

| Goal | Example | Features | What to inspect |
|---|---|---|---|
| Pool full-document token embeddings over spans | `late_chunking` | default | `text-splitter` chooses byte ranges; `Slab` records those ranges; `LateChunkingPooler` pools token vectors over them. |

## Reading path

Start with `late_chunking` when boundaries already exist and you need to pool
token-level embeddings from a full-document encoder. Boundary selection belongs
upstream. This crate keeps the spans and performs the pooling step.
