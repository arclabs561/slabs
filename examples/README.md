# slabs examples

Examples for late pooling over retrieval spans.

## Running

```sh
cargo run --example late_chunking
```

Use `cargo test --examples` to compile the examples.

## Example map

| Area | Example | Features | Check |
|---|---|---|---|
| Pool full-document token embeddings over spans | `late_chunking` | default | `text-splitter` chooses byte ranges; `Slab` records those ranges; `LateChunkingPooler` pools token vectors over them. |

## Reading order

Read `late_chunking` first. It shows the intended boundary: upstream code
chooses spans, `slabs` stores those spans, and late pooling turns token vectors
into one vector per span.
