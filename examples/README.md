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
| Pool full-document token embeddings into chunks | `late_chunking` | default | Fake token embeddings and tokenizer offsets are pooled over chunk byte spans, showing the core late-chunking operation. |

## Reading path

Start with `late_chunking` when chunk boundaries already exist and you need to
pool token-level embeddings from a full-document encoder. Use `text-splitter`,
parser output, extraction spans, or manual ranges to produce the `Slab`s before
pooling.
