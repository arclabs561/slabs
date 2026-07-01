# slabs examples

Examples for span pooling over retrieval spans.

## Running

```sh
cargo run --example span_pooling
```

## Which example should I run?

| I want to... | Example | What to check |
|---|---|---|
| Pool token embeddings over externally chosen spans | `span_pooling` | `text-splitter` chooses byte ranges; `Slab` records those ranges; `SpanPooler` pools token vectors over exact offsets. |

Use `cargo test --examples` to compile the examples.

`span_pooling` is the intended boundary in one file: upstream code chooses
spans, `slabs` stores them, and span pooling turns token vectors into one vector
per span.

Expected excerpt:

```text
span 1 [31..48]: "He became famous."
  pooled [einstein, pronoun, theory, physics]: [0.844, 0.533, 0.000, 0.067]
```
