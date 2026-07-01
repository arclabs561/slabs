# slabs examples

Runnable examples for retrieval spans and span pooling.

## Running

```sh
cargo run --example span_pooling
```

Use `cargo test --examples` to compile the examples.

## `span_pooling`

Pools token embeddings over externally chosen spans. `text-splitter` chooses
byte ranges, `Slab` records those ranges, and `SpanPooler` pools token vectors
over exact offsets.

Output:

```text
span 0 [0..30]: "Einstein developed relativity."
  pooled [einstein, pronoun, theory, physics]: [0.987, 0.000, 0.073, 0.146]

span 1 [31..48]: "He became famous."
  pooled [einstein, pronoun, theory, physics]: [0.844, 0.533, 0.000, 0.067]

span 2 [49..80]: "The theory transformed physics."
  pooled [einstein, pronoun, theory, physics]: [0.517, 0.000, 0.682, 0.517]
```
