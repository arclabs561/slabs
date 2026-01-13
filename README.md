# slabs

Text chunking for RAG pipelines.

## Strategies

| Strategy | Use Case | Speed |
|----------|----------|-------|
| Fixed | Homogeneous content, baselines | O(n) |
| Sentence | Prose, articles | O(n) |
| Recursive | General-purpose | O(n log n) |
| Semantic | Topic coherence (requires `semantic` feature) | O(n d) |

## Example

```rust
use slabs::{Chunker, RecursiveChunker};

let chunker = RecursiveChunker::prose(500);
let text = "Your long document here...";
let slabs = chunker.chunk(text);

for slab in slabs {
    println!("[{}..{}]: {}", slab.start, slab.end, slab.text);
}
```

## License

MIT OR Apache-2.0
