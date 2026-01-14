# slabs

Text chunking for RAG pipelines.

Dual-licensed under MIT or Apache-2.0.

## Strategies

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| Fixed | Homogeneous content, baselines | $O(n)$ |
| Sentence | Prose, articles | $O(n)$ |
| Recursive | General-purpose | $O(n \log n)$ |
| Semantic | Topic coherence (`semantic` feature) | $O(nd)$ |

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
