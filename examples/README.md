# Examples

## Quick Start

| Example | What It Teaches |
|---------|-----------------|
| `01_basic_chunking` | Minimal: chunk text by sentences |
| `chunking_strategies` | Compare fixed vs sentence vs recursive |

```sh
cargo run --example 01_basic_chunking
cargo run --example chunking_strategies
```

## Choosing a Strategy

```
Is your content prose (articles, docs)?
  └─> SentenceChunker

Is your content mixed (code + comments, markdown)?
  └─> RecursiveChunker  

Do you need topic-aware splits?
  └─> SemanticChunker (requires embedding model)

Just need a baseline?
  └─> FixedChunker
```

## Typical Settings

| Strategy | Chunk Size | Overlap | Use Case |
|----------|------------|---------|----------|
| Fixed | 500 chars | 50 chars | Logs, code |
| Sentence | 3-5 sentences | 0-1 sentence | Articles |
| Recursive | 500 chars | via separators | Mixed |
| Semantic | varies | topic-based | Research |
