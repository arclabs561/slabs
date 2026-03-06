# /arch-review -- Architectural review of slabs

Audit the structural design: dependency graph, API surface, trait design, error handling, feature gates, and extensibility. Read-only -- does not modify code.

## Procedure

### 0. Read prior arch reports

```bash
eza --sort=modified -r .claude/reports/arch-*.md 2>/dev/null | head -3
```

Read the most recent report if it exists. Note tracked issues.

### 1. Module graph and layering

slabs has the following module structure:

```
lib.rs         (public API surface, Chunker trait)
├── slab.rs         (Slab type, compute_char_offsets)
├── capacity.rs     (ChunkCapacity config)
├── error.rs        (Error enum)
├── fixed.rs        (FixedChunker)
├── sentence.rs     (SentenceChunker)
├── recursive.rs    (RecursiveChunker)
├── late.rs         (LateChunker, LateChunkingPooler)
├── model.rs        (ModelChunker -- private, unfinished)
├── semantic.rs     [feature: semantic] (SemanticChunker)
└── code.rs         [feature: code] (CodeChunker)
```

Verify:
- No circular dependencies between modules
- `slab.rs` and `error.rs` are leaf modules (no upward deps)
- `code.rs` uses `recursive.rs` internally (acceptable: fallback for large leaf nodes)
- Feature-gated modules (`semantic`, `code`) don't leak into default API

### 2. Trait design: `Chunker`

The core trait:

```rust
pub trait Chunker: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<Slab>;
    fn estimate_chunks(&self, text_len: usize) -> usize;
}
```

Audit:
- **`&self` constraint**: verified correct? `SemanticChunker` was wrapped in Mutex to satisfy this. Check that this doesn't introduce unnecessary contention for non-concurrent use.
- **`Vec<Slab>` return**: allocation-heavy for streaming use cases. Is this the right trade-off for a chunking library? (answer: yes, chunking is batch-oriented)
- **`Send + Sync` bounds**: required for embedding in async pipelines. Verify all implementors satisfy this.
- **Object safety**: `dyn Chunker` must work. Verify no `Sized` or associated type issues.

### 3. Type system audit

#### 3a. Slab type

- `start`/`end` are byte offsets. `char_start`/`char_end` are optional character offsets.
- `Option<usize>` for char offsets is a valid design (not all users need them).
- Check: is there any path where `start > end`? Any path where `end > text.len()`?
- `compute_char_offsets` builds a byte-to-char mapping in O(n). Verify it handles edge cases (empty slabs, overlapping slabs, non-ASCII text).

#### 3b. ChunkCapacity

- `desired`/`max` split is the right abstraction.
- Check: is `ChunkCapacity` used by any chunker? (it's exported but may be unused -- potential dead API)

#### 3c. Error type

- `Error` enum has 4 variants. Check they're all reachable from public API.
- `Error::SemanticFeatureRequired` -- is this actually returned anywhere, or dead?

### 4. Feature gate audit

```bash
rg '#\[cfg\(feature' --type rust src/ -n
```

Verify:
- `semantic` module is gated by `#[cfg(feature = "semantic")]`
- `code` module is gated by `#[cfg(feature = "code")]`
- No default-feature code accidentally depends on optional deps
- Feature names in code match `Cargo.toml` `[features]` section exactly (previously caught: `feature = "innr"` without a matching feature)

### 5. Dependency analysis

```bash
cargo tree --all-features --depth 1 2>&1
```

For a chunking library, the dependency surface matters:
- **Default features**: only `thiserror` + `unicode-segmentation`. This is lean -- good.
- **Semantic**: pulls `fastembed` (heavy: ONNX runtime, model downloads). Verify this doesn't leak into default.
- **Code**: pulls `tree-sitter` + 4 language grammars. Verify these are truly optional.
- **innr, textprep**: ecosystem crates. Check versions match published versions.

Flag any dep that seems unnecessary or overly heavy for what it provides.

### 6. Error handling

```bash
rg 'unwrap\(\)|panic!\(|expect\(' --type rust src/ -n --glob '!src/bin/*'
```

In library code:
- `unwrap()` on Mutex locks: acceptable (panic on poison)
- `unwrap()` on Vec operations with guaranteed elements: audit case-by-case
- `panic!()` via `assert!()` in constructors: acceptable (invalid config)
- `expect()`: check the message is descriptive

### 7. Extensibility audit

How easy is it to add a new chunking strategy?

1. Implement `Chunker` trait (2 methods)
2. Add module to `lib.rs`
3. Re-export from `lib.rs`

This is clean. Check:
- Is there any shared state or global registration that makes new strategies harder?
- Can users implement `Chunker` externally? (yes, trait is public and object-safe)

### 8. API surface review

List all public items:

```bash
rg '^pub ' --type rust src/lib.rs -n
rg '^pub (fn|struct|enum|trait|type|use|mod)' --type rust src/ -n
```

For each public item, ask:
- Should this be public? Or is it an implementation detail?
- Is it documented?
- Is the name clear without context?

Known issue: `ModelChunker` was previously exported but removed (finding from prior audit). Verify it stays private.

### 9. Write the report

Save to `.claude/reports/arch-YYYY-MM-DD.md`. Structure:

1. **Module graph**: verified layering, violations found
2. **Trait design**: Chunker audit findings
3. **Type system**: Slab, ChunkCapacity, Error audit
4. **Feature gates**: correctness, dead feature refs
5. **Dependencies**: weight analysis, leakage check
6. **Error handling**: unwrap census, panic paths
7. **Extensibility**: friction points for new strategies
8. **API surface**: public items that shouldn't be, missing items that should be
9. **Actionable items**: ordered by impact

## What this is NOT

- Not a QA check (that's `/qa`)
- Not a performance analysis (use benchmarks)

This answers: "is the slabs architecture clean, extensible, and well-bounded?"
