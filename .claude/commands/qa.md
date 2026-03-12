# /qa -- Quality audit of slabs

Run a comprehensive quality pass: build (all feature combos), lint, test, property tests, doc-tests, examples, and pre-publish gate. Produces a timestamped report in `.claude/reports/`.

## Execution strategy

- **Stop early on build failure**: if `cargo check --all-features` fails, the QA is blocked. Report and stop.
- **Capture exact output**: save command output to temp files so findings are reliable. Don't eyeball scrollback.
- **Read previous reports first**: comparison with prior runs catches regressions.
- **Full output**: read all diagnostic output. Do not truncate or pipe through head/tail.

## Report convention

Reports go in `.claude/reports/qa-YYYY-MM-DD.md` (gitignored). Append a `-suffix` for multiple same-day reports.

## Procedure

### 0. Read prior QA reports

Check for prior reports in order: `.claude/reports/`, `qa/reports/`, `.qa/reports/`, `.claude/` root (flat files like `audit-report.md`). Read the most recent found. If reports exist in old locations, move them to `.claude/reports/` with dated names before proceeding.

```bash
eza --sort=modified -r .claude/reports/qa-*.md qa/reports/qa-*.md 2>/dev/null | head -3
```

Read the most recent 1-2 reports if they exist. Note open issues to watch for.

### 1. Feature matrix build

slabs has 4 independent features (`cli`, `semantic`, `code`, default). All combos must compile:

```bash
cargo check 2>&1                           # default
cargo check --all-features 2>&1            # everything
cargo check --features semantic 2>&1       # semantic only
cargo check --features code 2>&1           # code only
cargo check --features cli 2>&1            # cli only
```

If any fail, stop and report. Everything else depends on compilation.

### 2. Format check

```bash
cargo fmt -- --check
```

If formatting is off, note which files. Don't auto-fix during QA -- just report.

### 3. Clippy (all feature combos)

```bash
cargo clippy --all-targets -- -D warnings 2>&1
cargo clippy --all-targets --all-features -- -D warnings 2>&1
```

Capture full output. Classify findings: correctness issues vs style nits.

### 4. Tests (default features)

```bash
cargo nextest run 2>&1
```

If nextest is not available, fall back to `cargo test`. Record: total tests, pass/fail count.

### 5. Property tests (extended)

Property tests are critical for slabs (chunking invariants: coverage, ordering, bounds, text-matches, overlap). Run with extra cases:

```bash
PROPTEST_CASES=500 cargo test --test property_tests 2>&1
```

If any property test fails, capture the seed and minimal failing case.

### 6. Feature-gated tests

```bash
cargo test --features code --test code_props 2>&1
cargo test --features semantic --test integration_innr 2>&1
```

These verify feature-gated chunking strategies work correctly.

### 7. Doc-tests

```bash
cargo test --doc 2>&1
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features 2>&1
```

Catches: broken doc examples, rustdoc warnings, missing doc attributes on public items.

### 8. Examples compile and run

```bash
cargo run --example 01_basic_chunking 2>&1
cargo run --example chunking_strategies 2>&1
```

Both must run without error and produce sensible output.

### 9. Benchmarks compile

```bash
cargo bench --no-run 2>&1
```

Don't run benchmarks (slow), but verify they compile. Flag deprecation warnings.

### 10. Domain-specific checks

#### 10a. Chunking invariants audit

For each chunker (Fixed, Sentence, Recursive), manually verify with a representative document:

```bash
cargo test -- --nocapture 2>&1 | head -100
```

Check that:
- Empty input -> empty output
- Single-char input -> single chunk
- Unicode text (CJK, Arabic, emoji) -> valid UTF-8 boundaries
- Overlap regions match the configured overlap

#### 10b. Slab offset consistency

Verify that `slab.text == &original[slab.start..slab.end]` holds for all chunkers. The property tests cover this, but manually spot-check with multibyte text.

#### 10c. Public API surface

Check `src/lib.rs` re-exports. Flag:
- Internal types leaking through public signatures
- Types that are `pub` but should be crate-private
- The `model` module should NOT be re-exported until it has concrete implementations

#### 10d. Unwrap census

```bash
rg 'unwrap\(\)' --type rust --glob 'src/**' -n
```

`unwrap()` in lib code (non-test) is a concern. Mutex lock unwraps are acceptable for poisoning. Others should be audited.

### 11. Pre-publish gate

#### 11a. Version coherence

- `Cargo.toml` version matches README quickstart `[dependencies]` block
- Git tags (if any) match the version
- `innr`, `textprep` dependency versions match their published versions

#### 11b. Dependency check

```bash
cargo tree --edges no-dev --prefix none --all-features 2>&1 | sort -u
```

Flag: yanked deps, path deps leaking into the published manifest, unpinned deps.

#### 11c. Publish dry-run

```bash
cargo publish --dry-run 2>&1
```

Must succeed. Check for: missing metadata, path deps, large file warnings.

### 12. Write the report

Save to `.claude/reports/qa-YYYY-MM-DD.md`. Structure:

1. **Test conditions**: date, commit SHA, rustc version, slabs version
2. **Feature matrix**: pass/fail for each feature combo
3. **Check results**: pass/fail for each check (fmt, clippy, tests, proptests, doc-tests)
4. **Domain findings**: chunking invariant issues, API surface concerns
5. **Pre-publish gate**: version coherence, dep check, dry-run result
6. **Bug table**: concrete issues found with file:line references
7. **Comparison with prior run**: regressions, improvements
8. **Actionable items**: specific things worth fixing, ordered by impact

## What this is NOT

- Not a performance benchmark (use `cargo bench` directly)
- Not an auto-fixer (run `cargo fmt` and `cargo clippy --fix` separately)
- Not an architectural review (that's `/arch-review`)

This answers: "is slabs healthy, correct, and ready to publish?"
