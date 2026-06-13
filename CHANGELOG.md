# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-19

### Changed

- Narrowed scope to code chunking and late chunking.
- Rewrote README to lead with code and late-chunking differentiators.

### Added

- `ChunkSizer` and `with_imports` for code chunking configuration.

## [0.1.4] - 2026-04-06

### Added

- Character offsets and sentence overlap for chunks.
- Optional CLI with `--json` output.
- `ChunkCapacity` for flexible chunk sizing.
- Late chunking strategy and feature-flag documentation.
- Integration tests and coverage tests for chunking strategies.

### Changed

- Made character offsets automatic via the `Chunker` trait.
- Capped `unicode-segmentation` below 1.13 for MSRV 1.81 compatibility.
- Raised MSRV to 1.81.

### Fixed

- Resolved clippy `manual_let_else` and redundant-pattern warnings.

### Removed

- Dead `CodeChunkerError` variants and unused exports.

[0.2.0]: https://github.com/arclabs561/slabs/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/arclabs561/slabs/releases/tag/v0.1.4
