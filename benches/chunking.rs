//! Benchmarks for text chunking strategies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker};

fn sample_text(size: usize) -> String {
    // Generate realistic text with sentence structure
    let sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "Pack my box with five dozen liquor jugs. ",
        "How vexingly quick daft zebras jump! ",
        "The five boxing wizards jump quickly. ",
        "Sphinx of black quartz, judge my vow. ",
    ];
    let mut text = String::with_capacity(size);
    let mut i = 0;
    while text.len() < size {
        text.push_str(sentences[i % sentences.len()]);
        i += 1;
    }
    text.truncate(size);
    text
}

fn bench_fixed_chunker(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_chunker");

    for size in [1_000, 10_000, 100_000] {
        let text = sample_text(size);
        let chunker = FixedChunker::new(500, 50);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("fixed", size), &text, |b, text| {
            b.iter(|| chunker.chunk(black_box(text)))
        });
    }

    group.finish();
}

fn bench_sentence_chunker(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_chunker");

    for size in [1_000, 10_000, 100_000] {
        let text = sample_text(size);
        let chunker = SentenceChunker::new(3); // 3 sentences per chunk

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("sentence", size), &text, |b, text| {
            b.iter(|| chunker.chunk(black_box(text)))
        });
    }

    group.finish();
}

fn bench_recursive_chunker(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_chunker");

    // Default separators: paragraphs -> sentences -> words
    let separators = &["\n\n", "\n", ". ", " "];

    for size in [1_000, 10_000, 100_000] {
        let text = sample_text(size);
        let chunker = RecursiveChunker::new(500, separators);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("recursive", size), &text, |b, text| {
            b.iter(|| chunker.chunk(black_box(text)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fixed_chunker,
    bench_sentence_chunker,
    bench_recursive_chunker
);
criterion_main!(benches);
