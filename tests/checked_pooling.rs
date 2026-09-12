use proptest::prelude::*;
use slabs::{PoolingError, Slab, SpanPooler};

fn naive_pool(
    embeddings: &[Vec<f32>],
    offsets: &[(usize, usize)],
    span: std::ops::Range<usize>,
) -> Vec<f32> {
    let selected: Vec<_> = offsets
        .iter()
        .zip(embeddings)
        .filter(|((start, end), _)| *start < span.end && *end > span.start)
        .map(|(_, embedding)| embedding)
        .collect();
    let mut sum = vec![0.0; embeddings[0].len()];
    for embedding in &selected {
        for (total, value) in sum.iter_mut().zip(embedding.iter()) {
            *total += value;
        }
    }
    for value in &mut sum {
        *value /= selected.len() as f32;
    }
    let norm = sum.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for value in &mut sum {
            *value /= norm;
        }
    }
    sum
}

proptest! {
    #[test]
    fn checked_byte_pooling_matches_independent_oracle(
        widths in prop::collection::vec(1usize..8, 1..80),
        gaps in prop::collection::vec(0usize..4, 1..80),
        raw_ranges in prop::collection::vec((0usize..80, 1usize..81), 1..30),
    ) {
        let count = widths.len().min(gaps.len());
        let mut cursor = 0usize;
        let mut offsets = Vec::with_capacity(count);
        for (&width, &gap) in widths.iter().zip(&gaps).take(count) {
            cursor += gap;
            offsets.push((cursor, cursor + width));
            cursor += width;
        }
        let embeddings: Vec<_> = (0..count)
            .map(|index| vec![index as f32 + 1.0, (index % 7) as f32 + 0.5, 1.0])
            .collect();

        let mut ranges: Vec<_> = raw_ranges
            .into_iter()
            .map(|(a, b)| {
                let left = a.min(count - 1);
                let right = b.min(count).max(left + 1);
                offsets[left].0..offsets[right - 1].1
            })
            .collect();
        ranges.sort_by_key(|range| range.start);
        let slabs: Vec<_> = ranges
            .iter()
            .enumerate()
            .map(|(index, range)| Slab::new("", range.start, range.end, index))
            .collect();

        let actual = SpanPooler::new(3)
            .try_pool_with_offsets(&embeddings, &offsets, &slabs)
            .unwrap();
        let expected: Vec<_> = ranges
            .into_iter()
            .map(|range| naive_pool(&embeddings, &offsets, range))
            .collect();
        prop_assert_eq!(actual, expected);
    }
}

#[test]
fn checked_character_pooling_handles_unicode_and_boundary_touch() {
    let source = "café 東京 🚀";
    let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let offsets = vec![(0, 4), (5, 7), (8, 9)];
    let slabs = vec![Slab::from_char_range(source, 0..5, 0).unwrap()];

    let actual = SpanPooler::new(2)
        .try_pool_with_char_offsets(&embeddings, &offsets, &slabs)
        .unwrap();

    assert_eq!(actual, vec![vec![1.0, 0.0]]);
}

#[test]
fn checked_pooling_ignores_empty_special_token_offsets() {
    let embeddings = vec![vec![9.0, 9.0], vec![1.0, 0.0], vec![9.0, 9.0]];
    let offsets = vec![(0, 0), (0, 4), (0, 0)];
    let slabs = vec![Slab::new("text", 0, 4, 0)];

    let actual = SpanPooler::new(2)
        .try_pool_with_offsets(&embeddings, &offsets, &slabs)
        .unwrap();

    assert_eq!(actual, vec![vec![1.0, 0.0]]);
}

#[test]
fn checked_pooling_distinguishes_empty_batch_from_invalid_input() {
    let pooler = SpanPooler::new(2);

    let empty = pooler
        .try_pool_with_offsets(&[], &[], &[])
        .expect("an empty batch is valid");
    assert!(empty.is_empty());

    assert!(matches!(
        pooler.try_pool_with_offsets(&[vec![1.0, 0.0]], &[], &[]),
        Err(PoolingError::TokenCountMismatch { .. })
    ));
}

#[test]
fn checked_pooling_rejects_malformed_contracts() {
    let pooler = SpanPooler::new(2);
    let slab = Slab::new("a", 0, 1, 0);

    assert!(matches!(
        pooler.try_pool_with_offsets(&[vec![1.0, 0.0]], &[], std::slice::from_ref(&slab)),
        Err(PoolingError::TokenCountMismatch { .. })
    ));
    assert!(matches!(
        pooler.try_pool_with_offsets(&[vec![1.0]], &[(0, 1)], std::slice::from_ref(&slab)),
        Err(PoolingError::EmbeddingDimension { .. })
    ));
    assert!(matches!(
        pooler.try_pool_with_offsets(
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
            &[(0, 2), (1, 3)],
            std::slice::from_ref(&slab)
        ),
        Err(PoolingError::InvalidTokenOffset { .. })
    ));
    assert!(matches!(
        pooler.try_pool_with_offsets(&[vec![1.0, 0.0]], &[(2, 3)], std::slice::from_ref(&slab)),
        Err(PoolingError::NoTokenOverlap { .. })
    ));
    assert!(matches!(
        pooler.try_pool_with_char_offsets(&[vec![1.0, 0.0]], &[(0, 1)], &[slab]),
        Err(PoolingError::MissingCharacterOffsets { .. })
    ));
    assert!(matches!(
        SpanPooler::new(0).try_pool_with_offsets(&[vec![]], &[(0, 1)], &[]),
        Err(PoolingError::ZeroDimension)
    ));
}

#[test]
fn checked_pooling_rejects_unsorted_slabs() {
    let pooler = SpanPooler::new(2);
    let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let offsets = vec![(0, 1), (2, 3)];
    let slabs = vec![Slab::new("b", 2, 3, 0), Slab::new("a", 0, 1, 1)];

    assert!(matches!(
        pooler.try_pool_with_offsets(&embeddings, &offsets, &slabs),
        Err(PoolingError::UnsortedSlabs { .. })
    ));

    // The original permissive API retains its per-slab behavior.
    assert_eq!(
        pooler.pool_with_offsets(&embeddings, &offsets, &slabs),
        vec![vec![0.0, 1.0], vec![1.0, 0.0]]
    );
}
