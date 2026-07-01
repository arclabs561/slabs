//! Late-pooling math against hand-computed oracles.
//!
//! `slabs` owns two pieces of pooling logic worth pinning exactly:
//!
//! 1. `SpanPooler::pool` maps a slab's byte span `[start, end)` to the
//!    half-open token-index range
//!    `[floor(start / doc_len * n_tokens), floor(end / doc_len * n_tokens))`,
//!    mean-pools those token vectors, and L2-normalizes the result. The mapping
//!    uses truncating (floor) arithmetic, so fractional positions round down.
//! 2. `pool_with_offsets` selects tokens by half-open overlap with the slab's
//!    byte span (`token.start < slab.end && token.end > slab.start`), so a token
//!    that merely touches a boundary is excluded.
//!
//! Vectors are chosen so the mean and the L2 norm have closed forms.

use slabs::{Slab, SpanPooler};

const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1 / sqrt(2)

fn assert_vec_close(got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch: {got:?} vs {want:?}"
    );
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "value mismatch: {got:?} vs {want:?}");
    }
}

/// `pool` floors the linear byte->token mapping.
///
/// doc_len = 10, n_tokens = 5, so byte offset `b` maps to token `floor(b / 2)`.
/// A slab over bytes 3..7 maps to tokens `[floor(1.5), floor(3.5)) = [1, 3)`,
/// i.e. tokens 1 and 2. Their mean is `[0.5, 0.5]`, normalized to
/// `[1/sqrt(2), 1/sqrt(2)]`.
#[test]
fn pool_floors_byte_to_token_mapping() {
    let pooler = SpanPooler::new(2);
    let token_embeddings = vec![
        vec![1.0, 0.0], // t0
        vec![1.0, 0.0], // t1  <- selected
        vec![0.0, 1.0], // t2  <- selected
        vec![0.0, 1.0], // t3
        vec![1.0, 1.0], // t4
    ];
    let slab = Slab::new("span", 3, 7, 0);

    let pooled = pooler.pool(&token_embeddings, &[slab], 10);

    assert_eq!(pooled.len(), 1);
    // tokens [t1, t2] -> mean [0.5, 0.5] -> normalized [1/sqrt(2), 1/sqrt(2)].
    assert_vec_close(&pooled[0], &[SQRT_HALF, SQRT_HALF]);
}

/// `pool` maps an identity-length document 1:1 and selects exactly its span.
///
/// doc_len = 4, n_tokens = 4, so byte offset == token index. Two adjacent slabs
/// partition the four tokens into disjoint halves with no overlap.
#[test]
fn pool_partitions_tokens_at_one_to_one_scale() {
    let pooler = SpanPooler::new(2);
    let token_embeddings = vec![
        vec![1.0, 0.0], // t0
        vec![1.0, 0.0], // t1
        vec![0.0, 1.0], // t2
        vec![0.0, 1.0], // t3
    ];
    let slabs = vec![Slab::new("a", 0, 2, 0), Slab::new("b", 2, 4, 1)];

    let pooled = pooler.pool(&token_embeddings, &slabs, 4);

    assert_eq!(pooled.len(), 2);
    assert_vec_close(&pooled[0], &[1.0, 0.0]); // mean([t0,t1]) -> [1,0]
    assert_vec_close(&pooled[1], &[0.0, 1.0]); // mean([t2,t3]) -> [0,1]
}

/// `pool_with_offsets` uses half-open overlap: a token whose end equals the
/// slab start, or whose start equals the slab end, is excluded.
///
/// Tokens at byte spans (0,3), (3,6), (6,9). A slab over bytes 3..6 overlaps
/// only the middle token: (0,3) ends exactly at 3 (no overlap), (6,9) starts
/// exactly at 6 (no overlap). The middle token is `[0, 2]`, normalized `[0, 1]`.
#[test]
fn pool_with_offsets_excludes_boundary_touching_tokens() {
    let pooler = SpanPooler::new(2);
    let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0], vec![2.0, 2.0]];
    let token_offsets = vec![(0, 3), (3, 6), (6, 9)];
    let slab = Slab::new("mid", 3, 6, 0);

    let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &[slab]);

    assert_eq!(pooled.len(), 1);
    assert_vec_close(&pooled[0], &[0.0, 1.0]);
}

/// `pool_with_offsets` averages every overlapping token before normalizing.
///
/// A slab over bytes 2..6 overlaps tokens (0,3) and (3,6) but not (6,9). Their
/// mean is `[1, 1]`, normalized to `[1/sqrt(2), 1/sqrt(2)]`.
#[test]
fn pool_with_offsets_averages_overlapping_tokens() {
    let pooler = SpanPooler::new(2);
    let token_embeddings = vec![vec![2.0, 0.0], vec![0.0, 2.0], vec![2.0, 2.0]];
    let token_offsets = vec![(0, 3), (3, 6), (6, 9)];
    let slab = Slab::new("two", 2, 6, 0);

    let pooled = pooler.pool_with_offsets(&token_embeddings, &token_offsets, &[slab]);

    assert_vec_close(&pooled[0], &[SQRT_HALF, SQRT_HALF]);
}
