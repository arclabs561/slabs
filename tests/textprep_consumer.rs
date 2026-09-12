//! Consumer contract for `textprep` Unicode scalar offsets.

use slabs::{Slab, SpanPooler};
use textprep::tokenize::tokenize_with_offsets;

fn chars(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn naive_pool(
    token_embeddings: &[Vec<f32>],
    token_offsets: &[(usize, usize)],
    slab: &Slab,
) -> Vec<f32> {
    let span = slab.char_span().expect("slab has character offsets");
    let selected: Vec<&[f32]> = token_offsets
        .iter()
        .zip(token_embeddings)
        .filter_map(|(&(start, end), embedding)| {
            (start < span.end && end > span.start).then_some(embedding.as_slice())
        })
        .collect();

    assert!(!selected.is_empty(), "test slab must overlap a token");
    let mut mean = vec![0.0; selected[0].len()];
    for embedding in &selected {
        for (sum, value) in mean.iter_mut().zip(*embedding) {
            *sum += value;
        }
    }
    for value in &mut mean {
        *value /= selected.len() as f32;
    }
    let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in &mut mean {
        *value /= norm;
    }
    mean
}

#[test]
fn textprep_scalar_offsets_reconstruct_tokens_and_pool_exactly() {
    let corpus = [
        "plain ASCII words",
        "composed café déjà",
        "東京 日本語 文",
        "astral 🚀 emoji 🦀",
        "ASCII café 東京 🚀 together",
    ];

    for source in corpus {
        let tokens = tokenize_with_offsets(source);
        assert!(
            tokens.len() >= 2,
            "fixture must produce at least two tokens"
        );
        for token in &tokens {
            assert_eq!(chars(source, token.start, token.end), token.text);
        }

        let token_offsets: Vec<_> = tokens
            .iter()
            .map(|token| (token.start, token.end))
            .collect();
        let token_embeddings: Vec<_> = (0..tokens.len())
            .map(|index| vec![index as f32 + 1.0, (index as f32 + 1.0).powi(2)])
            .collect();
        let slabs = vec![
            Slab::from_char_range(source, tokens[0].start..tokens[1].start, 0)
                .expect("valid first slab"),
            Slab::from_char_range(
                source,
                tokens[1].start..tokens.last().expect("tokens exist").end,
                1,
            )
            .expect("valid second slab"),
        ];

        let actual = SpanPooler::new(2)
            .try_pool_with_char_offsets(&token_embeddings, &token_offsets, &slabs)
            .expect("textprep offsets satisfy the checked pooling contract");
        let expected: Vec<_> = slabs
            .iter()
            .map(|slab| naive_pool(&token_embeddings, &token_offsets, slab))
            .collect();
        assert_eq!(actual, expected, "pooling parity for {source:?}");

        // The second token starts exactly where the first slab ends, so it must
        // not contribute to the first slab's half-open overlap interval.
        let unit = 1.0_f32 / 2.0_f32.sqrt();
        assert_eq!(actual[0], vec![unit, unit]);
    }
}
