//! Slab byte/char offset handling vs naive references on Unicode text.
//!
//! Byte<->char conversion is where Unicode boundary bugs hide. On text mixing
//! ASCII, accents (2 bytes), CJK (3 bytes), and emoji (4 bytes): from_byte_range
//! text matches the byte slice and the char offsets equal naive chars().count();
//! from_char_range slices back to the same text and round-trips to bytes.

use slabs::Slab;

const SOURCE: &str = "Hello, café! 日本語 test 🚀 rocket ☃ done. Ωmega résumé naïve.";

#[test]
fn from_byte_range_matches_naive() {
    let bounds: Vec<usize> = SOURCE
        .char_indices()
        .map(|(b, _)| b)
        .chain(std::iter::once(SOURCE.len()))
        .collect();
    for (bi, &bs) in bounds.iter().enumerate() {
        for &be in &bounds[bi..] {
            let slab = Slab::from_byte_range(SOURCE, bs..be, 0).expect("valid byte range");
            assert_eq!(slab.text, SOURCE[bs..be], "byte text {bs}..{be}");
            let span = slab.char_span().expect("char span");
            assert_eq!(
                span.start,
                SOURCE[..bs].chars().count(),
                "char_start at byte {bs}"
            );
            assert_eq!(
                span.end,
                SOURCE[..be].chars().count(),
                "char_end at byte {be}"
            );
        }
    }
}

#[test]
fn from_char_range_matches_naive_and_round_trips() {
    let char_count = SOURCE.chars().count();
    for cs in 0..=char_count {
        for ce in cs..=char_count {
            let slab = Slab::from_char_range(SOURCE, cs..ce, 0).expect("valid char range");
            let want: String = SOURCE.chars().skip(cs).take(ce - cs).collect();
            assert_eq!(slab.text, want, "char text {cs}..{ce}");
            assert_eq!(
                SOURCE[slab.start..slab.end],
                want,
                "char->byte slice {cs}..{ce}"
            );
        }
    }
}
