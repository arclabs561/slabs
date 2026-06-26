//! Validate Slab byte/char offset handling against naive references on Unicode.
//!
//! Byte<->char offset conversion is where Unicode boundary bugs hide. For text
//! mixing ASCII, accents (2 bytes), CJK (3 bytes), and emoji (4 bytes), this
//! checks: `from_byte_range(r).text == source[r]`; the stored char offsets equal
//! the naive `chars().count()` of the prefix; `from_char_range` slices back to
//! the same text; and byte->char->byte round-trips are identity.
//!
//! ```sh
//! cargo run --release --example offset_validation
//! ```

use std::process::ExitCode;

use slabs::Slab;

fn main() -> ExitCode {
    let source = "Hello, café! 日本語 test 🚀 rocket ☃ done. Ωmega résumé naïve.";
    let char_count = source.chars().count();

    // All valid byte-boundary offsets (start of each char + end of string).
    let byte_bounds: Vec<usize> = source
        .char_indices()
        .map(|(b, _)| b)
        .chain(std::iter::once(source.len()))
        .collect();

    let mut failures = 0u64;
    let mut checks = 0u64;
    let mut check = |cond: bool, what: String| {
        checks += 1;
        if !cond {
            failures += 1;
            if failures <= 8 {
                eprintln!("  VIOLATION: {what}");
            }
        }
    };

    // from_byte_range: text matches the byte slice; char offsets match naive counts.
    for (bi, &bs) in byte_bounds.iter().enumerate() {
        for &be in &byte_bounds[bi..] {
            let slab = Slab::from_byte_range(source, bs..be, 0).expect("valid byte range");
            check(slab.text == source[bs..be], format!("byte text {bs}..{be}"));
            let span = slab.char_span().expect("char span computed");
            check(
                span.start == source[..bs].chars().count(),
                format!("char_start at byte {bs}"),
            );
            check(
                span.end == source[..be].chars().count(),
                format!("char_end at byte {be}"),
            );
        }
    }

    // from_char_range: text matches the char slice; round-trips to the byte form.
    for cs in 0..=char_count {
        for ce in cs..=char_count {
            let slab = Slab::from_char_range(source, cs..ce, 0).expect("valid char range");
            let want: String = source.chars().skip(cs).take(ce - cs).collect();
            check(slab.text == want, format!("char text {cs}..{ce}"));
            // byte offsets must slice back to the same text
            check(
                source[slab.start..slab.end] == want,
                format!("char->byte slice {cs}..{ce}"),
            );
        }
    }

    println!("{checks} checks over {char_count}-char Unicode source, {failures} violations");
    if failures == 0 {
        println!("PASS: byte/char offset handling matches naive references");
        ExitCode::SUCCESS
    } else {
        eprintln!("FAIL: offset conversion diverged from naive references");
        ExitCode::FAILURE
    }
}
