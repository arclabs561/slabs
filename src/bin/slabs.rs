#[cfg(feature = "cli")]
use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker};
#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use clap::{Parser, ValueEnum};

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Text chunking for RAG pipelines", long_about = None)]
struct Args {
    /// Strategy to use for chunking
    #[arg(short = 'S', long, value_enum, default_value_t = Strategy::Recursive)]
    strategy: Strategy,

    /// Max size of each chunk (in characters)
    #[arg(short, long, default_value_t = 500)]
    size: usize,

    /// Overlap between chunks (for Fixed strategy)
    #[arg(short, long, default_value_t = 50)]
    overlap: usize,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = Format::Text)]
    format: Format,

    /// Input file to chunk (omit or use - for stdin)
    #[arg(value_name = "FILE")]
    input: Option<PathBuf>,
}

#[cfg(feature = "cli")]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Strategy {
    Fixed,
    Sentence,
    Recursive,
}

#[cfg(feature = "cli")]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Format {
    Text,
    Json,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cli")]
    {
        let args = Args::parse();

        let text = match &args.input {
            Some(path) if path.to_str() != Some("-") => std::fs::read_to_string(path)?,
            _ => {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            }
        };

        let chunker: Box<dyn Chunker> = match args.strategy {
            Strategy::Fixed => Box::new(FixedChunker::new(args.size, args.overlap)),
            Strategy::Sentence => Box::new(SentenceChunker::new(args.size / 100)),
            Strategy::Recursive => {
                Box::new(RecursiveChunker::new(args.size, &["\n\n", "\n", ". ", " "]))
            }
        };

        let chunks = chunker.chunk(&text);

        match args.format {
            Format::Text => {
                eprintln!(
                    "Found {} chunks using {:?} strategy:",
                    chunks.len(),
                    args.strategy
                );
                for (i, chunk) in chunks.iter().enumerate() {
                    println!("\n--- Chunk {} [{}..{}] ---", i, chunk.start, chunk.end);
                    println!("{}", chunk.text);
                }
            }
            Format::Json => {
                let slabs_json: Vec<_> = chunks
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "index": s.index,
                            "text": s.text,
                            "start": s.start,
                            "end": s.end,
                            "char_start": s.char_start,
                            "char_end": s.char_end,
                            "len": s.len(),
                            "char_len": s.char_len(),
                        })
                    })
                    .collect();

                let envelope = serde_json::json!({
                    "schema_version": 1,
                    "strategy": format!("{:?}", args.strategy).to_lowercase(),
                    "max_size": args.size,
                    "overlap": args.overlap,
                    "total_chunks": chunks.len(),
                    "slabs": slabs_json,
                });

                println!("{}", serde_json::to_string_pretty(&envelope)?);
            }
        }
    }

    #[cfg(not(feature = "cli"))]
    eprintln!("CLI feature is disabled. Build with --features cli to enable.");

    Ok(())
}
