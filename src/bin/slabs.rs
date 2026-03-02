#[cfg(feature = "cli")]
use slabs::{Chunker, FixedChunker, RecursiveChunker, SentenceChunker};
#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use clap::{Parser, ValueEnum};

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Visualize text chunking strategies", long_about = None)]
struct Args {
    /// Strategy to use for chunking
    #[arg(short, long, value_enum, default_value_t = Strategy::Recursive)]
    strategy: Strategy,

    /// Max size of each chunk (in characters)
    #[arg(short, long, default_value_t = 500)]
    size: usize,

    /// Overlap between chunks (for Fixed strategy)
    #[arg(short, long, default_value_t = 50)]
    overlap: usize,

    /// Input file to chunk
    #[arg(value_name = "FILE")]
    input: PathBuf,
}

#[cfg(feature = "cli")]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Strategy {
    Fixed,
    Sentence,
    Recursive,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cli")]
    {
        let args = Args::parse();
        let text = std::fs::read_to_string(&args.input)?;

        let chunker: Box<dyn Chunker> = match args.strategy {
            Strategy::Fixed => Box::new(FixedChunker::new(args.size, args.overlap)),
            Strategy::Sentence => Box::new(SentenceChunker::new(args.size / 100)), // Approximate sentences
            Strategy::Recursive => {
                Box::new(RecursiveChunker::new(args.size, &["\n\n", "\n", ". ", " "]))
            }
        };

        let chunks = chunker.chunk(&text);
        println!(
            "Found {} chunks using {:?} strategy:",
            chunks.len(),
            args.strategy
        );

        for (i, chunk) in chunks.iter().enumerate() {
            println!("\n--- Chunk {} [{}..{}] ---", i, chunk.start, chunk.end);
            println!("{}", chunk.text);
        }
    }

    #[cfg(not(feature = "cli"))]
    println!("CLI feature is disabled. Build with --features cli to enable.");

    Ok(())
}
