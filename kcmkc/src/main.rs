use std::time::Instant;

use anyhow::{Context, Result};
use kcmkc_base::{dataset::*, types::WikiPage};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("provide the path")?;
    let dataset = Dataset::new(path);
    let meta = dataset.metadata()?;
    println!("{:#?}", meta);
    let start = Instant::now();
    let items: Vec<WikiPage> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());
    Ok(())
}
