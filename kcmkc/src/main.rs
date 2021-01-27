use anyhow::{Context, Result};
use kcmkc_base::dataset::*;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("should provide the path")?;
    let dataset = Dataset::new(path);
    let meta = dataset.metadata()?;
    println!("{:?}", meta);

    Ok(())
}
