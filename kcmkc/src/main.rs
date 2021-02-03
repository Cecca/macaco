use std::time::Instant;

use anyhow::{Context, Result};
use kcmkc_base::{dataset::*, types::*};
use kcmkc_sequential::kcenter::*;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("provide the path")?;
    let dataset = Dataset::new(path);
    // let meta = dataset.metadata()?;
    let start = Instant::now();
    let items: Vec<WikiPage> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());

    let k = 5;
    let (centers, assignment) = kcenter(&items, k);
    for center in centers {
        println!("{:?}", center.title);
    }

    Ok(())
}
