use std::time::Instant;

use anyhow::{Context, Result};
use kcmkc_base::{dataset::*, types::*};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("provide the path")?;
    let dataset = Dataset::new(path);
    let meta = dataset.metadata()?;
    let start = Instant::now();
    let items: Vec<Song> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());

    // match meta.constraint {
    //     Constraint::Transversal { topics } => {
    //         let matroid = TransveralMatroid::<WikiPage>::new(topics);
    //         let start = Instant::now();
    //         let independent_set = matroid.maximal_independent_set(&items[..100]);
    //         let elapsed = start.elapsed();
    //         for page in independent_set {
    //             println!("{:?} {:?}", page.title, page.topics);
    //         }
    //         println!("Independent set found in {:?}", elapsed);
    //     }
    // }

    Ok(())
}
