use std::time::Instant;

use anyhow::{Context, Result};
use kcmkc_base::{dataset::*, matroid::TransveralMatroid, types::*};
use kcmkc_sequential::{chen_et_al::robust_matroid_center, kcenter::*};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("provide the path")?;
    let dataset = Dataset::new(path);
    // let meta = dataset.metadata()?;
    let start = Instant::now();
    let items: Vec<WikiPage> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());

    let matroid: TransveralMatroid<WikiPage> =
        TransveralMatroid::new((0..100u32).collect::<Vec<u32>>());

    let p = items.len() - 900;
    let (centers, uncovered, assignment) = robust_matroid_center(&items, matroid, p);
    let radius = assignment
        .flat_map(|(_, assignment)| assignment.into_iter())
        .map(|(_, d)| d)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "Found clustering with {} centers, {} uncovered nodes, and radius {}",
        centers.len(),
        uncovered,
        radius
    );
    for center in centers {
        println!("{:?}", center.title);
    }

    Ok(())
}
