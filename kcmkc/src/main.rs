mod configuration;

use anyhow::{Context, Result};
use configuration::*;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, matroid::Matroid, types::*};
use kcmkc_sequential::chen_et_al::robust_matroid_center;
use serde::Deserialize;
use std::{fmt::Debug, time::Instant};

fn run<V: Distance + Clone + Debug>(
    config: &Configuration,
    constraint: Box<dyn Matroid<V>>,
) -> Result<()>
where
    for<'de> V: Deserialize<'de>,
{
    let dataset = Dataset::new(&config.dataset);
    let start = Instant::now();
    let items: Vec<V> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());

    let outliers = config.outliers.num_outliers(items.len());
    let p = items.len() - outliers;
    let timer = Instant::now();
    let (centers, uncovered, assignment) = robust_matroid_center(&items, constraint, p);
    let elapsed = timer.elapsed();

    let radius = assignment
        .flat_map(|(_, assignment)| assignment.into_iter())
        .map(|(_, d)| d)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "Found clustering with {} centers in {:?}, with {} uncovered nodes, and radius {}",
        centers.len(),
        elapsed,
        uncovered,
        radius
    );
    for center in centers {
        println!("{:?}", center);
    }

    Ok(())
}

fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .context("provide the path to the configuration file")?;
    let config = Configuration::load(config_path)?;

    match config.datatype()? {
        Datatype::WikiPage => {
            let matroid = WikiPage::build_constaint(&config);
            run(&config, matroid)
        }
        Datatype::Song => {
            let matroid = Song::build_constaint(&config);
            run(&config, matroid)
        }
    }?;

    Ok(())
}
