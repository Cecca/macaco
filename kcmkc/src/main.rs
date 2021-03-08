mod configuration;

use anyhow::{Context, Result};
use configuration::*;
use kcmkc_base::{
    self, algorithm::Algorithm, dataset::Dataset, dataset::Datatype, matroid::Matroid, types::*,
};
use kcmkc_sequential::{chen_et_al::ChenEtAl, random::RandomClustering};
use serde::Deserialize;
use std::{fmt::Debug, time::Instant};

fn run<V: Distance + Clone + Debug + Configure>(config: &Configuration) -> Result<()>
where
    for<'de> V: Deserialize<'de>,
{
    let matroid = V::configure_constraint(&config);
    let mut algorithm = V::configure_algorithm(&config);

    let dataset = Dataset::new(&config.dataset);
    let start = Instant::now();
    let items: Vec<V> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());

    let outliers = config.outliers.num_outliers(items.len());
    let p = items.len() - outliers;
    let timer = Instant::now();
    let (centers, uncovered, assignment) = algorithm.run(&items[..], matroid, p)?;
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

    Ok(())
}

fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .context("provide the path to the configuration file")?;
    let config = Configuration::load(config_path)?;

    match config.datatype()? {
        Datatype::WikiPage => run::<WikiPage>(&config),
        Datatype::Song => run::<Song>(&config),
    }?;

    Ok(())
}
