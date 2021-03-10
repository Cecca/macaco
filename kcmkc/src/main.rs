mod configuration;
mod reporter;

use anyhow::{Context, Result};
use configuration::*;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, types::*};
use reporter::Reporter;
use serde::Deserialize;
use std::{fmt::Debug, time::Instant};

fn run<V: Distance + Clone + Debug + Configure>(config: &Configuration) -> Result<()>
where
    for<'de> V: Deserialize<'de>,
{
    let mut reporter = Reporter::from_config(config.clone());
    if let Some(id) = reporter.already_run()? {
        println!("Experiment already run ({})", id);
        return Ok(());
    }

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

    reporter.set_outcome(elapsed, radius, centers.len() as u32, uncovered as u32);
    reporter.save()?;

    Ok(())
}

fn main() -> Result<()> {
    let config_spec = std::env::args()
        .nth(1)
        .context("provide the specification of the configuration, either a file path or a json object encoded as base64")?;
    let config = Configuration::load(config_spec)?;

    match config.datatype()? {
        Datatype::WikiPage => run::<WikiPage>(&config),
        Datatype::Song => run::<Song>(&config),
    }?;

    Ok(())
}
