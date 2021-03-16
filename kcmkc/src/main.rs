mod configuration;
mod reporter;

use anyhow::{Context, Result};
use configuration::*;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, types::*};
use reporter::Reporter;
use serde::Deserialize;
use std::{collections::BTreeSet, fmt::Debug, time::Instant};

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
    let centers = algorithm.run(&items[..], matroid, p)?;
    let elapsed = timer.elapsed();

    let radius = compute_radius(&items, &centers, p);
    println!(
        "Found clustering with {} centers in {:?}, with radius {}",
        centers.len(),
        elapsed,
        radius
    );

    reporter.set_outcome(elapsed, radius, centers.len() as u32);
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

fn compute_radius<T: Distance>(dataset: &[T], centers: &[T], p: usize) -> f32 {
    let mut topk = TopK::new(dataset.len() - p);
    for x in dataset {
        let closest: OrderedF32 = centers.iter().map(|c| x.distance(c).into()).min().unwrap();
        topk.insert(closest);
    }
    topk.kth()
}

#[derive(Debug)]
struct TopK {
    k: usize,
    topk: BTreeSet<OrderedF32>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            topk: BTreeSet::new(),
        }
    }
    fn insert<I: Into<OrderedF32>>(&mut self, x: I) {
        self.topk.insert(x.into());
        if self.topk.len() == self.k {
            let min = *self.topk.iter().next().unwrap();
            self.topk.remove(&min);
        }
    }
    fn kth(&self) -> f32 {
        (*self.topk.iter().next().unwrap()).into()
    }
}
