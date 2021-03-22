mod configuration;
mod reporter;

use abomonation::Abomonation;
use anyhow::{Context, Result};
use configuration::*;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, types::*};
use reporter::Reporter;
use serde::Deserialize;
use std::{collections::BTreeSet, fmt::Debug, time::Instant};
use timely::{communication::Allocator, worker::Worker};

fn run_seq<V: Distance + Clone + Debug + Configure>(config: &Configuration) -> Result<()>
where
    for<'de> V: Deserialize<'de> + Abomonation,
{
    let mut reporter = Reporter::from_config(config.clone());
    let matroid = V::configure_constraint(&config);

    let start = Instant::now();
    let dataset = Dataset::new(&config.dataset);
    let items: Vec<V> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());
    let outliers = config.outliers.num_outliers(items.len());
    let p = items.len() - outliers;

    let mut algorithm = V::configure_sequential_algorithm(&config);
    let timer = Instant::now();
    let centers = algorithm.sequential_run(&items[..], matroid, p)?;
    let elapsed = timer.elapsed();

    let radius = compute_radius(&dataset.to_vec()?, &centers, outliers);
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

fn run_par<V: Distance + Clone + Debug + Configure>(
    config: &Configuration,
    worker: &mut Worker<Allocator>,
) -> Result<()>
where
    for<'de> V: Deserialize<'de> + Abomonation,
{
    let mut reporter = Reporter::from_config(config.clone());
    let matroid = V::configure_constraint(&config);

    let start = Instant::now();
    let dataset = Dataset::new(&config.dataset);
    let items: Vec<V> = dataset.to_vec()?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());
    let outliers = config.outliers.num_outliers(items.len());
    let p = items.len() - outliers;

    let mut algorithm = V::configure_parallel_algorithm(&config);
    let timer = Instant::now();
    let centers = algorithm.parallel_run(worker, &items[..], matroid, p)?;
    let elapsed = timer.elapsed();

    if worker.index() == 0 {
        let radius = compute_radius(&dataset.to_vec()?, &centers, outliers);
        println!(
            "Found clustering with {} centers in {:?}, with radius {}",
            centers.len(),
            elapsed,
            radius
        );

        reporter.set_outcome(elapsed, radius, centers.len() as u32);
        reporter.save()?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let config_spec = std::env::args()
        .nth(1)
        .context("provide the specification of the configuration, either a file path or a json object encoded as base64")?;
    let config = Configuration::load(config_spec)?;

    // Exit if the experiment has been already run
    if let Some(id) = Reporter::from_config(config.clone()).already_run()? {
        println!("Experiment already run ({})", id);
        return Ok(());
    }

    if config.algorithm.is_sequential() {
        match config.datatype()? {
            Datatype::WikiPage => run_seq::<WikiPage>(&config),
            Datatype::Song => run_seq::<Song>(&config),
        }?;
    } else {
        config.clone().execute(move |worker| {
            match config.datatype().unwrap() {
                Datatype::WikiPage => run_par::<WikiPage>(&config, worker),
                Datatype::Song => run_par::<Song>(&config, worker),
            }
            .unwrap();
        })?;
    }

    Ok(())
}

fn compute_radius<T: Distance>(dataset: &[T], centers: &[T], outliers: usize) -> f32 {
    let mut topk = TopK::new(outliers);
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
