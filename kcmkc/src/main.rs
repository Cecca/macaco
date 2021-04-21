use abomonation::Abomonation;
use anyhow::{Context, Result};
use kcmkc::configuration::*;
use kcmkc::reporter::Reporter;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, types::*};
use rayon::prelude::*;
use serde::Deserialize;
use std::sync::{Arc, Barrier, RwLock};
use std::{fmt::Debug, time::Instant};
use timely::{communication::Allocator, worker::Worker};
use indicatif::{ProgressBar, ParallelProgressIterator};
use log::*;

fn run_seq<V: Distance + Clone + Debug + Configure + Sync>(config: &Configuration) -> Result<()>
where
    for<'de> V: Deserialize<'de> + Abomonation,
{
    let mut reporter = Reporter::from_config(config.clone());
    let matroid = V::configure_constraint(&config);

    let start = Instant::now();
    let dataset = Dataset::new(&config.dataset);
    let items: Vec<V> = dataset.to_vec(Some(config.shuffle_seed))?;
    println!("loaded {} items in {:?}", items.len(), start.elapsed());
    let outliers = config.outliers.num_outliers(items.len());
    let p = items.len() - outliers;

    let mut algorithm = V::configure_sequential_algorithm(&config);
    let timer = Instant::now();
    let centers = algorithm.sequential_run(&items[..], matroid, p)?;
    let elapsed = timer.elapsed();

    let (radius_no_outliers, _radius_all_points) =
        compute_radius_outliers(&dataset.to_vec(None)?, &centers, outliers);
    assert!(radius_no_outliers < _radius_all_points);
    println!(
        "Found clustering with {} centers in {:?}, with radius {}",
        centers.len(),
        elapsed,
        radius_no_outliers
    );

    if let Some(coreset) = algorithm.coreset() {
        reporter.set_coreset_info(coreset.len())
    }

    reporter.set_outcome(elapsed, radius_no_outliers, centers.len() as u32);
    reporter.set_profile(algorithm.time_profile());
    reporter.set_counters(algorithm.counters());
    reporter.save()?;

    Ok(())
}

fn run_par<V: Distance + Clone + Debug + Configure + Sync>(
    config: &Configuration,
    worker: &mut Worker<Allocator>,
    items: Arc<RwLock<Option<Vec<V>>>>,
    barrier: Arc<Barrier>,
) -> Result<()>
where
    for<'de> V: Deserialize<'de> + Abomonation,
{
    let mut reporter = Reporter::from_config(config.clone());
    let matroid = V::configure_constraint(&config);

    let start = Instant::now();
    let dataset = Dataset::new(&config.dataset);
    // let items: Vec<V> = dataset.to_vec(Some(config.shuffle_seed))?;
    items
        .write()
        .unwrap()
        .get_or_insert_with(|| dataset.to_vec(Some(config.shuffle_seed)).unwrap());

    // Wait for all local threads
    barrier.wait();

    let n = items.read().unwrap().as_ref().unwrap().len();
    println!("loaded {} items in {:?}", n, start.elapsed());
    let outliers = config.outliers.num_outliers(n);
    let p = n - outliers;

    let mut algorithm = V::configure_parallel_algorithm(&config);
    let timer = Instant::now();
    let centers =
        algorithm.parallel_run(worker, items.read().unwrap().as_ref().unwrap(), matroid, p)?;
    let elapsed = timer.elapsed();

    if worker.index() == 0 {
        let (radius_no_outliers, _radius_all_points) =
            compute_radius_outliers(items.read().unwrap().as_ref().unwrap(), &centers, outliers);
        assert!(radius_no_outliers < _radius_all_points);
        println!(
            "Found clustering with {} centers in {:?}, with radius {}",
            centers.len(),
            elapsed,
            radius_no_outliers
        );

        if let Some(coreset) = algorithm.coreset() {
            reporter.set_coreset_info(coreset.len())
        }

        println!("Reporting result");
        println!(" . set outcome");
        reporter.set_outcome(elapsed, radius_no_outliers, centers.len() as u32);
        println!(" . set profile");
        reporter.set_profile(algorithm.time_profile());
        println!(" . set counters");
        reporter.set_counters(algorithm.counters());
        println!(" . save");
        reporter.save()?;
        println!("...done!")
    }
    Ok(())
}

fn main() -> Result<()> {
    env_logger::init();
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
            Datatype::WikiPageEuclidean => run_seq::<WikiPageEuclidean>(&config),
            Datatype::Song => run_seq::<Song>(&config),
        }?;
    } else {
        // config.clone().execute(move |worker| {
        //     match config.datatype().unwrap() {
        //         Datatype::WikiPage => run_par::<WikiPage>(&config, worker),
        //         Datatype::Song => run_par::<Song>(&config, worker),
        //     }
        //     .unwrap();
        // })?;

        let barrier = Arc::new(Barrier::new(config.clone().parallel.unwrap().threads));
        match config.datatype().unwrap() {
            Datatype::WikiPage => {
                let items: Arc<RwLock<Option<Vec<WikiPage>>>> = Arc::new(RwLock::new(None));
                config
                    .clone()
                    .execute(move |worker| {
                        run_par::<WikiPage>(
                            &config,
                            worker,
                            Arc::clone(&items),
                            Arc::clone(&barrier),
                        )
                    })
                    .unwrap();
            }
            Datatype::WikiPageEuclidean => {
                let items: Arc<RwLock<Option<Vec<WikiPageEuclidean>>>> =
                    Arc::new(RwLock::new(None));
                config
                    .clone()
                    .execute(move |worker| {
                        run_par::<WikiPageEuclidean>(
                            &config,
                            worker,
                            Arc::clone(&items),
                            Arc::clone(&barrier),
                        )
                    })
                    .unwrap();
            }
            Datatype::Song => {
                let items: Arc<RwLock<Option<Vec<Song>>>> = Arc::new(RwLock::new(None));
                config
                    .clone()
                    .execute(move |worker| {
                        run_par::<Song>(&config, worker, Arc::clone(&items), Arc::clone(&barrier))
                    })
                    .unwrap();
            }
        }
    }

    Ok(())
}

fn compute_radius_outliers<T: Distance + Sync>(
    dataset: &[T],
    centers: &[T],
    outliers: usize,
) -> (f32, f32) {
    info!("[radius computation] computing distances to centers");
    let mut distances: Vec<OrderedF32> = dataset
        .par_iter()
        .progress()
        .map(|x| {
            let closest: OrderedF32 = centers.iter().map(|c| x.distance(c).into()).min().unwrap();
            closest
        })
        .collect();
    info!("[radius computation] sorting distances");
    distances.par_sort_unstable();
    (
        distances[distances.len() - outliers - 1].into(),
        distances[distances.len() - 1].into(),
    )
}
