use abomonation::Abomonation;
use anyhow::{Context, Result};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use kcmkc::configuration::*;
use kcmkc::reporter::Reporter;
use kcmkc_base::{self, dataset::Dataset, dataset::Datatype, types::*};
use log::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::{cell::RefCell, rc::Rc};
use std::{fmt::Debug, time::Instant};
use timely::dataflow::operators::*;
use timely::{communication::Allocator, worker::Worker};

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
    println!("centers returned to main in {:?}", elapsed);

    let (radius_no_outliers, _radius_all_points) =
        compute_radius_outliers(&items, &centers, outliers);
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
) -> Result<()>
where
    for<'de> V: Deserialize<'de> + Abomonation,
{
    let mut reporter = Reporter::from_config(config.clone());
    let matroid = V::configure_constraint(&config);

    let start = Instant::now();
    let dataset = Dataset::new(&config.dataset);

    println!("read and exchange");
    // read and exchange information about the size of the dataset
    let n_handle = Rc::new(RefCell::new(0));
    let n = Rc::clone(&n_handle);
    let (mut input, probe) = worker.dataflow::<(), _, _>(move |scope| {
        let (input, stream) = scope.new_input::<usize>();
        let probe = stream.broadcast().map(move |n| n_handle.replace(n)).probe();
        (input, probe)
    });
    let items = if worker.index() == 0 {
        let data: Vec<V> = dataset.to_vec(Some(config.shuffle_seed))?;
        let n = data.len();
        input.send(n);
        data
    } else {
        Vec::new()
    };
    input.close();
    worker.step_while(|| !probe.done());
    let n = n.take();

    println!("loaded {} items in {:?}", n, start.elapsed());
    let outliers = config.outliers.num_outliers(n);
    let p = n - outliers;

    let mut algorithm = V::configure_parallel_algorithm(&config);
    let timer = Instant::now();
    let centers = algorithm.parallel_run(worker, &items, matroid, p)?;
    let elapsed = timer.elapsed();

    if worker.index() == 0 {
        let (radius_no_outliers, _radius_all_points) =
            compute_radius_outliers(&items, &centers, outliers);
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
        config
            .clone()
            .execute(move |worker| {
                // read and exchange information about the datatype
                let datatype_handle = Rc::new(RefCell::new(None));
                let datatype = Rc::clone(&datatype_handle);
                let (mut input, probe) = worker.dataflow::<(), _, _>(move |scope| {
                    let (input, stream) = scope.new_input::<Datatype>();
                    let probe = stream
                        .broadcast()
                        .map(move |n| datatype_handle.borrow_mut().replace(n))
                        .probe();
                    (input, probe)
                });
                if worker.index() == 0 {
                    let datatype = config.datatype()?;
                    input.send(datatype);
                }
                input.close();
                worker.step_while(|| !probe.done());
                let datatype: Datatype = datatype.take().take().unwrap();
                match datatype {
                    Datatype::WikiPage => run_par::<WikiPage>(&config, worker),
                    Datatype::WikiPageEuclidean => run_par::<WikiPageEuclidean>(&config, worker),
                    Datatype::Song => run_par::<Song>(&config, worker),
                }
            })
            .unwrap();
    }

    Ok(())
}

fn compute_radius_outliers<T: Distance + Sync + Debug>(
    dataset: &[T],
    centers: &[T],
    outliers: usize,
) -> (f32, f32) {
    info!("[radius computation] computing distances to centers");
    let style = ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
    );
    let pb = ProgressBar::new(dataset.len() as u64);
    pb.set_style(style);
    pb.set_draw_delta(10000);
    pb.set_message("Computing radius");
    let mut distances: Vec<OrderedF32> = dataset
        .par_iter()
        .progress_with(pb)
        .map(|x| {
            let closest: OrderedF32 = centers
                .iter()
                .map(|c| {
                    let d = x.distance(c);
                    assert!(!d.is_nan(), "NaN distance between {:?} and {:?}", c, x);
                    d.into()
                })
                .min()
                .unwrap();
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
