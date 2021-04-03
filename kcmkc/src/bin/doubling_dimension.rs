// Utilities to estimate the doubling dimension of a dataset

#[macro_use]
extern crate log;

use anyhow::{Context, Result};

use kcmkc::configuration::*;
use kcmkc_base::{
    dataset::{Dataset, Datatype},
    types::{Distance, OrderedF32, Song, WikiPage, WikiPageEuclidean},
};
use kcmkc_sequential::disks::*;
use serde::Deserialize;
use serde::Serialize;
use std::io::BufWriter;
use std::{fs::File, path::PathBuf};
use std::{io::prelude::*, time::Duration};

use timely::{
    communication::{Allocator, WorkerGuards},
    dataflow::channels::pact::Exchange,
    worker::Worker,
};
use timely::{
    dataflow::operators::*,
    dataflow::{InputHandle, ProbeHandle},
    ExchangeData,
};

fn estimate_doubling_dimension<T: Distance + ExchangeData>(
    config: Config,
    worker: &mut Worker<Allocator>,
) -> Result<()>
where
    for<'de> T: Deserialize<'de>,
{
    let dataset = Dataset::new(&config.dataset);
    let n = dataset.size::<T>()?;
    info!("Dataset of size {}", n);

    let mut outfile = BufWriter::new(File::create(config.output)?);
    let mut pl = Some(
        progress_logger::ProgressLogger::builder()
            .with_expected_updates(n as u64)
            .with_items_name("points")
            .with_frequency(Duration::from_secs(1))
            .start(),
    );
    let mut input: InputHandle<(), (usize, T)> = InputHandle::new();
    let mut probe: ProbeHandle<()> = ProbeHandle::new();
    let wid = worker.index();
    let peers = worker.peers();

    worker.dataflow(|scope| {
        scope
            .input_from(&mut input)
            // broadcast the input to all workers
            .broadcast()
            .accumulate(Vec::new(), |stash, data| {
                stash.extend(data.replace(Vec::new()))
            })
            .flat_map(move |stash| {
                let mut pl = progress_logger::ProgressLogger::builder()
                    .with_items_name("rows")
                    .with_expected_updates((stash.len() / peers) as u64)
                    .start();
                let mut res = Vec::new();
                for (i, x) in stash.iter() {
                    if i % peers == wid {
                        let mut dists: Vec<(usize, OrderedF32)> = stash
                            .iter()
                            .map(|(j, y)| (*j, x.distance(y).into()))
                            .collect();
                        dists.sort_unstable_by_key(|pair| pair.1);
                        let dists: Vec<(usize, f32)> =
                            dists.into_iter().map(|(i, d)| (i, d.into())).collect();
                        res.push((*i, dists));
                        pl.update(1u64);
                    }
                }
                pl.stop();
                res.into_iter()
            })
            .broadcast()
            .accumulate(Vec::new(), |stash, data| {
                stash.extend(data.replace(Vec::new()))
            })
            .flat_map(move |dists| {
                info!("building disk builder");
                let disk_builder = DiskBuilder::from_distances(dists);
                let mut bitmap = roaring::RoaringBitmap::new();
                info!("...done building disk builder");
                (0..n)
                    .into_iter()
                    .filter(move |u| *u % peers == wid)
                    .map(move |u| {
                        let ecc = disk_builder.eccentricity(u);
                        let mut doubling_dimension = 0u32;
                        for &r in &[ecc, ecc / 2.0, ecc / 4.0] {
                            bitmap.clear();
                            bitmap.insert(u as u32);
                            for x in disk_builder.disk(u, r) {
                                bitmap.insert(x as u32);
                            }
                            let mut cnt = 0;
                            while let Some(c) = bitmap.iter().next() {
                                bitmap.remove(c);
                                for v in disk_builder.disk(c as usize, r / 2.0) {
                                    bitmap.remove(v as u32);
                                }
                                cnt += 1;
                            }
                            assert!(bitmap.is_empty());
                            doubling_dimension = std::cmp::max(doubling_dimension, cnt);
                        }
                        (u, doubling_dimension)
                    })
            })
            // Direct everything to the first worker, which will write the output to file
            .unary_notify(
                Exchange::new(|_| 0),
                "file_writer",
                None,
                move |input, output, notificator| {
                    input.for_each(|t, data| {
                        pl.as_mut().unwrap().update(data.len() as u64);
                        for (i, dd) in data.replace(Vec::new()) {
                            writeln!(outfile, "{}, {}", i, dd).expect("error writing to file");
                        }
                        notificator.notify_at(t.retain());
                    });

                    notificator.for_each(|t, _, _| {
                        info!("flushing the output writer");
                        outfile.flush().unwrap();
                        info!("...done");
                        output.session(&t).give(());
                        pl.take().unwrap().stop();
                    });
                },
            )
            .probe_with(&mut probe);
    });

    if worker.index() == 0 {
        let dataset: Vec<T> = dataset.to_vec(None)?;
        for (i, x) in dataset.into_iter().enumerate() {
            input.send((i, x));
        }
    }
    input.close();
    worker.step_while(|| !probe.done());
    info!("completed dataflow");
    return Ok(());
}

#[derive(Deserialize, Serialize, Clone)]
struct Config {
    dataset: PathBuf,
    output: PathBuf,
    parallel: ParallelConfiguration,
}

impl Config {
    fn load(spec: String) -> anyhow::Result<Self> {
        let path = PathBuf::from(&spec);
        let config: Config = if path.is_file() {
            serde_json::from_reader(std::fs::File::open(path)?)?
        } else {
            let decoded_str = String::from_utf8(base64::decode(spec)?)?;
            serde_json::from_str(&decoded_str)?
        };

        Ok(config)
    }

    fn datatype(&self) -> anyhow::Result<Datatype> {
        Ok(Dataset::new(&self.dataset).metadata()?.datatype)
    }

    fn execute<T, F>(&self, func: F) -> Result<Option<WorkerGuards<T>>>
    where
        T: Send + 'static,
        F: Fn(&mut Worker<timely::communication::Allocator>) -> T + Send + Sync + 'static,
    {
        self.parallel.execute(func, self.clone())
    }
}

impl WithProcessId for Config {
    fn with_process_id(&self, process_id: usize) -> Self {
        let parallel = ParallelConfiguration {
            process_id: Some(process_id),
            ..self.parallel.clone()
        };
        Self {
            parallel,
            ..self.clone()
        }
    }
}

impl ConfEncode for Config {
    fn conf_encode(&self) -> String {
        base64::encode(&serde_json::to_string(&self).unwrap())
    }
}

fn main() -> Result<()> {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let config = Config::load(std::env::args().nth(1).context("missing argument")?)?;

    config.clone().execute(move |worker| {
        match config.datatype().unwrap() {
            Datatype::WikiPage => estimate_doubling_dimension::<WikiPage>(config.clone(), worker),
            Datatype::WikiPageEuclidean => {
                estimate_doubling_dimension::<WikiPageEuclidean>(config.clone(), worker)
            }
            Datatype::Song => estimate_doubling_dimension::<Song>(config.clone(), worker),
        }
        .unwrap();
    })?;

    Ok(())
}
