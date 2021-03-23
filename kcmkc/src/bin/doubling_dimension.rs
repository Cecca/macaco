// Utilities to estimate the doubling dimension of a dataset

#[macro_use]
extern crate log;

use anyhow::{Context, Result};
use kcmkc_base::{
    dataset::{Dataset, Datatype},
    types::{Distance, Song, WikiPage},
};
use kcmkc_sequential::disks::*;
use serde::Deserialize;
use serde::Serialize;
use std::io::prelude::*;
use std::{collections::BTreeSet, fs::File, path::PathBuf};

fn estimate_doubling_dimension<T: Distance>(
    config: Config,
) -> Result<Box<dyn Iterator<Item = (usize, u32)>>>
where
    for<'de> T: Deserialize<'de>,
{
    let dataset: Vec<T> = Dataset::new(&config.dataset).to_vec()?;

    info!("Computing pairwise distances...");
    let disk_builder = DiskBuilder::new(&dataset);
    info!("...done");

    let estimates = (0..dataset.len()).map(move |u| {
        let ecc = disk_builder.eccentricity(u);
        let mut doubling_dimension = 0u32;
        for &r in &[ecc, ecc / 2.0, ecc / 4.0] {
            let mut disk: BTreeSet<usize> = disk_builder.disk(u, r).collect();
            let mut cnt = 0;
            while let Some(c) = disk.iter().next() {
                for v in disk_builder.disk(*c, r / 2.0) {
                    disk.remove(&v);
                }
                cnt += 1;
            }
            assert!(disk.is_empty());
            doubling_dimension = std::cmp::max(doubling_dimension, cnt);
        }
        doubling_dimension
    });

    Ok(Box::new(estimates.enumerate()))
}

#[derive(Deserialize, Serialize, Clone)]
struct Config {
    dataset: PathBuf,
    output: PathBuf,
}

impl Config {
    fn load(spec: String) -> anyhow::Result<Self> {
        let path = PathBuf::from(&spec);
        let config: Config = if path.is_file() {
            serde_json::from_reader(std::fs::File::open(path)?)?
        } else {
            anyhow::bail!("Only config load from file is supported");
        };

        Ok(config)
    }

    fn datatype(&self) -> anyhow::Result<Datatype> {
        Ok(Dataset::new(&self.dataset).metadata()?.datatype)
    }
}

fn main() -> Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    pretty_env_logger::init();

    let config = Config::load(std::env::args().nth(1).context("missing argument")?)?;

    let res = match config.datatype()? {
        Datatype::WikiPage => estimate_doubling_dimension::<WikiPage>(config.clone()),
        Datatype::Song => estimate_doubling_dimension::<Song>(config.clone()),
    }?;

    let mut pl = progress_logger::ProgressLogger::builder().start();
    let mut out = GzEncoder::new(File::create(&config.output)?, Compression::best());
    for (i, dd) in res {
        writeln!(out, "{}, {}", i, dd)?;
        pl.update_light(1u64);
    }
    pl.stop();

    Ok(())
}
