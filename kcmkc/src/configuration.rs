use kcmkc_base::{
    dataset::{Constraint, Dataset, Datatype},
    matroid::{Matroid, PartitionMatroid, TransveralMatroid},
    types::{Song, WikiPage},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
pub enum Algorithm {
    ChenEtAl,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum OutliersSpec {
    Fixed(usize),
    Percentage(f64),
}

impl OutliersSpec {
    pub fn num_outliers(&self, data_size: usize) -> usize {
        match self {
            Self::Fixed(x) => *x,
            Self::Percentage(p) => (p * data_size as f64).floor() as usize,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub outliers: OutliersSpec,
    pub algorithm: Algorithm,
    pub dataset: PathBuf,
    pub constraint: Constraint,
}

impl Configuration {
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let config: Configuration = serde_json::from_reader(std::fs::File::open(path.as_ref())?)?;
        // Validate the constraint against the input
        let dataset_meta = Dataset::new(&config.dataset).metadata()?;
        if !config
            .constraint
            .is_compatible_with(&dataset_meta.constraint)
        {
            anyhow::bail!("incompatible constraints");
        };

        Ok(config)
    }

    pub fn datatype(&self) -> anyhow::Result<Datatype> {
        Ok(Dataset::new(&self.dataset).metadata()?.datatype)
    }
}

pub trait BuildConstraint {
    fn build_constaint(conf: &Configuration) -> Box<dyn Matroid<Self>>;
}

impl BuildConstraint for WikiPage {
    fn build_constaint(conf: &Configuration) -> Box<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Box::new(TransveralMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
}

impl BuildConstraint for Song {
    fn build_constaint(conf: &Configuration) -> Box<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => {
                Box::new(PartitionMatroid::new(categories.clone()))
            }
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
}
