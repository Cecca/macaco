use kcmkc_base::{
    algorithm::{self, Algorithm},
    dataset::{Constraint, Dataset, Datatype},
    matroid::{Matroid, PartitionMatroid, TransveralMatroid},
    types::{Song, WikiPage},
};
use kcmkc_sequential::{chen_et_al::ChenEtAl, random::RandomClustering};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub enum AlgorithmConfig {
    Random { seed: u64 },
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
    pub algorithm: AlgorithmConfig,
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

pub trait Configure {
    fn configure_constraint(conf: &Configuration) -> Box<dyn Matroid<Self>>;
    fn configure_algorithm(conf: &Configuration) -> Box<dyn Algorithm<Self>>;
}

impl Configure for WikiPage {
    fn configure_constraint(conf: &Configuration) -> Box<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Box::new(TransveralMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
    fn configure_algorithm(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering { seed }),
        }
    }
}

impl Configure for Song {
    fn configure_constraint(conf: &Configuration) -> Box<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => {
                Box::new(PartitionMatroid::new(categories.clone()))
            }
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_algorithm(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering { seed }),
        }
    }
}
