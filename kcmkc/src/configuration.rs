
use kcmkc_base::{
    algorithm::Algorithm,
    dataset::{Constraint, Dataset, Datatype, Metadata},
    matroid::{Matroid, PartitionMatroid, TransveralMatroid},
    types::{Song, WikiPage},
};
use kcmkc_sequential::{
    chen_et_al::ChenEtAl,
    random::RandomClustering,
    seq_coreset::{SeqCoreset},
};
use serde::{Deserialize, Serialize};
use std::path::{PathBuf};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AlgorithmConfig {
    Random { seed: u64 },
    ChenEtAl,
    SeqCoreset { epsilon: f32 },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OutliersSpec {
    Fixed(usize),
    Percentage(f64),
}

impl OutliersSpec {
    pub fn describe(&self) -> String {
        match self {
            Self::Fixed(x) => format!("Fixed({})", x),
            Self::Percentage(p) => format!("Percentage({})", p),
        }
    }

    pub fn num_outliers(&self, data_size: usize) -> usize {
        match self {
            Self::Fixed(x) => *x,
            Self::Percentage(p) => (p * data_size as f64).floor() as usize,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Configuration {
    pub outliers: OutliersSpec,
    pub algorithm: AlgorithmConfig,
    pub dataset: PathBuf,
    pub constraint: Constraint,
}

impl Configuration {
    pub fn load(spec: String) -> anyhow::Result<Self> {
        let path = PathBuf::from(&spec);
        let config: Configuration = if path.is_file() {
            serde_json::from_reader(std::fs::File::open(path)?)?
        } else {
            let decoded_str = String::from_utf8(base64::decode(spec)?)?;
            serde_json::from_str(&decoded_str)?
        };

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

    pub fn dataset_metadata(&self) -> anyhow::Result<Metadata> {
        Dataset::new(&self.dataset).metadata()
    }

    pub fn sha(&self) -> anyhow::Result<String> {
        use sha2::Digest;
        let mut sha = sha2::Sha256::new();

        let data_meta = Dataset::new(&self.dataset).metadata()?;
        sha.input(format!(
            "{}{}{:?}{:?}",
            data_meta.name,
            data_meta.version,
            data_meta.parameters_string(),
            data_meta.constraint
        ));
        sha.input(format!("{:?}", self.constraint.describe()));
        match self.datatype()? {
            Datatype::WikiPage => {
                let algorithm = WikiPage::configure_algorithm(&self);
                sha.input(format!(
                    "{}{}{}",
                    algorithm.name(),
                    algorithm.version(),
                    algorithm.parameters()
                ));
            }
            Datatype::Song => {
                let algorithm = Song::configure_algorithm(&self);
                sha.input(format!(
                    "{}{}{}",
                    algorithm.name(),
                    algorithm.version(),
                    algorithm.parameters()
                ));
            }
        }

        Ok(format!("{:x}", sha.result()))
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
            AlgorithmConfig::SeqCoreset { epsilon } => Box::new(SeqCoreset::new(epsilon)),
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
            AlgorithmConfig::SeqCoreset { epsilon } => Box::new(SeqCoreset::new(epsilon)),
        }
    }
}
