use anyhow::{Context, Result};
use kcmkc_base::{
    dataset::{Constraint, Dataset, Datatype, Metadata},
    matroid::{Matroid, PartitionMatroid, TransveralMatroid},
    types::{Song, WikiPage},
};
use kcmkc_sequential::{
    chen_et_al::ChenEtAl, random::RandomClustering, seq_coreset::SeqCoreset,
    streaming_coreset::StreamingCoreset, SequentialAlgorithm,
};
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, path::PathBuf, process::Command, rc::Rc};
use timely::communication::Config as TimelyConfig;
use timely::communication::{Allocator, WorkerGuards};
use timely::worker::Config as WorkerConfig;
use timely::worker::Worker;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AlgorithmConfig {
    Random { seed: u64 },
    ChenEtAl,
    SeqCoreset { tau: usize },
    StreamingCoreset { tau: usize },
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
pub struct ParallelConfiguration {
    /// don't set this manually, used to provide info to the child process
    pub process_id: Option<usize>,
    /// number of threads to use
    pub threads: usize,
    /// the hosts to run on
    pub hosts: Option<Vec<Host>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Configuration {
    pub outliers: OutliersSpec,
    pub algorithm: AlgorithmConfig,
    pub dataset: PathBuf,
    pub constraint: Constraint,
    pub parallel: Option<ParallelConfiguration>,
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
                let algorithm = WikiPage::configure_sequential_algorithm(&self);
                sha.input(format!(
                    "{}{}{}",
                    algorithm.name(),
                    algorithm.version(),
                    algorithm.parameters()
                ));
            }
            Datatype::Song => {
                let algorithm = Song::configure_sequential_algorithm(&self);
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

    fn with_process_id(&self, process_id: usize) -> Result<Self> {
        let parallel_conf = self
            .parallel
            .as_ref()
            .context("missing parallel configuration")?;
        let parallel_conf = ParallelConfiguration {
            process_id: Some(process_id),
            ..parallel_conf.clone()
        };
        Ok(Self {
            parallel: Some(parallel_conf),
            ..self.clone()
        })
    }

    fn encode(&self) -> Result<String> {
        Ok(base64::encode(&serde_json::to_string(&self)?))
    }

    pub fn execute<T, F>(&self, func: F) -> Result<Option<WorkerGuards<T>>>
    where
        T: Send + 'static,
        F: Fn(&mut Worker<timely::communication::Allocator>) -> T + Send + Sync + 'static,
    {
        let parallel_conf = self
            .parallel
            .as_ref()
            .context("missing parallel configuration")?;
        if parallel_conf.process_id.is_none() {
            let exec = std::env::args().nth(0).unwrap();
            println!("spawning executable {:?}", exec);
            // This is the top level invocation, which should spawn the processes with ssh
            let handles: Vec<std::process::Child> = parallel_conf
                .hosts
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(pid, host)| {
                    let encoded_config = self.with_process_id(pid).unwrap().encode().unwrap();
                    Command::new("ssh")
                        .arg(&host.name)
                        .arg(&exec)
                        .arg(encoded_config)
                        .spawn()
                        .context("problem spawning the ssh process")
                        .unwrap()
                })
                .collect();

            for mut h in handles {
                h.wait().expect("problem waiting for the ssh process");
            }

            Ok(None)
        } else {
            let worker_config = WorkerConfig::default();
            let communication_config = match &parallel_conf.hosts {
                None => {
                    if parallel_conf.threads == 1 {
                        TimelyConfig::Thread
                    } else {
                        TimelyConfig::Process(parallel_conf.threads)
                    }
                }
                Some(hosts) => TimelyConfig::Cluster {
                    threads: parallel_conf.threads,
                    process: parallel_conf.process_id.expect("missing process id"),
                    addresses: hosts.to_strings(),
                    report: false,
                    log_fn: Box::new(|_| None),
                },
            };
            let config = timely::execute::Config {
                communication: communication_config,
                worker: worker_config,
            };
            let guards = timely::execute(config, func)
                .map_err(|e| anyhow::anyhow!(e))
                .context("timely execute")?;
            Ok(Some(guards))
        }
    }
}

pub trait ToStrings {
    fn to_strings(&self) -> Vec<String>;
}

impl ToStrings for Vec<Host> {
    fn to_strings(&self) -> Vec<String> {
        self.iter().map(|h| h.to_string()).collect()
    }
}

pub trait Configure {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>>;
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>>;
}

impl Configure for WikiPage {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Rc::new(TransveralMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering { seed }),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
        }
    }
}

impl Configure for Song {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => {
                Rc::new(PartitionMatroid::new(categories.clone()))
            }
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering { seed }),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
        }
    }
}

pub fn get_hostname() -> String {
    use std::process::Command;
    let output = Command::new("hostname")
        .output()
        .expect("Failed to run the hostname command");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Host {
    name: String,
    port: String,
}

impl Host {
    fn to_string(&self) -> String {
        format!("{}:{}", self.name, self.port)
    }
}

impl TryFrom<&str> for Host {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut tokens = value.split(":");
        let name = tokens.next().ok_or("missing host part")?.to_owned();
        let port = tokens.next().ok_or("missing port part")?.to_owned();
        Ok(Self { name, port })
    }
}
