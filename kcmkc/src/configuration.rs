use anyhow::{Context, Result};
use kcmkc_base::{
    algorithm::Algorithm,
    dataset::{Constraint, Dataset, Datatype, Metadata},
    matroid::{Matroid, PartitionMatroid, TransveralMatroid},
    types::{Song, WikiPage},
};
use kcmkc_parallel::mapreduce_coreset::MapReduceCoreset;
use kcmkc_parallel::ParallelAlgorithm;
use kcmkc_sequential::{
    chen_et_al::ChenEtAl, random::RandomClustering, seq_coreset::SeqCoreset,
    streaming_coreset::StreamingCoreset, SequentialAlgorithm,
};
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, path::PathBuf, process::Command, rc::Rc};
use timely::communication::Config as TimelyConfig;
use timely::communication::{WorkerGuards};
use timely::worker::Config as WorkerConfig;
use timely::worker::Worker;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AlgorithmConfig {
    Random { seed: u64 },
    ChenEtAl,
    SeqCoreset { tau: usize },
    StreamingCoreset { tau: usize },
    MapReduceCoreset { tau: usize, seed: u64 },
}

impl AlgorithmConfig {
    pub fn is_sequential(&self) -> bool {
        match self {
            Self::MapReduceCoreset { .. } => false,
            _ => true,
        }
    }
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

pub trait WithProcessId {
    fn with_process_id(&self, pid: usize) -> Self;
}

pub trait ConfEncode {
    fn conf_encode(&self) -> String;
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

impl ParallelConfiguration {
    pub fn execute<C: WithProcessId + ConfEncode, T, F>(
        &self,
        func: F,
        configuration: C,
    ) -> Result<Option<WorkerGuards<T>>>
    where
        T: Send + 'static,
        F: Fn(&mut Worker<timely::communication::Allocator>) -> T + Send + Sync + 'static,
    {
        if self.hosts.is_some() && self.process_id.is_none() {
            let exec = std::env::args().nth(0).unwrap();
            // first, we copy the executable to a known location, so that then we can run it
            let remote_exec = PathBuf::from("/tmp").join(
                PathBuf::from(&exec)
                    .file_name()
                    .with_context(|| format!("cannot get file name for {}", exec))?,
            );
            println!("Copying the executable to minions");
            let handles: Vec<std::process::Child> = self
                .hosts
                .as_ref()
                .unwrap()
                .iter()
                .map(|host| {
                    let dest = format!(
                        "{}:{}",
                        host.name,
                        remote_exec.to_str().expect("cannot convert path to string")
                    );
                    Command::new("rsync")
                        .arg("--progress")
                        .arg(&exec)
                        .arg(dest)
                        .spawn()
                        .context("problem spawning the rsync process")
                        .unwrap()
                })
                .collect();

            for mut h in handles {
                h.wait().expect("problem waiting for the rsync process");
            }

            println!("spawning executable {:?}", exec);
            // This is the top level invocation, which should spawn the processes with ssh
            let handles: Vec<std::process::Child> = self
                .hosts
                .as_ref()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(pid, host)| {
                    let encoded_config = configuration.with_process_id(pid).conf_encode();
                    Command::new("ssh")
                        .arg(&host.name)
                        .arg(&remote_exec)
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
            let communication_config = match &self.hosts {
                None => {
                    if self.threads == 1 {
                        TimelyConfig::Thread
                    } else {
                        TimelyConfig::Process(self.threads)
                    }
                }
                Some(hosts) => TimelyConfig::Cluster {
                    threads: self.threads,
                    process: self.process_id.expect("missing process id"),
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Configuration {
    pub outliers: OutliersSpec,
    pub algorithm: AlgorithmConfig,
    pub dataset: PathBuf,
    pub constraint: Constraint,
    pub parallel: Option<ParallelConfiguration>,
}

impl WithProcessId for Configuration {
    fn with_process_id(&self, process_id: usize) -> Self {
        let parallel_conf = self
            .parallel
            .as_ref()
            .expect("missing parallel configuration");
        let parallel_conf = ParallelConfiguration {
            process_id: Some(process_id),
            ..parallel_conf.clone()
        };
        Self {
            parallel: Some(parallel_conf),
            ..self.clone()
        }
    }
}

impl ConfEncode for Configuration {
    fn conf_encode(&self) -> String {
        base64::encode(&serde_json::to_string(&self).unwrap())
    }
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
        if let Some(parallel) = self.parallel.as_ref() {
            sha.input(format!("{}", parallel.threads));
            if let Some(hosts) = parallel.hosts.as_ref() {
                let mut hosts = hosts.clone();
                hosts.sort();
                for h in hosts {
                    sha.input(format!("{}:{}", h.name, h.port));
                }
            }
        }
        sha.input(format!("{:?}", self.constraint.describe()));
        match self.datatype()? {
            Datatype::WikiPage => {
                let algorithm = WikiPage::configure_algorithm_info(&self);
                sha.input(format!(
                    "{}{}{}",
                    algorithm.name(),
                    algorithm.version(),
                    algorithm.parameters()
                ));
            }
            Datatype::Song => {
                let algorithm = Song::configure_algorithm_info(&self);
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

    pub fn execute<T, F>(&self, func: F) -> Result<Option<WorkerGuards<T>>>
    where
        T: Send + 'static,
        F: Fn(&mut Worker<timely::communication::Allocator>) -> T + Send + Sync + 'static,
    {
        let parallel_conf = self
            .parallel
            .as_ref()
            .context("missing parallel configuration")?;
        parallel_conf.execute(func, self.clone())
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
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>>;
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>>;
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>>;
}

impl Configure for WikiPage {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Rc::new(TransveralMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau, seed } => {
                Box::new(MapReduceCoreset::new(tau, seed))
            }
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau, seed } => {
                Box::new(MapReduceCoreset::new(tau, seed))
            }
            _ => panic!("Cannot run algorithm in parallel"),
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
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau, seed } => {
                Box::new(MapReduceCoreset::new(tau, seed))
            }
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau, seed } => {
                Box::new(MapReduceCoreset::new(tau, seed))
            }
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

pub fn get_hostname() -> String {
    let output = Command::new("hostname")
        .output()
        .expect("Failed to run the hostname command");
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

#[derive(Deserialize, Serialize, Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Host {
    name: String,
    port: String,
}

impl Host {
    pub fn to_string(&self) -> String {
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
