use anyhow::{Context, Result};
use macaco_base::{
    algorithm::Algorithm,
    dataset::{Constraint, Dataset, Datatype, Metadata},
    matroid::{Matroid, PartitionMatroid, TransversalMatroid},
    types::{ColorVector, Higgs, Phone, Song, WikiPage, WikiPageEuclidean},
};
use macaco_parallel::mapreduce_coreset::{MapReduceCoreset, MapReduceCoresetRec};
use macaco_parallel::ParallelAlgorithm;
use macaco_sequential::{
    chen_et_al::ChenEtAl, greedy_heuristic::GreedyHeuristic, kale::KaleStreaming,
    random::RandomClustering, seq_coreset::SeqCoreset, streaming_coreset::StreamingCoreset,
    SequentialAlgorithm,
};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::{collections::BTreeMap, convert::TryFrom, path::PathBuf, process::Command, rc::Rc};
use timely::communication::Config as TimelyConfig;
use timely::communication::WorkerGuards;
use timely::worker::Config as WorkerConfig;
use timely::worker::Worker;

pub trait Sha {
    fn update_sha<D: Digest>(&self, sha: &mut D);

    fn sha(&self) -> Result<String> {
        let mut sha = sha2::Sha256::new();
        self.update_sha(&mut sha);
        Ok(format!("{:x}", sha.result()))
    }
}

impl<T: Sha> Sha for Vec<T> {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        for x in self.iter() {
            x.update_sha(sha);
        }
    }
}

impl<K: Sha, V: Sha> Sha for BTreeMap<K, V> {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        for (k, v) in self.iter() {
            k.update_sha(sha);
            v.update_sha(sha);
        }
    }
}

impl Sha for usize {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.to_le_bytes());
    }
}

impl Sha for u64 {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.to_le_bytes());
    }
}

impl Sha for u32 {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.to_le_bytes());
    }
}

impl Sha for f32 {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.to_le_bytes());
    }
}

impl Sha for String {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.as_bytes());
    }
}

impl Sha for Constraint {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        match &self {
            Constraint::Transversal { topics } => {
                sha.input("transversal");
                topics.update_sha(sha);
            }
            Constraint::Partition { categories } => {
                sha.input("partition");
                categories.update_sha(sha);
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum AlgorithmConfig {
    Random { seed: u64 },
    ChenEtAl,
    Greedy,
    SeqCoreset { tau: usize },
    StreamingCoreset { tau: usize },
    MapReduceCoreset { tau: usize },
    MapReduceCoresetRec { tau: usize },
    KaleStreaming { epsilon: f32 },
}

impl AlgorithmConfig {
    pub fn is_sequential(&self) -> bool {
        match self {
            Self::MapReduceCoreset { .. } => false,
            Self::MapReduceCoresetRec { .. } => false,
            _ => true,
        }
    }
}

impl Sha for AlgorithmConfig {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        match &self {
            AlgorithmConfig::Random { seed } => {
                sha.input("random");
                sha.input(seed.to_le_bytes());
            }
            AlgorithmConfig::ChenEtAl => sha.input("chen-et-al"),
            AlgorithmConfig::Greedy => sha.input("greedy"),
            AlgorithmConfig::SeqCoreset { tau } => {
                sha.input("seq-coreset");
                sha.input(tau.to_le_bytes())
            }
            AlgorithmConfig::StreamingCoreset { tau } => {
                sha.input("streaming-coreset");
                sha.input(tau.to_le_bytes())
            }
            AlgorithmConfig::MapReduceCoreset { tau } => {
                sha.input("mapreduce-coreset");
                sha.input(tau.to_le_bytes())
            }
            AlgorithmConfig::MapReduceCoresetRec { tau } => {
                sha.input("mapreduce-coreset-rec");
                sha.input(tau.to_le_bytes())
            }
            AlgorithmConfig::KaleStreaming { epsilon } => {
                sha.input("kale-streaming");
                sha.input(epsilon.to_le_bytes())
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum OutliersSpec {
    Fixed(usize),
    Percentage(f64),
}

impl Sha for OutliersSpec {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        match &self {
            Self::Fixed(x) => sha.input(x.to_le_bytes()),
            Self::Percentage(x) => sha.input(x.to_le_bytes()),
        }
    }
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

impl Sha for ParallelConfiguration {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        if let Some(pid) = self.process_id {
            pid.update_sha(sha);
        }
        if let Some(hosts) = self.hosts.as_ref() {
            hosts.update_sha(sha)
        }
        self.threads.update_sha(sha);
    }
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

            println!("spawning executable {:?}", remote_exec);
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

            println!("waiting for the workers");
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
    pub shuffle_seed: u64,
    pub outliers: OutliersSpec,
    pub algorithm: AlgorithmConfig,
    pub dataset: PathBuf,
    pub constraint: Constraint,
    pub parallel: Option<ParallelConfiguration>,
}

impl Sha for Configuration {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        self.shuffle_seed.update_sha(sha);
        self.outliers.update_sha(sha);
        self.algorithm.update_sha(sha);
        self.dataset.to_str().unwrap().to_owned().update_sha(sha);
        self.constraint.update_sha(sha);
        if let Some(parallel) = self.parallel.as_ref() {
            parallel.update_sha(sha);
        }
    }
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
    pub fn is_remote(&self) -> bool {
        if let Some(parallel) = &self.parallel {
            parallel.process_id.is_some()
        } else {
            false
        }
    }

    pub fn load(spec: String) -> anyhow::Result<Self> {
        let path = PathBuf::from(&spec);
        let config: Configuration = if path.is_file() {
            serde_json::from_reader(std::fs::File::open(path)?)?
        } else {
            let decoded_str = String::from_utf8(base64::decode(spec)?)?;
            serde_json::from_str(&decoded_str)?
        };

        // // Validate the constraint against the input
        // this would break if not all the workers have a copy of the dataset
        // let dataset_meta = Dataset::new(&config.dataset).metadata()?;
        // if !config
        //     .constraint
        //     .is_compatible_with(&dataset_meta.constraint)
        // {
        //     anyhow::bail!("incompatible constraints");
        // };

        Ok(config)
    }

    pub fn datatype(&self) -> anyhow::Result<Datatype> {
        println!("Reading datatype from {:?}", self.dataset);
        let meta = Dataset::new(&self.dataset)
            .metadata()
            .context("reading datatype")?;
        println!("Metadata is is {:?}", meta);
        Ok(meta.datatype)
    }

    pub fn dataset_metadata(&self) -> anyhow::Result<Metadata> {
        Dataset::new(&self.dataset)
            .metadata()
            .context("reading metadata")
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

impl Configure for Phone {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => Rc::new(PartitionMatroid::new(
                HashMap::from_iter(categories.clone().into_iter()),
            )),
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

impl Configure for Higgs {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => Rc::new(PartitionMatroid::new(
                HashMap::from_iter(categories.clone().into_iter()),
            )),
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

impl Configure for ColorVector {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => Rc::new(PartitionMatroid::new(
                HashMap::from_iter(categories.clone().into_iter()),
            )),
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

impl Configure for WikiPage {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Rc::new(TransversalMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

impl Configure for WikiPageEuclidean {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Transversal { topics } => Rc::new(TransversalMatroid::new(topics.clone())),
            _ => panic!("Can only build a transversal matroid constraint for WikiPage"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
            _ => panic!("Cannot run algorithm in parallel"),
        }
    }
}

impl Configure for Song {
    fn configure_constraint(conf: &Configuration) -> Rc<dyn Matroid<Self>> {
        match &conf.constraint {
            Constraint::Partition { categories } => Rc::new(PartitionMatroid::new(
                HashMap::from_iter(categories.clone().into_iter()),
            )),
            _ => panic!("Can only build a partition matroid constraint for Song"),
        }
    }
    fn configure_algorithm_info(conf: &Configuration) -> Box<dyn Algorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
        }
    }
    fn configure_sequential_algorithm(conf: &Configuration) -> Box<dyn SequentialAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::Greedy => Box::new(GreedyHeuristic::default()),
            AlgorithmConfig::ChenEtAl => Box::new(ChenEtAl::default()),
            AlgorithmConfig::Random { seed } => Box::new(RandomClustering::new(seed)),
            AlgorithmConfig::SeqCoreset { tau } => Box::new(SeqCoreset::new(tau)),
            AlgorithmConfig::StreamingCoreset { tau } => Box::new(StreamingCoreset::new(tau)),
            AlgorithmConfig::KaleStreaming { epsilon } => Box::new(KaleStreaming::new(epsilon)),
            AlgorithmConfig::MapReduceCoreset { .. } => panic!("Cannot run MapReduce sequentially"),
            AlgorithmConfig::MapReduceCoresetRec { .. } => {
                panic!("Cannot run MapReduce sequentially")
            }
        }
    }
    fn configure_parallel_algorithm(conf: &Configuration) -> Box<dyn ParallelAlgorithm<Self>> {
        match conf.algorithm {
            AlgorithmConfig::MapReduceCoreset { tau } => Box::new(MapReduceCoreset::new(tau)),
            AlgorithmConfig::MapReduceCoresetRec { tau } => Box::new(MapReduceCoresetRec::new(tau)),
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
    pub name: String,
    pub port: String,
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

impl Sha for Host {
    fn update_sha<D: Digest>(&self, sha: &mut D) {
        sha.input(self.name.as_bytes());
        sha.input(self.port.as_bytes());
    }
}
