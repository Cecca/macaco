use crate::configuration::Sha;
use crate::configuration::*;
use anyhow::{Context, Result};
use chrono::prelude::*;
use macaco_base::{
    dataset::Datatype,
    types::{ColorVector, Higgs, Phone, Song, WikiPage, WikiPageEuclidean},
};
use rusqlite::*;
use std::path::PathBuf;
use std::time::Duration;

struct Outcome {
    pub total_time: Duration,
    pub radius: f32,
    pub num_centers: u32,
}

struct CoresetInfo {
    pub size: usize,
    pub radius: f64,
}

struct Counters {
    pub distance: u64,
    pub oracle: u64,
}

pub struct Reporter {
    db_path: PathBuf,
    date: DateTime<Utc>,
    config: Configuration,
    outcome: Option<Outcome>,
    coreset_info: Option<CoresetInfo>,
    counters: Option<Counters>,
    profile: Option<(Duration, Duration)>,
}

impl Reporter {
    pub fn from_config(config: Configuration) -> Self {
        println!("creating reporter from configuration");
        Self {
            db_path: Self::default_db_path(),
            date: Utc::now(),
            config,
            outcome: None,
            coreset_info: None,
            profile: None,
            counters: None,
        }
    }

    fn default_db_path() -> std::path::PathBuf {
        #[allow(deprecated)]
        let mut path = std::env::home_dir().expect("unable to get home directory");
        path.push("macaco-results.sqlite");
        path
    }

    pub fn set_outcome(&mut self, total_time: Duration, radius: f32, num_centers: u32) {
        self.outcome.replace(Outcome {
            total_time,
            radius,
            num_centers,
        });
    }

    pub fn set_counters(&mut self, (distance, oracle): (u64, u64)) {
        self.counters.replace(Counters { distance, oracle });
    }

    pub fn set_profile(&mut self, profile: (Duration, Duration)) {
        self.profile.replace(profile);
    }

    pub fn set_coreset_info(&mut self, size: usize, radius: f64) {
        self.coreset_info.replace(CoresetInfo { size, radius });
    }

    fn get_conn(&self) -> Result<Connection> {
        let dbpath = &self.db_path;
        let conn = Connection::open(dbpath).context("error connecting to the database")?;
        db_migrate(&conn).context("error running migrations")?;
        if !dbpath.is_file() {
            anyhow::bail!("migration did not setup database {:?}", dbpath);
        }
        Ok(conn)
    }

    pub fn already_run(&self) -> Result<Option<i64>> {
        let (algorithm, algorithm_version) = match self.config.datatype()? {
            Datatype::WikiPage => {
                let algo = WikiPage::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
            Datatype::WikiPageEuclidean => {
                let algo = WikiPageEuclidean::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
            Datatype::Song => {
                let algo = Song::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
            Datatype::ColorVector => {
                let algo = ColorVector::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
            Datatype::Higgs => {
                let algo = Higgs::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
            Datatype::Phone => {
                let algo = Phone::configure_algorithm_info(&self.config);
                (algo.name(), algo.version())
            }
        };

        println!("connecting to the database");
        let conn = self
            .get_conn()
            .context("error connecting to the database")?;
        println!("running query");
        conn.query_row(
            "SELECT id FROM result WHERE params_sha = ?1 AND algorithm = ?2 AND algorithm_version = ?3",
            params![self.config.sha()?, algorithm, algorithm_version],
            |row| Ok(row.get(0).expect("error getting id")),
        )
        .optional()
        .context("error running query")
    }

    pub fn save(self) -> Result<()> {
        let mut conn = self.get_conn()?;
        let (algorithm, algorithm_version, algorithm_params) = match self.config.datatype()? {
            Datatype::WikiPage => {
                let algo = WikiPage::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
            Datatype::WikiPageEuclidean => {
                let algo = WikiPageEuclidean::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
            Datatype::Song => {
                let algo = Song::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
            Datatype::ColorVector => {
                let algo = ColorVector::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
            Datatype::Higgs => {
                let algo = Higgs::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
            Datatype::Phone => {
                let algo = Phone::configure_algorithm_info(&self.config);
                (algo.name(), algo.version(), algo.parameters())
            }
        };

        let metadata = self.config.dataset_metadata()?;
        let dataset = &metadata.name;
        let dataset_version = metadata.version;
        let dataset_params = metadata.parameters_string();

        if let Some(outcome) = self.outcome {
            let counters = self.counters.context("missing counters")?;
            let (hosts, threads) = if let Some(parallel) = self.config.parallel.as_ref() {
                let hosts = if let Some(hosts) = parallel.hosts.as_ref() {
                    let mut hosts = hosts.clone();
                    hosts.sort();
                    let hosts: Vec<String> = hosts.into_iter().map(|h| h.to_string()).collect();
                    Some(hosts.join(" "))
                } else {
                    None
                };
                let threads = parallel.threads as u32;
                (hosts, Some(threads))
            } else {
                (None, None)
            };

            let tx = conn.transaction()?;
            tx.execute_named(
                "INSERT INTO result_raw (
                    code_version, date, hosts, threads, params_sha, outliers_spec,
                    algorithm, algorithm_params, algorithm_version,
                    dataset, dataset_params, dataset_version,
                    shuffle_seed,
                    constraint_params,
                    total_time_ms,
                    coreset_time_ms,
                    solution_time_ms,
                    distance_cnt,
                    oracle_cnt,
                    radius,
                    num_centers,
                    coreset_size,
                    coreset_radius
                ) VALUES (
                    :code_version, :date, :hosts, :threads, :params_sha, :outliers_spec,
                    :algorithm, :algorithm_params, :algorithm_version,
                    :dataset, :dataset_params, :dataset_version,
                    :shuffle_seed,
                    :constraint_params,
                    :total_time_ms,
                    :coreset_time_ms,
                    :solution_time_ms,
                    :distance_cnt,
                    :oracle_cnt,
                    :radius,
                    :num_centers,
                    :coreset_size,
                    :coreset_radius
                )",
                named_params! {
                    ":code_version": env!("VERGEN_GIT_SHA"),
                    ":date": self.date.to_rfc3339(),
                    ":hosts": hosts,
                    ":threads": threads,
                    ":params_sha": self.config.sha()?,
                    ":outliers_spec": self.config.outliers.describe(),
                    ":algorithm": algorithm,
                    ":algorithm_params": algorithm_params,
                    ":algorithm_version": algorithm_version,
                    ":dataset": dataset,
                    ":dataset_params": dataset_params,
                    ":dataset_version": dataset_version,
                    ":shuffle_seed": self.config.shuffle_seed as i64,
                    ":constraint_params": serde_json::to_string(&self.config.constraint)?,
                    ":total_time_ms": outcome.total_time.as_millis() as i64,
                    ":coreset_time_ms": self.profile.as_ref().unwrap().0.as_millis() as i64,
                    ":solution_time_ms": self.profile.as_ref().unwrap().1.as_millis() as i64,
                    ":distance_cnt": counters.distance as i64,
                    ":oracle_cnt": counters.oracle as i64,
                    ":radius": outcome.radius as f64,
                    ":num_centers": outcome.num_centers,
                    ":coreset_size": self.coreset_info.as_ref().map(|ci| ci.size as u32),
                    ":coreset_radius": self.coreset_info.as_ref().map(|ci| ci.radius),
                },
            )?;

            tx.commit()?;
        } else {
            anyhow::bail!("no outcome registered");
        }

        conn.close().map_err(|e| e.1).context("closing database")
    }
}

fn db_migrate(conn: &Connection) -> Result<()> {
    let version: u32 = conn
        .query_row(
            "SELECT user_version FROM pragma_user_version",
            params![],
            |row| row.get(0),
        )
        .context("cannot get version of the database")?;

    if version < 1 {
        conn.execute_batch(include_str!("migrations/v1.sql"))
            .context("error applying version 1")?;
    }
    if version < 2 {
        conn.execute_batch(include_str!("migrations/v2.sql"))
            .context("error applying version 2")?;
    }

    Ok(())
}
