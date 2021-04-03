use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use rand::{prelude::SliceRandom, SeedableRng};
use rand_xorshift::XorShiftRng;
use rmp_serde::decode::Error;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::{collections::HashMap, io::BufReader};

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Integer(u32),
    Float(f64),
}

impl MetadataValue {
    fn as_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Integer(i) => format!("{}", i),
            Self::Float(f) => format!("{}", f),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum Constraint {
    #[serde(rename = "transversal")]
    Transversal { topics: Vec<u32> },
    #[serde(rename = "partition")]
    Partition { categories: HashMap<String, u32> },
}

impl Constraint {
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Transversal {
                    topics: topics_self,
                },
                Self::Transversal {
                    topics: topics_other,
                },
            ) => topics_self
                .iter()
                .all(|t_self| topics_other.contains(t_self)),
            (
                Self::Partition {
                    categories: categories_self,
                },
                Self::Partition {
                    categories: categories_other,
                },
            ) => categories_self.iter().all(|(cat, limit)| {
                categories_other
                    .get(cat)
                    .map(|limit_other| limit <= limit_other)
                    .unwrap_or(false)
            }),
            _ => false,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Transversal { topics } => {
                let mut topics = topics.clone();
                topics.sort();
                let topics: Vec<String> = topics.into_iter().map(|x| format!("{}", x)).collect();
                format!("Transversal({})", topics.join(", "))
            }
            Self::Partition { categories } => {
                let categories: std::collections::BTreeMap<String, String> = categories
                    .iter()
                    .map(|(cat, cnt)| (cat.clone(), format!("{}", cnt)))
                    .collect();
                let categories: Vec<String> = categories
                    .iter()
                    .map(|(cat, cnt)| format!("{}={}", cat, cnt))
                    .collect();
                format!("Partition({})", categories.join(", "))
            }
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct Metadata {
    pub version: u32,
    pub name: String,
    pub datatype: Datatype,
    pub constraint: Constraint,
    pub parameters: HashMap<String, MetadataValue>,
}

impl Metadata {
    pub fn parameters_string(&self) -> String {
        let parameters: std::collections::BTreeMap<String, String> = self
            .parameters
            .iter()
            .map(|(k, v)| (k.clone(), v.as_string()))
            .collect();
        let parameters: Vec<String> = parameters
            .iter()
            .map(|(k, v)| format!("\"{}\": \"{}\"", k, v))
            .collect();
        format!("{{ {} }}", parameters.join(", "))
    }
}

#[derive(Deserialize, Debug)]
pub enum Datatype {
    WikiPage,
    Song,
}

#[derive(Deserialize, Debug)]
pub struct Dataset {
    path: PathBuf,
}

impl Dataset {
    pub fn new<I: Into<PathBuf>>(path: I) -> Self {
        Self { path: path.into() }
    }
    pub fn metadata(&self) -> Result<Metadata> {
        let file = BufReader::new(std::fs::File::open(&self.path)?);
        let input = GzDecoder::new(file);
        rmp_serde::from_read(input).context("reading metadata")
    }
    pub fn datatype(&self) -> Result<Datatype> {
        let meta = self.metadata()?;
        Ok(meta.datatype)
    }
    pub fn for_each<T, F: FnMut(u32, T)>(&self, mut f: F) -> Result<()>
    where
        for<'de> T: Deserialize<'de>,
    {
        let file = BufReader::new(std::fs::File::open(&self.path)?);
        let mut input = GzDecoder::new(file);
        // Skip metadata
        let _meta: Metadata = rmp_serde::from_read(&mut input).context("reading metadata")?;
        let mut cnt = 0;
        loop {
            match rmp_serde::from_read::<_, T>(&mut input) {
                Ok(item) => f(cnt, item),
                Err(Error::InvalidDataRead(_)) | Err(Error::InvalidMarkerRead(_)) => return Ok(()),
                Err(e) => bail!(e),
            }
            cnt += 1;
        }
    }

    pub fn size<T>(&self) -> Result<usize>
    where
        for<'de> T: Deserialize<'de>,
    {
        let mut cnt = 0;
        self.for_each(|_, _: T| cnt += 1)?;
        Ok(cnt)
    }

    pub fn to_vec<T>(&self, shuffle_seed: Option<u64>) -> Result<Vec<T>>
    where
        for<'de> T: Deserialize<'de>,
    {
        println!("Loading dataset into vector");
        let mut result = Vec::new();
        let mut pl = progress_logger::ProgressLogger::builder()
            .with_items_name("items")
            .start();
        self.for_each(|_, item| {
            result.push(item);
            pl.update(1u64);
        })?;
        pl.stop();
        if let Some(seed) = shuffle_seed {
            let mut rng = XorShiftRng::seed_from_u64(seed);
            result.shuffle(&mut rng);
        }

        Ok(result)
    }
}
