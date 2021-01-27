use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::path::PathBuf;
use std::{collections::HashMap, io::BufReader};

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Integer(u32),
    Float(f64),
}

#[derive(Deserialize, Debug)]
pub enum Constraint {
    #[serde(rename = "transversal")]
    Transversal { topics: Vec<u32> },
}

#[derive(Deserialize, Debug)]
pub struct Metadata {
    version: u32,
    name: String,
    constraint: Constraint,
    parameters: HashMap<String, MetadataValue>,
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
}
