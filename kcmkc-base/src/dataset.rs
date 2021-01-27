use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use rmp_serde::decode::Error;
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
    pub version: u32,
    pub name: String,
    pub datatype: Datatype,
    pub constraint: Constraint,
    pub parameters: HashMap<String, MetadataValue>,
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

    pub fn to_vec<T>(&self) -> Result<Vec<T>>
    where
        for<'de> T: Deserialize<'de>,
    {
        let mut result = Vec::new();
        self.for_each(|_, item| {
            result.push(item);
        })?;
        Ok(result)
    }
}
