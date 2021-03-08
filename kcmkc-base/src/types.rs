use crate::matroid::*;
use abomonation_derive::Abomonation;
use serde::Deserialize;
use std::marker::PhantomData;

/// A vector in a d-dimensional space.
///
/// We store the norm in the vector, along with the data, to speed up some computations,
/// like the cosine distance. We can compute the
/// inner product between two vectors.
///
/// The underlying data type is f32 for two reasons:
///
///  - save space when transmitting over the network
///  - more efficient SIMD computations, since we can pack more operands
///
/// We manually implement deserialize in order to be able to compute the norm during de-serialization
#[derive(Debug, Abomonation, Clone)]
pub struct Vector {
    data: Vec<f32>,
    norm: f32,
}

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data: Vec<f32> = Vec::deserialize(deserializer)?;
        Ok(Self::new(data))
    }
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Self {
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { data, norm }
    }

    /// Compute the inner product between two vectors.
    pub fn inner_product(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(x, y)| x * y)
            .sum()
    }

    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let angle = self.inner_product(other) / (self.norm * other.norm);
        if angle >= 1.0 {
            // This branch is to handle some cases where precision errors make /
            // the ratio computed above larger than one
            0.0
        } else {
            angle.acos() * std::f32::consts::FRAC_1_PI
        }
    }
}

#[derive(Debug, Deserialize, Abomonation, Clone)]
pub struct SparseVector {
    #[serde(rename = "d")]
    pub dimension: u32,
    #[serde(rename = "c")]
    data: Vec<(u32, f32)>,
}

impl SparseVector {
    pub fn new(dimension: u32, data: Vec<(u32, f32)>) -> Self {
        Self { dimension, data }
    }

    pub fn norm(&self) -> f32 {
        self.data
            .iter()
            .map(|pair| pair.1 * pair.1)
            .sum::<f32>()
            .sqrt()
    }

    pub fn inner_product(&self, other: &Self) -> f32 {
        let mut s_iter = self.data.iter();
        let mut o_iter = other.data.iter();

        let mut sum = 0.0;

        let mut cur_s = s_iter.next();
        let mut cur_o = o_iter.next();
        loop {
            if cur_s.is_none() || cur_o.is_none() {
                return sum;
            }
            let s = cur_s.unwrap();
            let o = cur_o.unwrap();
            if s.0 < o.0 {
                cur_s = s_iter.next();
            } else if s.0 > o.0 {
                cur_o = o_iter.next();
            } else {
                sum += s.1 * o.1;
                cur_s = s_iter.next();
                cur_o = o_iter.next();
            }
        }
    }

    pub fn cosine_distance(&self, other: &Self) -> f32 {
        (self.inner_product(other) / (self.norm() * other.norm())).acos()
            * std::f32::consts::FRAC_1_PI
    }
}

pub trait Distance {
    fn distance(&self, other: &Self) -> f32;
}

/// A page of wikipedia, represented as a d-dimensional vector,
/// with a set of topics.
#[derive(Deserialize, Debug, Abomonation, Clone)]
pub struct WikiPage {
    pub id: u32,
    pub title: String,
    pub vector: Vector,
    pub topics: Vec<u32>,
}

impl WikiPage {
    pub fn distance(&self, other: &Self) -> f32 {
        self.vector.cosine_distance(&other.vector)
    }
}

impl TransveralMatroidElement for WikiPage {
    fn topics<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a> {
        let iter = self.topics.iter().copied();
        Box::new(iter) as Box<dyn Iterator<Item = u32>>
    }
}

#[derive(Deserialize, Debug, Abomonation, Clone)]
pub struct Song {
    pub track_id: String,
    pub genre: String,
    pub vector: SparseVector,
}

impl Distance for WikiPage {
    fn distance(&self, other: &Self) -> f32 {
        self.vector.cosine_distance(&other.vector)
    }
}

impl Distance for Song {
    fn distance(&self, other: &Self) -> f32 {
        self.vector.cosine_distance(&other.vector)
    }
}

impl PartitionMatroidElement for Song {
    fn category<'a>(&'a self) -> &'a String {
        &self.genre
    }
}

#[test]
fn test_wiki_matroid() {
    let set = vec![
        WikiPage {
            id: 1,
            topics: vec![9, 65, 70, 84, 97],
            title: String::from("a"),
            vector: Vector::new(vec![1.0]),
        },
        WikiPage {
            id: 2,
            topics: vec![8, 27, 45],
            title: String::from("b"),
            vector: Vector::new(vec![1.0]),
        },
        WikiPage {
            id: 3,
            topics: vec![1, 44, 97],
            title: String::from("c"),
            vector: Vector::new(vec![1.0]),
        },
        WikiPage {
            id: 4,
            topics: vec![9],
            title: String::from("d"),
            vector: Vector::new(vec![1.0]),
        },
        WikiPage {
            id: 5,
            topics: vec![0, 81],
            title: String::from("e"),
            vector: Vector::new(vec![1.0]),
        },
    ];

    let matroid: TransveralMatroid<WikiPage> =
        TransveralMatroid::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    // assert!(matroid.is_independent(&set));
}
