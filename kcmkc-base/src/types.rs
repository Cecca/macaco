use crate::matroid::*;
use abomonation_derive::Abomonation;
use serde::Deserialize;

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
#[derive(Debug, Abomonation, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f32>,
    norm: f32,
    norm_squared: f32,
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
        let data = Vec::from(&data[..4]);
        let norm_squared: f32 = data.iter().map(|x| x * x).sum::<f32>();
        let norm: f32 = norm_squared.sqrt();
        Self {
            data,
            norm,
            norm_squared,
        }
    }

    /// Compute the inner product between two vectors.
    pub fn inner_product(&self, other: &Self) -> f32 {
        let n = self.data.len();
        let mut sum = 0.0;
        for i in 0..n {
            sum += unsafe { self.data.get_unchecked(i) * other.data.get_unchecked(i) };
        }
        sum
        // self.data
        //     .iter()
        //     .zip(other.data.iter())
        //     .map(|(x, y)| x * y)
        //     .sum()
    }

    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let angle = self.inner_product(other) / (self.norm * other.norm);
        if angle >= 1.0 {
            // This branch is to handle some cases where precision errors make
            // the ratio computed above larger than one
            0.0
        } else {
            angle.acos() * std::f32::consts::FRAC_1_PI
        }
    }

    pub fn squared_euclidean_distance(&self, other: &Self) -> f32 {
        self.norm_squared + other.norm_squared - 2.0 * self.inner_product(other)
    }
}

#[derive(Debug, Deserialize, Abomonation, Clone, PartialEq)]
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
        let angle = self.inner_product(other) / (self.norm() * other.norm());
        if angle > 1.0 {
            // hadle rounding errors
            0.0
        } else {
            angle.acos() * std::f32::consts::FRAC_1_PI
        }
    }
}

pub trait Distance {
    fn distance(&self, other: &Self) -> f32;

    fn set_distance<'a, I: IntoIterator<Item = &'a Self>>(&'a self, set: I) -> (usize, OrderedF32) {
        set.into_iter()
            .enumerate()
            .map(|(i, other)| (i, OrderedF32(self.distance(other))))
            .min_by_key(|pair| pair.1)
            .unwrap()
    }
}

fn unit_weight() -> u32 {
    1
}

#[derive(Deserialize, Debug, Abomonation, Clone, PartialEq)]
pub struct ColorVector {
    pub color: String,
    pub vector: Vector,
    #[serde(skip, default = "unit_weight")]
    pub weight: u32,
}

impl Distance for ColorVector {
    /// ColorVectors use the euclidean distance
    fn distance(&self, other: &Self) -> f32 {
        self.vector.norm_squared + other.vector.norm_squared
            - 2.0 * self.vector.inner_product(&other.vector)
    }
}

impl PartitionMatroidElement for ColorVector {
    fn category(&self) -> &String {
        &self.color
    }
}

impl Weight for ColorVector {
    fn weight(&self) -> u32 {
        self.weight
    }
}

/// A page of wikipedia, represented as a d-dimensional vector,
/// with a set of topics.
#[derive(Deserialize, Debug, Abomonation, Clone, PartialEq)]
pub struct WikiPage {
    pub id: u32,
    pub title: String,
    pub vector: Vector,
    pub topics: Vec<u32>,
    #[serde(skip, default = "unit_weight")]
    pub weight: u32,
}

impl WikiPage {
    pub fn distance(&self, other: &Self) -> f32 {
        self.vector.cosine_distance(&other.vector)
    }
}

impl TransversalMatroidElement for WikiPage {
    fn topics<'a>(&'a self) -> &'a [u32] {
        &self.topics
    }
}

impl Weight for WikiPage {
    fn weight(&self) -> u32 {
        self.weight
    }
}

/// A page of wikipedia, represented as a d-dimensional vector,
/// with a set of topics.
#[derive(Deserialize, Debug, Abomonation, Clone, PartialEq)]
pub struct WikiPageEuclidean {
    pub id: u32,
    pub title: String,
    pub vector: Vector,
    pub topics: Vec<u32>,
    #[serde(skip, default = "unit_weight")]
    pub weight: u32,
}

impl TransversalMatroidElement for WikiPageEuclidean {
    fn topics<'a>(&'a self) -> &'a [u32] {
        &self.topics
    }
}

impl Weight for WikiPageEuclidean {
    fn weight(&self) -> u32 {
        self.weight
    }
}

impl Distance for WikiPageEuclidean {
    fn distance(&self, other: &Self) -> f32 {
        // perf_counters::inc_distance_count();
        self.vector.squared_euclidean_distance(&other.vector)
    }
}

#[derive(Deserialize, Debug, Abomonation, Clone, PartialEq)]
pub struct Song {
    pub track_id: String,
    pub genre: String,
    pub vector: SparseVector,
    #[serde(skip, default = "unit_weight")]
    pub weight: u32,
}

impl Distance for WikiPage {
    fn distance(&self, other: &Self) -> f32 {
        // perf_counters::inc_distance_count();
        self.vector.cosine_distance(&other.vector)
    }
}

impl Distance for Song {
    fn distance(&self, other: &Self) -> f32 {
        // perf_counters::inc_distance_count();
        self.vector.cosine_distance(&other.vector)
    }
}

impl PartitionMatroidElement for Song {
    fn category<'a>(&'a self) -> &'a String {
        &self.genre
    }
}

impl Weight for Song {
    fn weight(&self) -> u32 {
        self.weight
    }
}

// Some utility types

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug, Abomonation)]
pub struct OrderedF32(pub f32);

impl Eq for OrderedF32 {}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or_else(|| panic!("self: {:?} other: {:?}", self, other))
    }
}

impl Into<OrderedF32> for f32 {
    fn into(self) -> OrderedF32 {
        OrderedF32(self)
    }
}

impl Into<f32> for OrderedF32 {
    fn into(self) -> f32 {
        self.0
    }
}
