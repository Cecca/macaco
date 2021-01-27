use std::marker::PhantomData;

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
        self.inner_product(other).acos() * std::f32::consts::FRAC_1_PI
    }
}

pub trait Matroid<T>
where
    T: Clone,
{
    fn is_independent(&self, set: &[T]) -> bool;

    /// Implementation of the (unweighted) greedy algorithm
    /// for the maximal independent set:
    ///
    ///  - start with an empty set
    ///  - while there are elements to add
    ///    - add the next element to the set if this keeps the set independent
    fn maximal_independent_set(&self, set: &[T]) -> Vec<T> {
        let mut is = Vec::new();

        for x in set {
            is.push(x.clone());
            if !self.is_independent(&is) {
                is.pop();
            }
        }

        is
    }
}

/// The intersection matroid of two give matroids.
///
/// For efficiency, the suggestion is to give the most efficient
/// matroid to verify as the first one, since the independence
/// oracle short circuits.
pub struct IntersectionMatroid<T, A, B>
where
    T: Clone,
    A: Matroid<T>,
    B: Matroid<T>,
{
    a: A,
    b: B,
    _marker: PhantomData<T>,
}

impl<T, A, B> IntersectionMatroid<T, A, B>
where
    T: Clone,
    A: Matroid<T>,
    B: Matroid<T>,
{
    pub fn intersection(a: A, b: B) -> Self {
        Self {
            a,
            b,
            _marker: PhantomData,
        }
    }
}

impl<T, A, B> Matroid<T> for IntersectionMatroid<T, A, B>
where
    T: Clone,
    A: Matroid<T>,
    B: Matroid<T>,
{
    fn is_independent(&self, set: &[T]) -> bool {
        self.a.is_independent(set) && self.b.is_independent(set)
    }
}

/// Element of a set on which we can impose a transversal matroid
pub trait TransveralMatroidElement: Clone {
    fn topics<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a>;
}

pub struct TransveralMatroid<T> {
    topics: Vec<u32>,
    _marker: PhantomData<T>,
}

impl<T: TransveralMatroidElement> Matroid<T> for TransveralMatroid<T> {
    fn is_independent(&self, set: &[T]) -> bool {
        set.len() < self.topics.len() && self.maximum_matching(set).count() == set.len()
    }
}

impl<T: TransveralMatroidElement> TransveralMatroid<T> {
    // Return the indices of the elements in `set` that form a maximum matching
    // wrt the ground topics
    fn maximum_matching(&self, set: &[T]) -> impl Iterator<Item = usize> {
        let mut visited = vec![false; self.topics.len()];
        let mut representatives = vec![None; self.topics.len()];

        for idx in 0..set.len() {
            // reset the flags
            for flag in visited.iter_mut() {
                *flag = false;
            }
            // try to accomodate the new element
            self.find_matching_for(set, idx, &mut representatives, &mut visited);
        }

        representatives.into_iter().filter_map(|opt| opt)
    }

    fn find_matching_for(
        &self,
        set: &[T],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for (topic_idx, topic) in self.topics.iter().enumerate() {
            if set[idx].topics().find(|t| t == topic).is_some() && !visited[topic_idx] {
                visited[topic_idx] = true;
                let can_set = if let Some(displacing_idx) = representatives[topic_idx] {
                    // try to move the representative to another set
                    self.find_matching_for(set, displacing_idx, representatives, visited)
                } else {
                    true
                };

                if can_set {
                    representatives[topic_idx].replace(idx);
                    return true;
                }
            }
        }

        false
    }
}

/// Element of a set on which we can impose a partition matroid
pub trait PartitionMatroidElement: Clone {
    fn category<'a>(&'a self) -> usize;
}

pub struct PartitionMatroid<T: PartitionMatroidElement> {
    categories: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T: PartitionMatroidElement> Matroid<T> for PartitionMatroid<T> {
    fn is_independent(&self, set: &[T]) -> bool {
        let mut counts = self.categories.clone();
        for x in set {
            let cat = x.category();
            if counts[cat] == 0 {
                return false;
            } else {
                counts[cat] -= 1;
            }
        }
        true
    }
}

/// A page of wikipedia, represented as a d-dimensional vector,
/// with a set of topics.
#[derive(Deserialize, Debug, Abomonation, Clone)]
pub struct WikiPage {
    id: u32,
    vector: Vector,
    topics: Vec<u32>,
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
