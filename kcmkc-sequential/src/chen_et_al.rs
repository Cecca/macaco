use rayon::prelude::*;
use std::fmt::Debug;

use kcmkc_base::types::{Distance, OrderedF32};
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{weighted_matroid_intersection, Matroid, Weight},
};

pub struct ChenEtAl;

impl<T: Distance + Clone + Debug> Algorithm<T> for ChenEtAl {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("ChenEtAl")
    }

    fn parameters(&self) -> String {
        String::new()
    }

    fn run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Box<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        Ok(robust_matroid_center(dataset, matroid, p, &UnitWeightMap))
    }
}

// Pre-compute all pairwise distances, and then keep the distances from each point in sorted order.
// This allows the retrieval of disks in time proportional to the size of the disk itself.
struct DiskBuilder {
    distances: Vec<Vec<(usize, f32)>>,
}

impl DiskBuilder {
    fn new<V: Distance>(points: &[V]) -> Self {
        println!("Pre-computing distances");
        let distances: Vec<Vec<(usize, f32)>> = points
            .iter()
            .map(|a| {
                let mut dists: Vec<(usize, f32)> = points
                    .iter()
                    .enumerate()
                    .map(|(j, b)| (j, a.distance(b)))
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists
            })
            .collect();
        Self { distances }
    }

    /// Iterates through the distances in the matrix in sorted order.
    fn iter_distances(&self) -> impl Iterator<Item = f32> {
        use std::collections::BTreeSet;
        println!("Sorting distances to get candidate radii");
        let dists: BTreeSet<OrderedF32> = self
            .distances
            .iter()
            .flat_map(|row| row.iter().filter(|f| f.1 > 0.0).map(|f| f.1.into()))
            .collect();

        dists.into_iter().map(|wrapper| wrapper.into())
    }

    /// Get the indices of points in the ball of radius `r` around point `i`.
    /// The indices are not in sorted order, so to compute the intersection between disks
    /// in linear time we should first sort them!
    fn disk<'a>(&'a self, i: usize, r: f32) -> impl Iterator<Item = usize> + 'a {
        self.distances[i]
            .iter()
            .take_while(move |(_i, d)| *d <= r)
            .map(|pair| pair.0)
    }
}

fn intersection<I1: Iterator<Item = usize>, I2: Iterator<Item = usize>>(
    mut a: I1,
    mut b: I2,
) -> impl Iterator<Item = usize> {
    let mut next_a = a.next();
    let mut next_b = b.next();
    std::iter::from_fn(move || loop {
        if next_a.is_none() || next_b.is_none() {
            return None;
        }
        let idx_a = next_a.unwrap();
        let idx_b = next_b.unwrap();
        if idx_a < idx_b {
            next_a = a.next();
        } else if idx_a > idx_b {
            next_b = b.next();
        } else {
            let ret = idx_a;
            next_a = a.next();
            next_b = b.next();
            return Some(ret);
        }
    })
}

pub fn robust_matroid_center<'a, V: Distance + Clone, W: WeightMap>(
    points: &'a [V],
    matroid: Box<dyn Matroid<V>>,
    p: usize,
    weight_map: &W,
) -> Vec<V> {
    let distances = DiskBuilder::new(points);

    let distinct_distances: Vec<f32> = distances.iter_distances().skip(100).collect();
    let mut i = 1;

    while i < distinct_distances.len() {
        let r = distinct_distances[i];
        println!("Iteration with radius {} [i={}]", r, i);
        match run_robust_matroid_center(points, &matroid, r, p, &distances, weight_map) {
            Ok(centers) => {
                return centers;
            }
            Err(covered) => println!("covered only {} out of {}", covered, p),
        }
        i = std::cmp::min(distinct_distances.len() - 1, i * 2);
    }
    panic!("should never get here, the last radius we try should cover everything");
}

/// Returns a triplet of centers, number of uncovered nodes, and an
/// iterator of optional assignments.
fn run_robust_matroid_center<'a, V: Distance + Clone, W: WeightMap>(
    points: &'a [V],
    matroid: &Box<dyn Matroid<V>>,
    r: f32,
    p: usize,
    distances: &DiskBuilder,
    weight_map: &W,
) -> Result<Vec<V>, usize> {
    let n = points.len();
    // Mapping between points and the center they are assigned to
    let mut assignment: Vec<Option<usize>> = vec![None; points.len()];
    // The initial centers indices, which will eventually be remapped using the
    // intersection matroid, along with their expanded disks
    let mut centers: Vec<(usize, Vec<usize>)> = Vec::new();
    // The number of uncovered nodes
    let mut n_uncovered = n;
    // The following invariant should hold in any iteration
    debug_assert!(n_uncovered == assignment.iter().filter(|a| a.is_none()).count());

    println!("  Build disks");
    while n_uncovered > 0 {
        // Get the center covering the most uncovered points
        let c = (0..n)
            .into_par_iter()
            .max_by_key(|i| {
                distances
                    .disk(*i, r)
                    .filter(|j| assignment[*j].is_none())
                    .count()
            })
            .expect("max on an empty iterator");
        let mut expanded_disk: Vec<usize> = distances
            .disk(c, 3.0 * r)
            .filter(|j| assignment[*j].is_none())
            .collect();
        // Sort to be able to compute the intersection in linear time, later on.
        expanded_disk.sort_unstable();

        // Mark the nodes as covered
        for j in expanded_disk.iter() {
            debug_assert!(assignment[*j].is_none());
            assignment[*j].replace(c);
            n_uncovered -= 1;
        }
        // Check the invariant
        debug_assert!(n_uncovered == assignment.iter().filter(|a| a.is_none()).count());

        centers.push((c, expanded_disk));
    }

    println!("  Enforce matroid constraints");
    println!("    Building vertex disk pairs");
    // Build the candidate center/disk pairs. Disks are references
    // to the original ones, to avoind wasting space by duplicating them
    let vertex_disk_pairs: Vec<ExpandedDisk<W>> = (0..n)
        .flat_map(|v| {
            let disk: Vec<usize> = distances.disk(v, r).collect();
            centers
                .iter()
                .filter(move |(_, expanded_disk)| {
                    intersection(disk.iter().copied(), expanded_disk.iter().copied()).count() > 0
                })
                .map(move |(_, expanded_disk)| ExpandedDisk {
                    center: v,
                    points: expanded_disk,
                    weights: weight_map,
                })
        })
        .collect();

    println!("    Compute weighted matroid intersection");
    let m1 = DiskMatroid1::new(&matroid, points);
    let m2 = DiskMatroid2;
    let solution: Vec<&ExpandedDisk<W>> =
        weighted_matroid_intersection(&vertex_disk_pairs, &m1, &m2).collect();
    assert!(m1.is_independent(&solution));
    assert!(m2.is_independent(&solution));
    let covered_nodes: usize = solution.iter().map(|disk| disk.weight() as usize).sum();
    if covered_nodes < p {
        println!("    Covered nodes {} < {}", covered_nodes, p);
        return Err(covered_nodes);
    }

    assert!(
        covered_nodes
            <= (0..points.len())
                .map(|i| weight_map.weight_of(i))
                .sum::<u32>() as usize
    );
    let centers: Vec<V> = solution
        .iter()
        .map(|disk| points[disk.center].clone())
        .collect();

    Ok(centers)
}

/// Allows to attach arbitrary weights to points of set
pub trait WeightMap {
    fn weight_of(&self, i: usize) -> u32;
}

pub struct UnitWeightMap;

impl WeightMap for UnitWeightMap {
    fn weight_of(&self, _i: usize) -> u32 {
        1
    }
}

pub struct VecWeightMap {
    weights: Vec<u32>,
}

impl WeightMap for VecWeightMap {
    fn weight_of(&self, i: usize) -> u32 {
        self.weights[i]
    }
}

impl VecWeightMap {
    pub fn new(weights: Vec<u32>) -> Self {
        Self { weights }
    }
}

struct ExpandedDisk<'a, W: WeightMap> {
    center: usize,
    points: &'a Vec<usize>,
    weights: &'a W,
}

impl<'a, W: WeightMap> Weight for ExpandedDisk<'a, W> {
    fn weight(&self) -> u32 {
        self.points.iter().map(|i| self.weights.weight_of(*i)).sum()
    }
}

struct DiskMatroid1<'a, T: Clone> {
    inner: &'a Box<dyn Matroid<T>>,
    base_set: &'a [T],
}

impl<'a, T: Clone> DiskMatroid1<'a, T> {
    fn new(inner: &'a Box<dyn Matroid<T>>, base_set: &'a [T]) -> Self {
        Self { inner, base_set }
    }
}

impl<'a, T: Clone, W: WeightMap> Matroid<ExpandedDisk<'a, W>> for DiskMatroid1<'a, T> {
    fn is_independent(&self, set: &[&ExpandedDisk<W>]) -> bool {
        // First, we need to check if the identifiers are all distinct
        let ids: std::collections::BTreeSet<usize> = set.iter().map(|disk| disk.center).collect();
        if ids.len() != set.len() {
            return false;
        }

        let elements_set: Vec<&T> = set.iter().map(|disk| &self.base_set[disk.center]).collect();
        self.inner.is_independent(&elements_set)
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }
}

struct DiskMatroid2;

impl<'a, W: WeightMap> Matroid<ExpandedDisk<'a, W>> for DiskMatroid2 {
    fn is_independent(&self, set: &[&ExpandedDisk<W>]) -> bool {
        let disks: std::collections::BTreeSet<&Vec<usize>> =
            set.iter().map(|disk| disk.points).collect();
        disks.len() == set.len()
    }

    fn rank(&self) -> usize {
        unimplemented!()
    }
}
