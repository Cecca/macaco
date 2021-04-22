use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{weighted_matroid_intersection, Matroid, Weight},
    perf_counters,
};
use kcmkc_base::{matroid::augment, types::Distance};
use log::*;
use rayon::prelude::*;
use std::rc::Rc;
use std::{
    fmt::Debug,
    time::{Duration, Instant},
};

use crate::{disks::*, SequentialAlgorithm};

#[derive(Default)]
pub struct ChenEtAl {
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
}

impl<T: Distance + Clone + Debug + PartialEq> Algorithm<T> for ChenEtAl {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("ChenEtAl")
    }

    fn parameters(&self) -> String {
        String::new()
    }

    fn coreset(&self) -> Option<Vec<T>> {
        None
    }

    fn time_profile(&self) -> (Duration, Duration) {
        self.profile.clone().unwrap()
    }

    fn counters(&self) -> (u64, u64) {
        self.counters.clone().unwrap()
    }
}

impl<T: Distance + Clone + Debug + PartialEq + Sync> SequentialAlgorithm<T> for ChenEtAl {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        let start = Instant::now();
        let solution = robust_matroid_center(dataset, matroid, p, &UnitWeightMap);
        let elapsed = start.elapsed();
        println!("centers computed in {:?}", elapsed);
        self.profile.replace((Duration::from_secs(0), elapsed));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));
        Ok(solution)
    }
}

pub fn robust_matroid_center<'a, V: Distance + Clone + PartialEq + Sync, W: WeightMap>(
    points: &'a [V],
    matroid: Rc<dyn Matroid<V>>,
    p: usize,
    weight_map: &W,
) -> Vec<V> {
    let distances = DiskBuilder::new(points);

    let centers = distances.bynary_search_distances(|d| {
        run_robust_matroid_center(points, Rc::clone(&matroid), d, p, &distances, weight_map)
    });
    println!("Found {} centers", centers.len());

    if !matroid.is_maximal(&centers, points) {
        println!("the returned set of centers is not maximal: extend it to maximality");
        augment(Rc::clone(&matroid), &centers, points)
    } else {
        println!("Returning centers as is");
        centers
    }
}

/// Returns a triplet of centers, number of uncovered nodes, and an
/// iterator of optional assignments.
fn run_robust_matroid_center<'a, V: Distance + Clone, W: WeightMap>(
    points: &'a [V],
    matroid: Rc<dyn Matroid<V>>,
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

    // debug!("  Build disks");
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
    println!(" . Disk centers: {}", centers.len());

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

    // debug!("    Compute weighted matroid intersection");
    let m1 = DiskMatroid1::new(Rc::clone(&matroid), points);
    let m2 = DiskMatroid2;
    let solution: Vec<&ExpandedDisk<W>> =
        weighted_matroid_intersection(&vertex_disk_pairs, &m1, &m2).collect();
    assert!(m1.is_independent_ref(&solution));
    assert!(m2.is_independent_ref(&solution));
    let covered_nodes: usize = solution.iter().map(|disk| disk.weight() as usize).sum();
    if covered_nodes < p {
        println!("    Covered nodes {} < {}, solution with {} centers", covered_nodes, p, solution.len());
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
    assert!(matroid.is_independent(&centers));

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
    inner: Rc<dyn Matroid<T>>,
    base_set: &'a [T],
}

impl<'a, T: Clone> DiskMatroid1<'a, T> {
    fn new(inner: Rc<dyn Matroid<T>>, base_set: &'a [T]) -> Self {
        Self { inner, base_set }
    }
}

impl<'a, T: Clone, W: WeightMap> Matroid<ExpandedDisk<'a, W>> for DiskMatroid1<'a, T> {
    fn is_independent(&self, _set: &[ExpandedDisk<W>]) -> bool {
        todo!()
    }

    fn is_independent_ref(&self, set: &[&ExpandedDisk<W>]) -> bool {
        // First, we need to check if the identifiers are all distinct
        let ids: std::collections::BTreeSet<usize> = set.iter().map(|disk| disk.center).collect();
        if ids.len() != set.len() {
            return false;
        }

        let elements_set: Vec<&T> = set.iter().map(|disk| &self.base_set[disk.center]).collect();
        self.inner.is_independent_ref(&elements_set)
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }
}

struct DiskMatroid2;

impl<'a, W: WeightMap> Matroid<ExpandedDisk<'a, W>> for DiskMatroid2 {
    fn is_independent(&self, _set: &[ExpandedDisk<W>]) -> bool {
        todo!()
    }

    fn is_independent_ref(&self, set: &[&ExpandedDisk<W>]) -> bool {
        let disks: std::collections::BTreeSet<&Vec<usize>> =
            set.iter().map(|disk| disk.points).collect();
        disks.len() == set.len()
    }

    fn rank(&self) -> usize {
        unimplemented!()
    }
}
