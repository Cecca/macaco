use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{weighted_matroid_intersection, Matroid, Weight},
    perf_counters,
    types::OrderedF32,
};
use kcmkc_base::{matroid::augment, types::Distance};
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

    #[cfg(debug_assertions)]
    println!("Weights are {:?}", weight_map);

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
    let mut weight_uncovered: u32 = points
        .iter()
        .enumerate()
        .map(|(i, _)| weight_map.weight_of(i))
        .sum();
    // The following invariant should hold in any iteration
    assert!(
        weight_uncovered
            == assignment
                .iter()
                .enumerate()
                .filter(|(_i, a)| a.is_none())
                .map(|(i, _)| weight_map.weight_of(i))
                .sum()
    );

    // debug!("  Build disks");
    while weight_uncovered > 0 {
        // Get the center covering the largest weight
        let c = (0..n)
            .into_par_iter()
            .max_by_key(|i| {
                distances
                    .disk(*i, r)
                    .filter(|j| assignment[*j].is_none())
                    .map(|j| weight_map.weight_of(j))
                    .sum::<u32>()
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
            assert!(assignment[*j].is_none());
            assignment[*j].replace(c);
            weight_uncovered -= weight_map.weight_of(*j);
        }
        // Check the invariant
        assert!(
            weight_uncovered
                == assignment
                    .iter()
                    .enumerate()
                    .filter(|(_i, a)| a.is_none())
                    .map(|(i, _)| weight_map.weight_of(i))
                    .sum()
        );

        centers.push((c, expanded_disk));
    }

    #[cfg(debug_assertions)]
    {
        let _centers: Vec<V> = centers.iter().map(|c| points[c.0].clone()).collect();
        let (_radius_outliers, _radius_full) = radii(points, weight_map, &_centers, p as u32);
        println!(
            " . Disk centers: {}/{} radius outliers {} radius full {}",
            centers.len(),
            points.len(),
            _radius_outliers,
            _radius_full,
        );
    }

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
        println!(
            "    Covered nodes {} < {}, invalid solution with {} centers",
            covered_nodes,
            p,
            solution.len()
        );
        return Err(covered_nodes);
    } else {
        println!(
            "    Covered nodes {} >= {}, valid solution with {} centers",
            covered_nodes,
            p,
            solution.len()
        );
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

    #[cfg(debug_assertions)]
    {
        let (_radius_outliers, _radius_full) = radii(points, weight_map, &centers, p as u32);
        println!(
            " -> radius adjusted: {}/{} radius outliers {} radius full {}",
            centers.len(),
            points.len(),
            _radius_outliers,
            _radius_full,
        );
        println!(
            "centers: {:?}",
            solution
                .iter()
                .map(|disk| disk.center)
                .collect::<Vec<usize>>()
        );
    }

    Ok(centers)
}

fn radii<V: Distance, W: WeightMap>(
    points: &[V],
    weight_map: &W,
    centers: &[V],
    p: u32,
) -> (f32, f32) {
    let mut _dists: Vec<(OrderedF32, u32)> = points
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            centers
                .iter()
                .map(move |c| (OrderedF32(c.distance(p)), weight_map.weight_of(i)))
        })
        .collect();
    _dists.sort();
    let mut cnt = 0;
    let radius_outliers = _dists
        .iter()
        .skip_while(|(_, w)| {
            cnt += w;
            cnt < p
        })
        .next()
        .unwrap()
        .0;
    let radius_full = _dists.last().unwrap().0;

    (radius_outliers.0, radius_full.0)
}

/// Allows to attach arbitrary weights to points of set
pub trait WeightMap: Debug + Send + Sync {
    fn weight_of(&self, i: usize) -> u32;
}

#[derive(Debug)]
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

impl std::fmt::Debug for VecWeightMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total: u32 = self.weights.iter().sum();
        write!(
            f,
            "VecWeightMap {{ total: {}, weights: {:?} }}",
            total, self.weights
        )
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
