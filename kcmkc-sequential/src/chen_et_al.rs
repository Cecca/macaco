use kcmkc_base::types::Distance;
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{weighted_matroid_intersection, Matroid, Weight},
};

pub struct ChenEtAl;

impl<T: Distance + Clone> Algorithm<T> for ChenEtAl {
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
    ) -> anyhow::Result<(
        Vec<&'a T>,
        usize,
        Box<dyn Iterator<Item = (&'a T, Option<(usize, f32)>)> + 'a>,
    )> {
        Ok(robust_matroid_center(dataset, matroid, p))
    }
}

/// Representation of a symmetric matrix as a _lower_ triangular matrix, to
/// save space
struct DistanceMatrix {
    distances: Vec<Vec<f32>>,
}

impl DistanceMatrix {
    fn new<V: Distance>(points: &[V]) -> Self {
        let distances: Vec<Vec<f32>> = points
            .iter()
            .enumerate()
            .map(|(i, a)| {
                points
                    .iter()
                    .enumerate()
                    // build a lower triangular matrix
                    .filter(|(j, _)| i >= *j)
                    .map(|(_, b)| a.distance(b))
                    .collect()
            })
            .collect();
        println!(
            "Allocated matrix of distances with {} rows",
            distances.len()
        );
        Self { distances }
    }

    fn distance(&self, i: usize, j: usize) -> f32 {
        if i >= j {
            self.distances[i][j]
        } else {
            self.distances[j][i]
        }
    }

    /// Iterates through the distances in the matrix in sorted order.
    fn iter_distances(&self) -> impl Iterator<Item = f32> {
        use std::collections::BTreeSet;
        let dists: BTreeSet<OrderedF32> = self
            .distances
            .iter()
            .flat_map(|row| row.iter().filter(|f| **f > 0.0).map(|f| OrderedF32(*f)))
            .collect();

        dists.into_iter().map(|wrapper| wrapper.0)
    }

    /// Get the indices of points in the ball of radius `r` around point `i`.
    /// The indices are in sorted order, so intersection between disks
    /// can be computed in linear time
    fn disk<'a>(&'a self, i: usize, r: f32) -> impl Iterator<Item = usize> + 'a {
        let n = self.distances.len();
        (0..n).filter(move |j| self.distance(i, *j) <= r)
    }
}

#[derive(PartialEq, PartialOrd)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
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

pub fn robust_matroid_center<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: Box<dyn Matroid<V>>,
    p: usize,
) -> (
    Vec<&'a V>,
    usize,
    Box<dyn Iterator<Item = (&'a V, Option<(usize, f32)>)> + 'a>,
) {
    let distances = DistanceMatrix::new(points);

    for r in distances.iter_distances().skip(2) {
        //}.skip_while(|f| *f < 0.01) {
        println!("Iteration with radius {}", r);
        match run_robust_matroid_center(points, &matroid, r, p, &distances) {
            Ok(triplet) => {
                return triplet;
            }
            Err(covered) => println!("covered only {} out of {}", covered, p),
        }
    }
    panic!("should never get here, the last radius we try should cover everything");
}

/// Returns a triplet of centers, number of uncovered nodes, and an
/// iterator of optional assignments.
fn run_robust_matroid_center<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: &Box<dyn Matroid<V>>,
    r: f32,
    p: usize,
    distances: &DistanceMatrix,
) -> Result<
    (
        Vec<&'a V>,
        usize,
        Box<dyn Iterator<Item = (&'a V, Option<(usize, f32)>)> + 'a>,
    ),
    usize,
> {
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
            .max_by_key(|i| {
                distances
                    .disk(*i, r)
                    .filter(|j| assignment[*j].is_none())
                    .count()
            })
            .expect("max on an empty iterator");
        let expanded_disk: Vec<usize> = distances
            .disk(c, 3.0 * r)
            .filter(|j| assignment[*j].is_none())
            .collect();

        // Mark the nodes as covered
        for &j in expanded_disk.iter() {
            debug_assert!(assignment[j].is_none());
            assignment[j].replace(c);
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
    let vertex_disk_pairs: Vec<(usize, &Vec<usize>)> = (0..n)
        .flat_map(|v| {
            let disk: Vec<usize> = distances.disk(v, r).collect();
            centers
                .iter()
                .filter(move |(_, expanded_disk)| {
                    intersection(disk.iter().copied(), expanded_disk.iter().copied()).count() > 0
                })
                .map(move |(_, expanded_disk)| (v, expanded_disk))
        })
        .collect();

    println!("    Compute weighted matroid intersection");
    let m1 = DiskMatroid1::new(&matroid, points);
    let m2 = DiskMatroid2;
    let solution: Vec<&(usize, &Vec<usize>)> =
        weighted_matroid_intersection(&vertex_disk_pairs, &m1, &m2).collect();
    let covered_nodes: usize = solution.iter().map(|p| p.1.len()).sum();
    if covered_nodes < p {
        println!("    Covered nodes {} < {}", covered_nodes, p);
        return Err(covered_nodes);
    }

    let uncovered_nodes = points.len() - covered_nodes;
    let centers: Vec<&V> = solution.iter().map(|p| &points[p.0]).collect();
    let mut assignments = vec![None; points.len()];
    for (c, cluster) in solution {
        for p in cluster.iter() {
            let d = distances.distance(*c, *p);
            assignments[*p] = Some((*c, d));
        }
    }
    let assignments = assignments
        .into_iter()
        .enumerate()
        .map(move |(i, assignment)| (&points[i], assignment));

    Ok((centers, uncovered_nodes, Box::new(assignments)))
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

impl<'a, T: Clone> Matroid<(usize, &Vec<usize>)> for DiskMatroid1<'a, T> {
    fn is_independent(&self, set: &[&(usize, &Vec<usize>)]) -> bool {
        // First, we need to check if the identifiers are all distinct
        let ids: std::collections::BTreeSet<usize> = set.iter().map(|p| p.0).collect();
        if ids.len() != set.len() {
            return false;
        }

        let elements_set: Vec<&T> = set.iter().map(|p| &self.base_set[p.0]).collect();
        self.inner.is_independent(&elements_set)
    }
}

struct DiskMatroid2;

impl Matroid<(usize, &Vec<usize>)> for DiskMatroid2 {
    fn is_independent(&self, set: &[&(usize, &Vec<usize>)]) -> bool {
        let disks: std::collections::BTreeSet<&Vec<usize>> = set.iter().map(|p| p.1).collect();
        disks.len() == set.len()
    }
}
