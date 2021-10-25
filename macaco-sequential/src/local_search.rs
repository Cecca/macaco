use std::{iter::FromIterator, rc::Rc};

use macaco_base::{
    matroid::{Matroid, Weight},
    types::*,
};
use rayon::prelude::*;

use crate::{chen_et_al::WeightMap, disks::DiskBuilder};

pub fn local_search<'a, T: Clone + Distance + Sync, W: WeightMap>(
    dataset: &'a [T],
    matroid: std::rc::Rc<dyn macaco_base::matroid::Matroid<T>>,
    p: usize,
    weight_map: &'a W,
) -> anyhow::Result<Vec<T>> {
    let refs = Vec::from_iter(dataset.iter());
    let rank = matroid.maximal_independent_set(&refs).len();
    println!("Running k-center with outliers with k={}", rank);
    let distances = DiskBuilder::new(&dataset);
    let disks = k_outliers(dataset, rank, p, &distances, weight_map);
    let centers = disks
        .into_iter()
        .map(|disk| disk.center)
        .collect::<Vec<usize>>();

    // Now the disk centers are a good approximation in terms of radius, but they may not form
    // a maximal independent set. Therefore we run local search on them to adjust the
    // solution to a maximal independent set. Note that this migth fail.
    Ok(distances.bynary_search_distances(|d| {
        let candidates: Vec<Vec<usize>> = centers
            .iter()
            .map(|c| distances.disk(*c, d).collect::<Vec<usize>>())
            .collect();
        for cands in &candidates {
            let w: u32 = cands.iter().map(|i| weight_map.weight_of(*i)).sum();
            let ps = Vec::from_iter(cands.iter().map(|i| &dataset[*i]));
            let is = matroid.maximal_independent_set(&ps);
            println!(
                "maximal independent set of size {}/{} (w={})",
                is.len(),
                ps.len(),
                w
            )
        }

        build_independent_set(dataset, Rc::clone(&matroid), &candidates, 0, Vec::new())
            .map_err(|partial| partial.len())
    }))
}

fn build_independent_set<T: Clone>(
    dataset: &[T],
    matroid: Rc<dyn Matroid<T>>,
    candidates: &Vec<Vec<usize>>,
    i: usize,
    current: Vec<T>,
) -> Result<Vec<T>, Vec<T>> {
    assert!(
        current.len() == i,
        "was expecting {} elements, got {}",
        i,
        current.len()
    );
    assert!(matroid.is_independent(&current));
    if i == candidates.len() {
        // we have a solution
        return Ok(current);
    }
    for c in candidates[i].iter() {
        let mut tentative = current.clone();
        tentative.push(dataset[*c].clone());
        if matroid.is_independent(&tentative) {
            match build_independent_set(dataset, Rc::clone(&matroid), candidates, i + 1, tentative)
            {
                Ok(solution) => return Ok(solution),
                Err(_) => (),
            }
        }
    }
    Err(current)
}

fn k_outliers<'a, V: Distance + Clone, W: WeightMap>(
    points: &'a [V],
    k: usize,
    p: usize,
    distances: &DiskBuilder,
    weight_map: &'a W,
) -> Vec<ExpandedDisk<'a, W>> {
    let disks = distances
        .bynary_search_distances(|d| run_k_outliers(points, k, d, p, &distances, weight_map));
    println!("Found {} centers", disks.len());
    assert!(disks.len() == k);

    disks
}

fn run_k_outliers<'a, V: Distance + Clone, W: WeightMap>(
    points: &'a [V],
    k: usize,
    r: f32,
    p: usize,
    distances: &DiskBuilder,
    weight_map: &'a W,
) -> Result<Vec<ExpandedDisk<'a, W>>, usize> {
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
    while centers.len() < k {
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

    let solution: Vec<ExpandedDisk<W>> = centers
        .into_iter()
        .map(|(i, disk)| ExpandedDisk {
            center: i,
            points: disk,
            weights: weight_map,
        })
        .collect();

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

    Ok(solution)
}

#[allow(dead_code)]
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

struct ExpandedDisk<'a, W: WeightMap> {
    center: usize,
    points: Vec<usize>,
    weights: &'a W,
}

impl<'a, W: WeightMap> Weight for ExpandedDisk<'a, W> {
    fn weight(&self) -> u32 {
        self.points.iter().map(|i| self.weights.weight_of(*i)).sum()
    }
}

pub fn intersection<I1: Iterator<Item = usize>, I2: Iterator<Item = usize>>(
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
