use crate::SequentialAlgorithm;
use macaco_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    perf_counters,
    types::Distance,
    types::OrderedF32,
};
use std::{
    collections::BTreeMap,
    rc::Rc,
    time::{Duration, Instant},
};

#[derive(Default)]
pub struct GreedyHeuristic {
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
}

impl<V: Distance + Clone + Weight + PartialEq> Algorithm<V> for GreedyHeuristic {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("Greedy")
    }

    fn parameters(&self) -> String {
        String::new()
    }

    fn coreset(&self) -> Option<Vec<V>> {
        None
    }

    fn time_profile(&self) -> (Duration, Duration) {
        self.profile.clone().unwrap()
    }

    fn memory_usage(&self) -> Option<usize> {
        None
    }

    fn counters(&self) -> (u64, u64) {
        self.counters.clone().unwrap()
    }
}

impl<V: Distance + Clone + Weight + PartialEq + Sync> SequentialAlgorithm<V> for GreedyHeuristic {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [V],
        matroid: Rc<dyn Matroid<V>>,
        p: usize,
    ) -> anyhow::Result<Vec<V>> {
        let _z = dataset.len() - p;
        let _k = matroid.rank();

        let start = Instant::now();
        let solution: Vec<V> = greedy_heuristic(dataset, Rc::clone(&matroid))
            .into_iter()
            .cloned()
            .collect();
        let elapsed_solution = start.elapsed();
        assert!(matroid.is_maximal(&solution, &dataset));

        self.profile
            .replace((Duration::from_secs(0), elapsed_solution));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));

        Ok(solution)
    }
}

fn greedy_heuristic<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: Rc<dyn Matroid<V>>,
) -> Vec<&'a V> {
    let mut min_dist = vec![std::f32::INFINITY; points.len()];
    let mut centers = Vec::new();
    let mut ranked: BTreeMap<OrderedF32, usize> = BTreeMap::new();

    let mut i = 0;
    centers.push(&points[i]);
    while !matroid.is_independent_ref(&centers) {
        centers.pop();
        i += 1;
        centers.push(&points[i]);
    }
    assert!(matroid.is_independent_ref(&centers));
    let mut maximal = false;
    let mut i = 0;
    while !maximal && i < points.len() {
        ranked.clear();
        let c = centers[i];

        // Update distances
        for (j, p) in points.iter().enumerate() {
            let d = c.distance(p);
            assert!(d.is_finite());
            if d < min_dist[j] {
                min_dist[j] = d;
            }
            ranked.insert(min_dist[j].into(), j);
        }

        // Reset this flag in the next loop
        maximal = true;
        // Iterate throught the nodes from the farthest
        for (_dist, idx) in ranked.iter().rev() {
            centers.push(&points[*idx]);
            if matroid.is_independent_ref(&centers) {
                maximal = false;
                break;
            }
            // If the point could not be added to the set of centers,
            // remove it
            centers.pop();
        }
        i += 1;
    }

    assert!(matroid.is_independent_ref(&centers));

    centers
}
