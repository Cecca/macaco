use crate::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    SequentialAlgorithm,
};
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    perf_counters,
    types::{Distance, OrderedF32},
};
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

pub struct StreamingCoreset<T> {
    tau: usize,
    coreset: Option<Vec<T>>,
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
}

impl<V> StreamingCoreset<V> {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            coreset: None,
            profile: None,
            counters: None,
        }
    }
}

impl<V: Distance + Clone + Weight + PartialEq> Algorithm<V> for StreamingCoreset<V> {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("StreamingCoreset")
    }

    fn parameters(&self) -> String {
        format!("{{\"tau\": {}}}", self.tau)
    }

    fn coreset(&self) -> Option<Vec<V>> {
        self.coreset.clone()
    }

    fn time_profile(&self) -> (Duration, Duration) {
        self.profile.clone().unwrap()
    }

    fn counters(&self) -> (u64, u64) {
        self.counters.clone().unwrap()
    }
}

impl<V: Distance + Clone + Weight + PartialEq + Sync> SequentialAlgorithm<V>
    for StreamingCoreset<V>
{
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [V],
        matroid: Rc<dyn Matroid<V>>,
        p: usize,
    ) -> anyhow::Result<Vec<V>> {
        let start = Instant::now();
        let mut state = StreamingState::new(self.tau, Rc::clone(&matroid));
        for x in dataset {
            state.update(x);
        }

        let coreset = state.coreset();
        let weights = VecWeightMap::new(coreset.iter().map(|p| p.1).collect());
        let coreset: Vec<V> = coreset.into_iter().map(|p| p.0).collect();
        let elapsed_coreset = start.elapsed();
        println!("Coreset of size {}", coreset.len());

        let start = Instant::now();
        let solution = robust_matroid_center(&coreset, Rc::clone(&matroid), p, &weights);
        let elapsed_solution = start.elapsed();
        assert!(matroid.is_maximal(&solution, &dataset));

        self.coreset.replace(coreset);
        self.profile.replace((elapsed_coreset, elapsed_solution));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));

        Ok(solution)
    }
}

struct StreamingState<T: Distance> {
    k: usize,
    /// upper bound to the distance of any node to the
    /// closest cluster center. It is `None` until the
    /// state has been initialized, which happens after the
    /// first `k` elements of the stream have been inserted
    distance_bound: Option<f32>,
    /// The first element of each pair is the center, then we have an
    /// independent set and the corresponding weights
    clusters: Vec<(T, Vec<T>, Vec<u32>)>,
    matroid: Rc<dyn Matroid<T>>,
}

impl<T: Clone + Distance> StreamingState<T> {
    fn new(k: usize, matroid: Rc<dyn Matroid<T>>) -> Self {
        Self {
            k,
            distance_bound: None,
            clusters: Vec::new(),
            matroid,
        }
    }

    fn coreset(self) -> Vec<(T, u32)> {
        let weights: Vec<u32> = self
            .clusters
            .iter()
            .flat_map(|triplet| triplet.2.iter())
            .copied()
            .collect();
        let points: Vec<T> = self
            .clusters
            .into_iter()
            .flat_map(|triplet| triplet.1)
            .collect();
        assert!(weights.len() == points.len());
        points.into_iter().zip(weights).collect()
    }

    fn update(&mut self, x: &T) {
        if let Some(distance_bound) = self.distance_bound {
            let (d, (_center, cluster, weights)) = self
                .clusters
                .iter_mut()
                .map(|triplet| (OrderedF32(triplet.0.distance(x)), triplet))
                .min_by_key(|pair| pair.0)
                .unwrap();
            if d.0 > distance_bound {
                self.clusters.push((x.clone(), vec![x.clone()], vec![1]));
            } else {
                // The node is covered by the existing centers, we should just check
                // if we should add it to the independent sets or if we just need to increase
                // the counter of an existing element
                cluster.push(x.clone());
                if self.matroid.is_independent(&cluster) {
                    weights.push(1);
                } else {
                    cluster.pop();
                    // add to the weight of the one with minimum weight
                    *weights.iter_mut().min().unwrap() += 1;
                }
            }
            if self.clusters.len() == self.k + 1 {
                self.merge();
            }
        } else {
            self.clusters.push((x.clone(), vec![x.clone()], vec![1]));
            if self.clusters.len() == self.k + 1 {
                // define the distance bound, and merge the clusters
                self.distance_bound.replace(
                    self.clusters
                        .iter()
                        .flat_map(|(x, _, _)| {
                            self.clusters
                                .iter()
                                .map(move |(y, _, _)| OrderedF32(x.distance(y)))
                                .filter(|d| d > &OrderedF32(0.0))
                        })
                        .min()
                        .unwrap()
                        .into(),
                );
                self.merge();
            }
        }
        assert!(self.clusters.len() <= self.k);
    }

    /// Reduce the number of clusters to `k` by doubling the radius
    /// (repeatedly) merging clusters that are within the new radius
    /// from each other
    fn merge(&mut self) {
        while self.clusters.len() > self.k {
            self.distance_bound
                .replace(self.distance_bound.unwrap() * 2.0);
            let mut i = 0;
            while i < self.clusters.len() {
                let mut j = i + 1;
                let mut n = self.clusters.len();
                // We don't use `self.clusters.len()` in the condition of
                // the loop because of borrowing rules
                while j < n && i < n {
                    if self.clusters[i].0.distance(&self.clusters[j].0)
                        <= self.distance_bound.unwrap()
                    {
                        // remove the center to be merged
                        let (_center_to_merge, cluster_to_merge, weights_to_merge) =
                            self.clusters.swap_remove(j);

                        // Merge the independent sets and the weights
                        let (_, current_cluster, current_weights) = &mut self.clusters[i];
                        for (x, w) in cluster_to_merge
                            .into_iter()
                            .zip(weights_to_merge.into_iter())
                        {
                            current_cluster.push(x);
                            if self.matroid.is_independent(&current_cluster) {
                                current_weights.push(w);
                            } else {
                                // Remove the point we just added, since it does not
                                // form an indepedent set
                                current_cluster.pop();
                                // Give the weight of the point that is already represented
                                // to the center with the smallest weight
                                *current_weights.iter_mut().min().unwrap() += w;
                            }
                        }

                        n -= 1;
                    }
                    j += 1;
                }
                i += 1;
            }
        }
    }
}
