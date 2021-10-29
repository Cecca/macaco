use crate::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    kcenter::kcenter,
    SequentialAlgorithm,
};
use macaco_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    perf_counters,
    types::Distance,
};
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

pub struct SeqCoreset<V> {
    tau: usize,
    coreset: Option<Vec<V>>,
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
    memory: Option<usize>,
}

impl<V> SeqCoreset<V> {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            coreset: None,
            profile: None,
            counters: None,
            memory: None,
        }
    }
}

impl<V: Distance + Clone + Weight + PartialEq> Algorithm<V> for SeqCoreset<V> {
    fn version(&self) -> u32 {
        4
    }

    fn name(&self) -> String {
        String::from("SeqCoreset")
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

    fn memory_usage(&self) -> Option<usize> {
        self.memory
    }
}

impl<V: Distance + Clone + Weight + PartialEq + Sync> SequentialAlgorithm<V> for SeqCoreset<V> {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [V],
        matroid: Rc<dyn Matroid<V>>,
        p: usize,
    ) -> anyhow::Result<Vec<V>> {
        let _z = dataset.len() - p;
        let _k = matroid.rank();

        let start_memory = macaco_base::allocator::allocated();

        // Cloning the dataset improves data locality, in that it rearranges
        // the vectors that each element points to.
        let mut loc_dataset = Vec::new();
        loc_dataset.extend(dataset.iter().cloned());
        let dataset = loc_dataset;

        let start = Instant::now();
        // First find a clustering of tau centers minimizing the radius, with no
        // matroid constraints
        let (centers, assignments) = kcenter(&dataset, self.tau);
        println!("[{:?}] k-center completed", start.elapsed());

        // Build disks by assigning each center to the closest point
        let mut disks = vec![Vec::new(); centers.len()];
        for (v, i, _) in assignments {
            disks[i].push(v);
        }
        println!("[{:?}] points assigned to closest center", start.elapsed());

        // Then, get a maximal independent set from each disk,
        // and make each of its points a proxy
        let coreset: Vec<(V, u32)> = disks
            .iter()
            .flat_map(|disk| {
                assert!(disk.len() > 0);
                let is = matroid.maximal_independent_set(&disk);
                // There might be no independent set in this disk
                // In this case, no point in the disk can be part of the solution, but they do
                // still count towards the radius.
                //
                // In such case, we add an arbitrary point as a proxy for the
                // entire disk, just to be able to take into account all the
                // proxied points in the final solution computation.
                let proxies = if is.len() > 0 { is } else { vec![disk[0]] };

                let n_proxies = proxies.len();
                let mut weights = vec![0u32; n_proxies];

                // Fill-in weights by counting the assignments to proxies.
                // In the paper, we write that each point is assigned to the closest proxy in the
                // disk, but then we use the disk's radius in the proof.
                //
                // In practice, this is a huge bottleneck, hence we just assign to arbitrary elements
                // of the independent set so that the weights are balanced.
                for (i, _p) in disk.into_iter().enumerate() {
                    weights[i % n_proxies] += 1;
                }
                // for p in disk.into_iter() {
                //     let closest = proxies
                //         .iter()
                //         .enumerate()
                //         .map(|(j, c)| (OrderedF32(p.distance(c)), j))
                //         .min()
                //         .unwrap()
                //         .1;
                //     weights[closest] += 1;
                // }

                proxies.into_iter().cloned().zip(weights.into_iter())
            })
            .collect();
        // println!("[{:?}] coreset constructed", start.elapsed());

        let weights = VecWeightMap::new(coreset.iter().map(|p| p.1).collect());
        let coreset: Vec<V> = coreset.into_iter().map(|p| p.0).collect();
        let elapsed_coreset = start.elapsed();
        println!("Coreset of size {}", coreset.len(),);

        let end_memory = macaco_base::allocator::allocated();
        let coreset_memory = end_memory - start_memory;
        println!("used {} bytes to build the coreset", coreset_memory);
        self.memory.replace(coreset_memory);

        let start = Instant::now();
        let solution = robust_matroid_center(&coreset, Rc::clone(&matroid), p, &weights);
        // let solution = local_search(&coreset, Rc::clone(&matroid), p, &weights)
        //     .expect("could not find solution with local search");
        let elapsed_solution = start.elapsed();
        println!("Solution found on coreset in {:.2?}", elapsed_solution);
        debug_assert!(matroid.is_maximal(&solution, &dataset));

        self.coreset.replace(coreset);
        self.profile.replace((elapsed_coreset, elapsed_solution));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));

        println!("Returning solution to main");
        Ok(solution)
    }
}
