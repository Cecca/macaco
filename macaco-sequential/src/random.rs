use macaco_base::{algorithm::Algorithm, matroid::Matroid, perf_counters, types::Distance};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use crate::SequentialAlgorithm;
pub struct RandomClustering {
    pub seed: u64,
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
}

impl RandomClustering {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            profile: None,
            counters: None,
        }
    }
}

impl<T: Distance + Clone> Algorithm<T> for RandomClustering {
    fn version(&self) -> u32 {
        1u32
    }

    fn name(&self) -> String {
        String::from("Random")
    }

    fn parameters(&self) -> String {
        format!(r#"{{ "seed": {} }}"#, self.seed)
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

impl<T: Distance + Clone> SequentialAlgorithm<T> for RandomClustering {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        let start = Instant::now();
        let sol = random_matroid_center(dataset, matroid, p, self.seed);
        let sol = sol.into_iter().cloned().collect();
        let elapsed = start.elapsed();
        self.profile.replace((Duration::from_secs(0), elapsed));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));
        Ok(sol)
    }
}

fn random_matroid_center<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: Rc<dyn Matroid<V>>,
    _p: usize,
    seed: u64,
) -> Vec<&'a V> {
    use std::iter::FromIterator;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut points = Vec::from_iter(points.iter());
    points.shuffle(&mut rng);

    let centers: Vec<&'a V> = matroid.maximal_independent_set(&points[..]);
    centers
}
