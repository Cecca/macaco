use log::info;
/// Implementation of the streaming algorithm by Sagar Kale
///   "Small Space Stream Summary for Matroid Center"
/// http://arxiv.org/abs/1810.06267v2
use macaco_base::{
    algorithm::Algorithm,
    matroid::Matroid,
    perf_counters,
    types::{Distance, OrderedF32},
};
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use crate::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    SequentialAlgorithm,
};

pub struct KaleStreaming<T: Distance + Clone> {
    initial_guess: f32,
    epsilon: f32,
    coreset: Option<Vec<T>>,
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
    memory: Option<usize>,
}

impl<T: Distance + Clone> KaleStreaming<T> {
    pub fn new(initial_guess: f32, epsilon: f32) -> Self {
        Self {
            initial_guess,
            epsilon,
            coreset: None,
            profile: None,
            counters: None,
            memory: None,
        }
    }
}

impl<T: Distance + Clone> Algorithm<T> for KaleStreaming<T> {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        "KaleStreaming".to_owned()
    }

    fn parameters(&self) -> String {
        format!(
            "{{\"initial_guess\": {}, \"epsilon\": {}}}",
            self.initial_guess, self.epsilon
        )
    }

    fn coreset(&self) -> Option<Vec<T>> {
        self.coreset.clone()
    }

    fn time_profile(&self) -> (Duration, Duration) {
        self.profile.clone().unwrap()
    }

    fn memory_usage(&self) -> Option<usize> {
        self.memory
    }

    fn counters(&self) -> (u64, u64) {
        self.counters.clone().unwrap()
    }
}

impl<T: Distance + Clone + PartialEq + Sync> SequentialAlgorithm<T> for KaleStreaming<T> {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        let mut local_dataset: Vec<T> = Vec::new();
        local_dataset.extend(dataset.iter().cloned());
        let dataset = local_dataset;
        let rank = matroid
            .maximal_independent_set(&dataset.iter().collect::<Vec<&T>>())
            .len();
        let z = dataset.len() - p;

        let start_memory = macaco_base::allocator::allocated();
        let start = Instant::now();

        let mut state = KaleState::new(
            z,
            rank,
            self.initial_guess,
            self.epsilon,
            Rc::clone(&matroid),
        );

        let mut cnt = 0;
        for x in &dataset {
            state.update(x);
            cnt += 1;
        }
        info!("Processed {} points", cnt);
        let end_memory = macaco_base::allocator::allocated();
        let coreset_memory = end_memory - start_memory;
        println!("used {} bytes to build the coreset", coreset_memory);
        self.memory.replace(coreset_memory);

        let coreset = state.coreset();
        let elapsed_coreset = start.elapsed();
        println!(
            "Coreset of size {} computed in {:?}",
            coreset.len(),
            elapsed_coreset
        );

        println!("Setting up weights");
        let mut weights = vec![0; coreset.len()];
        for x in &dataset {
            let i = coreset
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| OrderedF32(c.distance(x)))
                .unwrap()
                .0;
            weights[i] += 1;
        }
        let weights = VecWeightMap::new(weights);

        let start = Instant::now();
        let solution = robust_matroid_center(&coreset, Rc::clone(&matroid), p, &weights);
        let elapsed_solution = start.elapsed();
        debug_assert!(matroid.is_maximal(&solution, &dataset));

        self.coreset.replace(coreset);
        self.profile.replace((elapsed_coreset, elapsed_solution));
        self.counters.replace((
            perf_counters::distance_count(),
            perf_counters::matroid_oracle_count(),
        ));

        Ok(solution)
    }
}

pub struct KaleState<T: Distance + Clone> {
    child_factor: f32,
    instances: Vec<StreamingInstance<T>>,
}

impl<T: Distance + Clone> KaleState<T> {
    pub fn new(
        z: usize,
        rank: usize,
        initial_guess: f32,
        epsilon: f32,
        matroid: Rc<dyn Matroid<T>>,
    ) -> Self {
        let mut instances = Vec::new();
        let mut radius = initial_guess;
        instances.push(StreamingInstance::new(z, rank, radius, Rc::clone(&matroid)));
        let threshold = initial_guess * (2.0 + epsilon) / epsilon;
        while radius < threshold {
            radius = radius * (1.0 + epsilon);
            println!("Pushing with radius {}", radius);
            instances.push(StreamingInstance::new(z, rank, radius, Rc::clone(&matroid)));
        }
        let child_factor = instances.last().unwrap().radius / initial_guess;

        Self {
            child_factor,
            instances,
        }
    }

    pub fn update(&mut self, x: &T) {
        let mut new_instances = Vec::new();
        for instance in self.instances.iter_mut() {
            if !instance.update(x) {
                let mut new_inst = StreamingInstance {
                    active: true,
                    z: instance.z,
                    rank: instance.rank,
                    radius: instance.radius * self.child_factor,
                    l: 0,
                    pivots: Vec::new(),
                    free_points: Vec::new(),
                    matroid: Rc::clone(&instance.matroid),
                };
                println!("Creating new instance with radius {}", new_inst.radius);

                for (_, ac, ic) in &instance.pivots {
                    for x in ac {
                        new_inst.update(x);
                    }
                    for x in ic {
                        new_inst.update(x);
                    }
                }
                for x in &instance.free_points {
                    new_inst.update(x);
                }

                new_instances.push(new_inst);
            }
        }

        // Try to add x to any instance that might have been created
        for instance in &mut new_instances {
            instance.update(x);
        }

        self.instances.extend(new_instances);
    }

    fn coreset(&self) -> Vec<T> {
        let winner = self
            .instances
            .iter()
            .filter(|i| i.active)
            .min_by_key(|i| OrderedF32(i.radius))
            .expect("No instance is active")
            .clone();

        let mut res = Vec::new();
        for (_, ac, ic) in winner.pivots {
            res.extend(ac);
            res.extend(ic);
        }
        res.extend(winner.free_points);
        res
    }
}

#[derive(Clone)]
struct StreamingInstance<T: Distance + Clone> {
    active: bool,
    z: usize,
    rank: usize,
    radius: f32,
    l: usize,
    pivots: Vec<(T, Vec<T>, Vec<T>)>,
    free_points: Vec<T>,
    matroid: Rc<dyn Matroid<T>>,
}

impl<T: Distance + Clone> StreamingInstance<T> {
    fn new(z: usize, rank: usize, radius: f32, matroid: Rc<dyn Matroid<T>>) -> Self {
        Self {
            active: true,
            z,
            rank,
            radius,
            l: 0,
            pivots: Vec::new(),
            free_points: Vec::new(),
            matroid,
        }
    }

    fn update(&mut self, x: &T) -> bool {
        if !self.active {
            return false;
        }
        // Find the closest center
        let (dist, (_c, _ac, ic)) = self
            .pivots
            .iter_mut()
            .map(|triplet| (OrderedF32(triplet.0.distance(&x)), triplet))
            .min_by_key(|pair| pair.0)
            .unwrap();

        // Try to add the point to the cluster
        if dist.0 <= 4.0 * self.radius {
            ic.push(x.clone());
            if !self.matroid.is_independent(&ic) {
                ic.pop().unwrap();
            }
        } else {
            self.free_points.push(x.clone());
        }

        if self.free_points.len() >= (self.rank - self.l + 1) as usize * self.z + 1 {
            self.l += 1;
            if self.l >= self.rank + 1 {
                self.active = false;
                return false;
            }

            let c = self
                .free_points
                .iter()
                .find(|c| {
                    self.free_points
                        .iter()
                        .filter(|x| c.distance(x) <= (2.0 * self.radius))
                        .count()
                        >= self.z + 1
                })
                .cloned();

            if c.is_none() {
                self.active = false;
                return false;
            }
            let c = c.unwrap();

            let mut ic = Vec::new();
            let mut ac = Vec::new();
            ac.push(c.clone());
            ic.push(c.clone());

            let mut new_free = Vec::new();
            for x in self.free_points.drain(..) {
                if c.distance(&x) <= 4.0 * self.radius {
                    if ac.len() <= self.z {
                        ac.push(x.clone());
                    }
                    ic.push(x);
                    if !self.matroid.is_independent(&ic) {
                        ic.pop();
                    }
                } else {
                    new_free.push(x);
                }
            }

            self.pivots.push((c.clone(), ac, ic));
        }

        true
    }
}
