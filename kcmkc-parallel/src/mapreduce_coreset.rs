use abomonation::Abomonation;
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    perf_counters,
    types::Distance,
};
use kcmkc_sequential::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    kcenter::kcenter,
};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use std::{cell::RefCell, time::Instant};
use std::{ops::Deref, sync::atomic::AtomicU64};
use timely::dataflow::operators::{aggregation::Aggregate, Exchange};
use timely::dataflow::{InputHandle, ProbeHandle};
use timely::ExchangeData;
use timely::{communication::Allocate, worker::Worker};
use timely::{
    communication::Allocator,
    dataflow::{channels::pact::Exchange as ExchangePact, operators::*},
};

use crate::ParallelAlgorithm;

pub struct MapReduceCoreset<V> {
    tau: usize,
    coreset: Option<Vec<V>>,
    profile: Option<(Duration, Duration)>,
    counters: Option<(u64, u64)>,
}

impl<V> MapReduceCoreset<V> {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            coreset: None,
            profile: None,
            counters: None,
        }
    }
}

impl<V: Distance + Clone + Weight + PartialEq> Algorithm<V> for MapReduceCoreset<V> {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("MRCoreset")
    }

    fn parameters(&self) -> String {
        format!("{{ \"tau\": {} }}", self.tau)
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

impl<T: Distance + Clone + Weight + PartialEq + Abomonation + ExchangeData> ParallelAlgorithm<T>
    for MapReduceCoreset<T>
{
    fn parallel_run(
        &mut self,
        worker: &mut Worker<Allocator>,
        dataset: &[T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        let start = Instant::now();
        let coreset = mapreduce_coreset(worker, dataset, Rc::clone(&matroid), self.tau);

        let weights = VecWeightMap::new(coreset.iter().map(|p| p.1).collect());
        let coreset: Vec<T> = coreset.into_iter().map(|p| p.0).collect();
        let elapsed_coreset = start.elapsed();
        println!("Coreset of size {} ({:?})", coreset.len(), elapsed_coreset);

        let start = Instant::now();
        let solution = if worker.index() == 0 {
            let s = robust_matroid_center(&coreset, Rc::clone(&matroid), p, &weights);
            assert!(matroid.is_maximal(&s, &dataset));
            s
        } else {
            Vec::new()
        };
        let elapsed_solution = start.elapsed();

        self.coreset.replace(coreset);
        self.profile.replace((elapsed_coreset, elapsed_solution));
        self.counters.replace(collect_counters(worker));

        Ok(solution)
    }
}

// Set up and run a small dataflow to collect performance counters from all workers
fn collect_counters(worker: &mut Worker<Allocator>) -> (u64, u64) {
    println!("Collecting counters");
    let distance_counter = Arc::new(AtomicU64::new(0));
    let distance_counter2 = Arc::clone(&distance_counter);
    let oracle_counter = Arc::new(AtomicU64::new(0));
    let oracle_counter2 = Arc::clone(&oracle_counter);

    // Collect the counters from all the workers into the first worker
    let (mut input, probe) = worker.dataflow::<(), _, _>(move |scope| {
        let (input, stream) = scope.new_input::<(u8, u64)>();
        let probe = stream
            .aggregate(
                |_key, val, agg| {
                    *agg += val;
                },
                |key, agg: u64| (key, agg),
                |key| *key as u64,
            )
            .exchange(|_| 0)
            .map(move |(typ, cnt)| {
                if typ == 0 {
                    distance_counter.fetch_add(cnt, std::sync::atomic::Ordering::SeqCst);
                } else {
                    oracle_counter.fetch_add(cnt, std::sync::atomic::Ordering::SeqCst);
                }
                ()
            })
            .probe();
        (input, probe)
    });
    input.send((0, perf_counters::distance_count()));
    input.send((1, perf_counters::matroid_oracle_count()));
    input.close();
    worker.step_while(|| !probe.done());
    println!("Counters collected");

    (
        distance_counter2.load(std::sync::atomic::Ordering::SeqCst),
        oracle_counter2.load(std::sync::atomic::Ordering::SeqCst),
    )
}

fn mapreduce_coreset<'a, T: ExchangeData + Distance, A: Allocate>(
    worker: &mut Worker<A>,
    dataset: &[T],
    matroid: Rc<dyn Matroid<T> + 'static>,
    tau: usize,
) -> Vec<(T, u32)> {
    let mut input: InputHandle<(), (u64, T)> = InputHandle::new();
    let mut probe = ProbeHandle::new();
    let result1: Rc<RefCell<Vec<(T, u32)>>> = Rc::new(RefCell::new(Vec::new()));
    let result2 = Rc::clone(&result1);

    worker.dataflow(|scope| {
        let mut stash = Vec::new();

        scope
            // Hook the input in the dataflow
            .input_from(&mut input)
            // Exchange the vectors randomly, and then build the coreset in each partition
            .unary_notify(
                ExchangePact::new(|x: &(u64, T)| x.0),
                "coreset_builder",
                None,
                move |input, output, notificator| {
                    // stash the input as it streams in
                    input.for_each(|t, data| {
                        stash.extend(data.replace(Vec::new()).into_iter().map(|pair| pair.1));
                        notificator.notify_at(t.retain());
                    });

                    // When ready, compute the output and send it
                    notificator.for_each(|t, _, _| {
                        let (centers, assignments) = kcenter(&stash, tau);
                        let mut disks = vec![Vec::new(); centers.len()];
                        for (v, i, _) in assignments {
                            disks[i].push(v);
                        }

                        let coreset = disks.iter().flat_map(|disk| {
                            assert!(disk.len() > 0);
                            let is = matroid.maximal_independent_set(&disk);

                            let proxies = if is.len() > 0 { is } else { vec![disk[0]] };
                            let mut weights = vec![0u32; proxies.len()];

                            // Fill-in weights by counting the assignments to proxies
                            disk.iter()
                                .map(|p| {
                                    proxies
                                        .iter()
                                        .enumerate()
                                        .map(|(i, c)| (i, p.distance(c)))
                                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                        .unwrap()
                                        .0
                                })
                                .for_each(|i| weights[i] += 1);

                            proxies.into_iter().cloned().zip(weights.into_iter())
                        });

                        output.session(&t).give_iterator(coreset);
                    });
                },
            )
            // Collect all the data into the first worker and stash it into the output
            .unary(ExchangePact::new(|_| 0), "output_collector", |_, _| {
                move |input, output| {
                    input.for_each(|t, data| {
                        result1
                            .deref()
                            .borrow_mut()
                            .extend(data.replace(Vec::new()).into_iter());
                        output.session(&t).give(());
                    });
                }
            })
            .probe_with(&mut probe);
    });

    // Populate the input, only with worker 0 that will
    // distribute all the input to the other workers
    if worker.index() == 0 {
        let mut cnt = 0;
        for x in dataset {
            input.send((cnt, x.clone()));
            if cnt % 10000 == 0 {
                // Do some work in order not to exhaust the buffers
                worker.step();
            }
            cnt += 1;
        }
    }
    input.close();
    // do all the work in the dataflow
    worker.step_while(|| !probe.done());

    // at this point, the result is ready to be collected
    result2.take()
}
