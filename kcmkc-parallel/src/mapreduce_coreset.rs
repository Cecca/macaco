use abomonation::Abomonation;
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    types::Distance,
};
use kcmkc_sequential::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    kcenter::kcenter,
};
use rand::{RngCore, SeedableRng};
use rand_xorshift::XorShiftRng;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::ops::Deref;
use std::rc::Rc;
use timely::dataflow::{InputHandle, ProbeHandle};
use timely::ExchangeData;
use timely::{communication::Allocate, worker::Worker};
use timely::{
    communication::Allocator,
    dataflow::{channels::pact::Exchange, operators::*},
};

use crate::ParallelAlgorithm;

pub struct MapReduceCoreset<V> {
    tau: usize,
    seed: u64,
    coreset: Option<Vec<V>>,
}

impl<V> MapReduceCoreset<V> {
    pub fn new(tau: usize, seed: u64) -> Self {
        Self {
            tau,
            seed,
            coreset: None,
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
        format!("{{ \"tau\": {}, \"seed\": {} }}", self.tau, self.seed)
    }

    fn coreset(&self) -> Option<Vec<V>> {
        self.coreset.clone()
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
        let coreset = mapreduce_coreset(worker, dataset, Rc::clone(&matroid), self.tau, self.seed);

        let weights = VecWeightMap::new(coreset.iter().map(|p| p.1).collect());
        let coreset: Vec<T> = coreset.into_iter().map(|p| p.0).collect();
        println!("Coreset of size {}", coreset.len());

        let solution = if worker.index() == 0 {
            let s = robust_matroid_center(&coreset, Rc::clone(&matroid), p, &weights);
            assert!(matroid.is_maximal(&s, &dataset));
            s
        } else {
            Vec::new()
        };

        self.coreset.replace(coreset);

        Ok(solution)
    }
}

fn mapreduce_coreset<'a, T: ExchangeData + Distance, A: Allocate>(
    worker: &mut Worker<A>,
    dataset: &[T],
    matroid: Rc<dyn Matroid<T> + 'static>,
    tau: usize,
    seed: u64,
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
                Exchange::new(|x: &(u64, T)| x.0),
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
            .unary(Exchange::new(|_| 0), "output_collector", |_, _| {
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
        let mut rng = XorShiftRng::seed_from_u64(seed);
        let mut cnt = 0;
        for x in dataset {
            input.send((rng.next_u64(), x.clone()));
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
