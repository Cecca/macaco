use kcmkc_base::{matroid::Matroid, types::Distance};
use kcmkc_sequential::kcenter::kcenter;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::rc::Rc;
use std::{
    hash::{Hash, Hasher},
    ops::Deref,
};
use timely::dataflow::{channels::pact::Exchange, operators::*};
use timely::dataflow::{InputHandle, ProbeHandle};
use timely::ExchangeData;
use timely::{communication::Allocate, worker::Worker};

fn mapreduce_coreset<T: ExchangeData + Hash + Distance, A: Allocate>(
    worker: &mut Worker<A>,
    dataset: &[T],
    matroid: &'static Box<dyn Matroid<T> + 'static>,
    tau: usize,
) -> Vec<(T, u32)> {
    let mut input: InputHandle<(), T> = InputHandle::new();
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
                Exchange::new(|x: &T| {
                    let mut h = DefaultHasher::new();
                    x.hash(&mut h);
                    h.finish()
                }),
                "coreset_builder",
                None,
                move |input, output, notificator| {
                    // stash the input as it streams in
                    input.for_each(|t, data| {
                        stash.extend(data.replace(Vec::new()));
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

    // Populate the input
    let mut cnt = 0;
    for x in dataset {
        input.send(x.clone());
        if cnt % 10000 == 0 {
            // Do some work in order not to exhaust the buffers
            worker.step();
        }
        cnt += 1;
    }
    input.close();
    // do all the work in the dataflow
    worker.step_while(|| !probe.done());

    // at this point, the result is ready to be collected
    result2.take()
}
