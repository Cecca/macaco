use abomonation::Abomonation;
use kcmkc_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};
use std::hash::Hash;
use std::rc::Rc;
use timely::{communication::Allocate, worker::Worker};

pub mod mapreduce_coreset;

pub trait ParallelAlgorithm<T: Distance + Clone + Hash + Abomonation>: Algorithm<T> {
    fn parallel_run<A: Allocate>(
        &mut self,
        worker: &mut Worker<A>,
        dataset: &[T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>>;
}
