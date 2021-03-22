use abomonation::Abomonation;
use kcmkc_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};
use std::hash::Hash;
use std::rc::Rc;
use timely::{communication::Allocator, worker::Worker};

pub mod mapreduce_coreset;

pub trait ParallelAlgorithm<T: Distance + Clone + Abomonation>: Algorithm<T> {
    fn parallel_run(
        &mut self,
        worker: &mut Worker<Allocator>,
        dataset: &[T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>>;
}
