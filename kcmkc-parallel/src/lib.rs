use std::hash::Hash;

use abomonation::Abomonation;
use kcmkc_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};

pub mod mapreduce_coreset;

pub trait ParallelAlgorithm<T: Distance + Clone + Hash + Abomonation>: Algorithm<T> {
    fn parallel_run(&mut self, dataset: &[T], matroid: Box<dyn Matroid<T>>, p: usize);
}
