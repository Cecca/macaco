use macaco_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};
use std::rc::Rc;

pub mod chen_et_al;
pub mod disks;
pub mod greedy_heuristic;
pub mod kcenter;
pub mod local_search;
pub mod random;
pub mod seq_coreset;
pub mod streaming_coreset;

pub trait SequentialAlgorithm<T: Distance + Clone>: Algorithm<T> {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Rc<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>>;
}
