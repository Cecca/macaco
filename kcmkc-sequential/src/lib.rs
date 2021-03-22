use kcmkc_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};

pub mod chen_et_al;
pub mod kcenter;
pub mod random;
pub mod seq_coreset;
pub mod streaming_coreset;

pub trait SequentialAlgorithm<T: Distance + Clone>: Algorithm<T> {
    fn sequential_run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Box<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>>;
}
