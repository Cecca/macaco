use crate::{matroid::Matroid, types::Distance};

pub trait Algorithm<T: Distance + Clone> {
    fn version(&self) -> u32;
    fn name(&self) -> String;
    fn parameters(&self) -> String;
    fn run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Box<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>>;
}
