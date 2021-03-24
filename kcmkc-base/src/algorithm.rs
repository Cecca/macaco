use crate::types::Distance;

pub trait Algorithm<T: Distance + Clone> {
    fn version(&self) -> u32;
    fn name(&self) -> String;
    fn parameters(&self) -> String;
    /// Returns the size and the proxy radius of the coreset,
    /// if appropriate.
    fn coreset(&self) -> Option<Vec<T>>;
}
