use std::time::Duration;

use crate::types::Distance;

pub trait Algorithm<T: Distance + Clone> {
    fn version(&self) -> u32;
    fn name(&self) -> String;
    fn parameters(&self) -> String;
    /// Returns the size and the proxy radius of the coreset,
    /// if appropriate.
    fn coreset(&self) -> Option<Vec<T>>;
    /// How long it took to build the coreset (first element)
    /// and to run the approximation algorithm on the coreset
    /// (second element)
    fn time_profile(&self) -> (Duration, Duration);

    fn counters(&self) -> (u64, u64);
}
