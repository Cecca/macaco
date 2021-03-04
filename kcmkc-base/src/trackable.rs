use std::collections::HashMap;

/// A trait for things that can be tracked in the result database
pub trait Trackable {
    fn version(&self) -> u32;
    fn name(&self) -> String;
    fn parameters(&self) -> HashMap<String, String>;
    fn parameters_string(&self) -> String {
        let v: Vec<String> = self
            .parameters()
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        v.join(" ")
    }
}
