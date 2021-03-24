use std::sync::atomic::AtomicU64;

pub static GLOBAL_DISTANCE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static GLOBAL_MATROID_ORACLE_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn inc_distance_count() {
    GLOBAL_DISTANCE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

pub fn inc_matroid_oracle_count() {
    GLOBAL_MATROID_ORACLE_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// gets the value of the counter, and sets the value to 0.
/// if multiple threads read this concurrently, they will all get 0 except the first one,
/// assuming the counter is not incremented in-between
pub fn distance_count() -> u64 {
    GLOBAL_DISTANCE_COUNT.fetch_and(0, std::sync::atomic::Ordering::SeqCst)
}

pub fn matroid_oracle_count() -> u64 {
    GLOBAL_MATROID_ORACLE_COUNT.fetch_and(0, std::sync::atomic::Ordering::SeqCst)
}
