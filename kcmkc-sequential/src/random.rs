use kcmkc_base::{algorithm::Algorithm, matroid::Matroid, types::Distance};

use rand::prelude::*;
use rand_xorshift::XorShiftRng;
pub struct RandomClustering {
    pub seed: u64,
}

impl<T: Distance + Clone> Algorithm<T> for RandomClustering {
    fn version(&self) -> u32 {
        1u32
    }

    fn name(&self) -> String {
        String::from("Random")
    }

    fn parameters(&self) -> String {
        format!(r#"{{ "seed": {} }}"#, self.seed)
    }

    fn run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Box<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<Vec<T>> {
        let sol = random_matroid_center(dataset, matroid, p, self.seed);
        let sol = sol.into_iter().cloned().collect();
        Ok(sol)
    }
}

fn random_matroid_center<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: Box<dyn Matroid<V>>,
    _p: usize,
    seed: u64,
) -> Vec<&'a V> {
    use std::iter::FromIterator;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut points = Vec::from_iter(points.iter());
    points.shuffle(&mut rng);

    let centers: Vec<&'a V> = matroid.maximal_independent_set(&points[..]);
    centers
}
