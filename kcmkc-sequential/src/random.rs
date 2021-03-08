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
        format!(
            r#"{{
                "seed": {}
            }}"#,
            self.seed
        )
    }

    fn run<'a>(
        &mut self,
        dataset: &'a [T],
        matroid: Box<dyn Matroid<T>>,
        p: usize,
    ) -> anyhow::Result<(
        Vec<&'a T>,
        usize,
        Box<dyn Iterator<Item = (&'a T, Option<(usize, f32)>)> + 'a>,
    )> {
        Ok(random_matroid_center(dataset, matroid, p, self.seed))
    }
}

fn random_matroid_center<'a, V: Distance + Clone>(
    points: &'a [V],
    matroid: Box<dyn Matroid<V>>,
    p: usize,
    seed: u64,
) -> (
    Vec<&'a V>,
    usize,
    Box<dyn Iterator<Item = (&'a V, Option<(usize, f32)>)> + 'a>,
) {
    use std::iter::FromIterator;

    let mut rng = XorShiftRng::seed_from_u64(seed);
    let mut points = Vec::from_iter(points.iter());
    points.shuffle(&mut rng);

    let centers: Vec<&'a V> = matroid.maximal_independent_set(&points[..]);

    let mut w_distances: Vec<(&V, Option<(usize, f32)>)> = points
        .iter()
        .map(|p| {
            let (c, d) = centers
                .iter()
                .enumerate()
                .map(|(i, c)| (i, p.distance(c)))
                .min_by(|p1, p2| p1.1.partial_cmp(&p2.1).unwrap())
                .unwrap();
            (*p, Some((c, d)))
        })
        .collect();

    w_distances.sort_by(|p1, p2| p1.1.unwrap().1.partial_cmp(&p2.1.unwrap().1).unwrap());

    let threshold = w_distances[p].1.unwrap().1;

    w_distances.iter_mut().for_each(|(_, assignment)| {
        if let Some((_, d)) = assignment {
            if *d > threshold {
                // unassign the point, i.e. make it an outlier
                assignment.take();
            }
        }
    });

    let outliers = w_distances.iter().filter(|pair| pair.1.is_none()).count();
    (centers, outliers, Box::new(w_distances.into_iter()))
}
