use crate::{
    chen_et_al::{robust_matroid_center, VecWeightMap},
    kcenter::kcenter,
};
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{Matroid, Weight},
    types::Distance,
};

pub struct SeqCoreset {
    epsilon: f32,
}

impl SeqCoreset {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}

impl<V: Distance + Clone + Weight> Algorithm<V> for SeqCoreset {
    fn version(&self) -> u32 {
        1
    }

    fn name(&self) -> String {
        String::from("SeqCoreset")
    }

    fn parameters(&self) -> String {
        format!("{{\"epsilon\": {}}}", self.epsilon)
    }

    fn run<'a>(
        &mut self,
        dataset: &'a [V],
        matroid: Box<dyn Matroid<V>>,
        p: usize,
    ) -> anyhow::Result<Vec<V>> {
        // The approximation factor of the algorithm that extracts the solution from the coreset
        let beta = 3.0;

        let z = dataset.len() - p;
        let k = matroid.rank();

        // First find a clustering of z + k centers minimizing the radius, with no
        // matroid constraints, and get its radius
        let (_centers, assignments) = kcenter(dataset, k + z);
        let radius = assignments
            .map(|triplet| triplet.2)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Now cover the dataset with small-radius disks
        let disks = disk_cover(dataset, radius * self.epsilon / (2.0 * beta));

        // Then, get a maximal independent set from each disk,
        // and make each of its points a proxy
        let coreset: Vec<(V, u32)> = disks
            .iter()
            .flat_map(|disk| {
                assert!(disk.len() > 0);
                let is = matroid.maximal_independent_set(&disk);
                // There might be no independent set in this disk
                // In this case, no point in the disk can be part of the solution, but they do
                // still count towards the radius.
                //
                // In such case, we add an arbitrary point as a proxy for the
                // entire disk, just to be able to take into account all the
                // proxied points in the final solution computation.
                let proxies = if is.len() > 0 { is } else { vec![disk[0]] };

                let mut weights = vec![0u32; proxies.len()];

                // Fill-in weights by counting the assignments to proxies
                disk.iter()
                    .map(|p| {
                        proxies
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (i, p.distance(c)))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap()
                            .0
                    })
                    .for_each(|i| weights[i] += 1);

                proxies.into_iter().cloned().zip(weights.into_iter())
            })
            .collect();

        let weights = VecWeightMap::new(coreset.iter().map(|p| p.1).collect());
        let coreset: Vec<V> = coreset.into_iter().map(|p| p.0).collect();
        println!("Coreset of size {}", coreset.len());

        let solution = robust_matroid_center(&coreset, &matroid, p, &weights);
        assert!(matroid.is_maximal(&solution, &dataset));

        Ok(solution)
    }
}

fn disk_cover<'a, V: Distance>(points: &'a [V], radius: f32) -> Vec<Vec<&'a V>> {
    let mut disks = Vec::new();
    let mut covered = vec![false; points.len()];

    let mut i = 0;
    while i < points.len() {
        if !covered[i] {
            let mut disk = Vec::new();
            disk.push(&points[i]);
            let current_center = &points[i];
            covered[i] = true;
            let mut j = i + 1;
            while j < points.len() {
                if !covered[j] {
                    let d = points[j].distance(current_center);
                    if d <= radius {
                        covered[j] = true;
                        disk.push(&points[j]);
                    }
                }
                j += 1;
            }
            disks.push(disk);
        }
        i += 1;
    }

    disks
}
