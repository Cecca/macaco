use crate::{
    chen_et_al::{self, robust_matroid_center},
    kcenter::kcenter,
};
use kcmkc_base::{
    algorithm::Algorithm,
    matroid::{Matroid, PartitionMatroidElement, SetWeight, TransveralMatroidElement, Weight},
    types::Distance,
};

struct SeqCoreset {
    epsilon: f32,
}

impl<V: Distance + Clone + Weight + SetWeight> Algorithm<V> for SeqCoreset {
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
    ) -> anyhow::Result<(
        Vec<&'a V>,
        usize,
        Box<dyn Iterator<Item = (&'a V, Option<(usize, f32)>)> + 'a>,
    )> {
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
        let coreset: Vec<V> = disks
            .iter()
            .flat_map(|disk| {
                let mut is: Vec<V> = matroid
                    .maximal_independent_set(&disk)
                    .into_iter()
                    .cloned()
                    .collect();

                let closest: Vec<(usize, f32)> = disk
                    .iter()
                    .map(|p| {
                        is.iter()
                            .enumerate()
                            .map(|(i, c)| (i, p.distance(c)))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap()
                    })
                    .collect();

                for (i, c) in is.iter_mut().enumerate() {
                    let multiplicity = closest.iter().filter(|pair| pair.0 == i).count() as u32;
                    c.set_weight(multiplicity);
                }

                is.into_iter()
            })
            .collect();

        // Ok(robust_matroid_center(&coreset, matroid, p))
        todo!()
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
