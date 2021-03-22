use kcmkc_base::types::*;

// Pre-compute all pairwise distances, and then keep the distances from each point in sorted order.
// This allows the retrieval of disks in time proportional to the size of the disk itself.
pub struct DiskBuilder {
    distances: Vec<Vec<(usize, f32)>>,
}

impl DiskBuilder {
    pub fn new<V: Distance>(points: &[V]) -> Self {
        println!("Pre-computing distances");
        let distances: Vec<Vec<(usize, f32)>> = points
            .iter()
            .map(|a| {
                let mut dists: Vec<(usize, f32)> = points
                    .iter()
                    .enumerate()
                    .map(|(j, b)| (j, a.distance(b)))
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists
            })
            .collect();
        Self { distances }
    }

    /// Invoke the provided function on a sequence of distances. If the functino
    /// returns `Error(uncovered)`, the radius is too small and we look for a larger one.
    /// Otherwise it may be too large, so we look for a smaller one.
    /// The search stops when the distance between radii is 0.1% of the
    /// maximum distance in the dataset.
    pub fn bynary_search_distances<O, F: FnMut(f32) -> Result<O, usize>>(&self, mut f: F) -> O {
        use std::collections::BTreeSet;
        println!("Sorting distances to get candidate radii");
        let dists: BTreeSet<OrderedF32> = self
            .distances
            .iter()
            .flat_map(|row| row.iter().filter(|f| f.1 > 0.0).map(|f| f.1.into()))
            .collect();
        let dists: Vec<OrderedF32> = dists.into_iter().collect();
        assert!(dists.len() > 0);
        let max_distance = dists.last().unwrap().0;
        let min_difference = 0.001 * max_distance;
        println!(
            "Max distance {}, min difference {}",
            max_distance, min_difference
        );

        // Binary search code adaptded from Rust's standard library
        let mut size = dists.len();
        let mut last_valid_solution: Option<O> = None;
        let mut last_distance = max_distance;
        let mut base = 0usize;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            if (dists[mid].0 - last_distance).abs() <= min_difference {
                println!("Early stop");
                break;
            }
            println!("Looking at distance {}", dists[mid].0);
            last_distance = dists[mid].0;
            let res = f(dists[mid].into());
            base = match res {
                Ok(res) => {
                    last_valid_solution.replace(res);
                    base
                }
                Err(_) => mid,
            };
            size -= half;
        }
        last_valid_solution.unwrap()
    }

    /// Get the indices of points in the ball of radius `r` around point `i`.
    /// The indices are not in sorted order, so to compute the intersection between disks
    /// in linear time we should first sort them!
    pub fn disk<'a>(&'a self, i: usize, r: f32) -> impl Iterator<Item = usize> + 'a {
        self.distances[i]
            .iter()
            .take_while(move |(_i, d)| *d <= r)
            .map(|pair| pair.0)
    }
}

pub fn intersection<I1: Iterator<Item = usize>, I2: Iterator<Item = usize>>(
    mut a: I1,
    mut b: I2,
) -> impl Iterator<Item = usize> {
    let mut next_a = a.next();
    let mut next_b = b.next();
    std::iter::from_fn(move || loop {
        if next_a.is_none() || next_b.is_none() {
            return None;
        }
        let idx_a = next_a.unwrap();
        let idx_b = next_b.unwrap();
        if idx_a < idx_b {
            next_a = a.next();
        } else if idx_a > idx_b {
            next_b = b.next();
        } else {
            let ret = idx_a;
            next_a = a.next();
            next_b = b.next();
            return Some(ret);
        }
    })
}
