use kcmkc_base::types::Distance;

pub fn kcenter<'a, V: Distance>(
    points: &'a [V],
    k: usize,
) -> (
    Vec<&'a V>,
    Box<dyn Iterator<Item = (&'a V, usize, f32)> + 'a>,
) {
    if points.len() < k {
        // Trivial case, all points are centers
        return (
            points.iter().collect(),
            Box::new(points.iter().enumerate().map(|(i, p)| (p, i, 0.0))),
        );
    }
    let mut closest = vec![std::f32::INFINITY; points.len()];
    let mut assignments = vec![0usize; points.len()];
    let mut centers = Vec::with_capacity(k);

    centers.push(&points[0]);
    for i in 0..k {
        let c = centers[i];

        // Initialize the farthest as the center itself
        let mut farthest = i;
        let mut farthest_dist = 0.0f32;

        // Look for the farthest, updating distances on the go
        for (j, p) in points.iter().enumerate() {
            let d = c.distance(p);
            if d < closest[j] {
                closest[j] = d;
                assignments[j] = i;
            }
            if d > farthest_dist {
                farthest_dist = d;
                farthest = j;
            }
        }

        if i < k - 1 {
            // Set up the center for the next iteration
            centers.push(&points[farthest]);
        }
    }

    assert!(centers.len() == k);

    (
        centers,
        Box::new(
            points
                .iter()
                .zip(assignments.into_iter())
                .zip(closest.into_iter())
                .map(|((v, c_idx), d)| (v, c_idx, d)),
        ),
    )
}
