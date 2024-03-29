use macaco_base::types::Distance;

/// Solve the k-center clustering by means of the farthest-first heuristic.
///
/// Returns a vector of refernces to the cluster centers, and an iterator
/// of assignments of points to clusters. The first element of each
/// item is a reference to the point, the second is the index of the cluster
/// center (in the centers vector), and the third is the distance
/// between the point and its center.
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
    let n = points.len();
    let timer = std::time::Instant::now();
    let mut min_dist = vec![std::f32::INFINITY; points.len()];
    let mut assignments = vec![0usize; points.len()];
    let mut centers = Vec::with_capacity(k);

    centers.push(&points[0]);
    for i in 0..k {
        let c = centers[i];

        let mut farthest = i;
        let mut farthest_dist = 0.0f32;

        // let timer = std::time::Instant::now();
        // Look for the farthest, updating distances on the go
        // for (j, p) in points.iter().enumerate() {
        for j in 0..n {
            let p = unsafe { &points.get_unchecked(j) };
            let d = c.distance(p);
            // assert!(d.is_finite());
            if d < min_dist[j] {
                min_dist[j] = d;
                assignments[j] = i;
            }
            if min_dist[j] > farthest_dist {
                farthest_dist = min_dist[j];
                farthest = j;
            }
        }
        // let elapsed = timer.elapsed();
        // println!(
        //     "  [it {}] iterating over {} points {:?} ({:?} per point)",
        //     i,
        //     points.len(),
        //     elapsed,
        //     elapsed / points.len() as u32
        // );

        if i < k - 1 {
            // Set up the center for the next iteration
            centers.push(&points[farthest]);
        }
    }

    println!(
        "Radius of clustering is {} ({:?} on {} points with {} centers)",
        min_dist
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        timer.elapsed(),
        points.len(),
        k
    );
    assert!(
        centers.len() == k,
        "expected centers.len() == k == {}, but got {} != {}",
        k,
        centers.len(),
        k
    );

    (
        centers,
        Box::new(
            points
                .iter()
                .zip(assignments.into_iter())
                .zip(min_dist.into_iter())
                .map(|((v, c_idx), d)| (v, c_idx, d)),
        ),
    )
}
