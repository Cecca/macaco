use std::marker::PhantomData;

pub trait Matroid<T>
where
    T: Clone,
{
    fn is_independent(&self, set: &[T]) -> bool;

    /// Implementation of the (unweighted) greedy algorithm
    /// for the maximal independent set:
    ///
    ///  - start with an empty set
    ///  - while there are elements to add
    ///    - add the next element to the set if this keeps the set independent
    fn maximal_independent_set(&self, set: &[T]) -> Vec<T> {
        let mut is = Vec::new();

        for x in set {
            is.push(x.clone());
            if !self.is_independent(&is) {
                is.pop();
            }
        }

        is
    }
}

/// Element of a set on which we can impose a transversal matroid
pub trait TransveralMatroidElement: Clone {
    fn topics<'a>(&'a self) -> Box<dyn Iterator<Item = u32> + 'a>;
}

pub struct TransveralMatroid<T> {
    topics: Vec<u32>,
    _marker: PhantomData<T>,
}

impl<T: TransveralMatroidElement> Matroid<T> for TransveralMatroid<T> {
    fn is_independent(&self, set: &[T]) -> bool {
        set.len() < self.topics.len() && self.maximum_matching(set).count() == set.len()
    }
}

impl<T: TransveralMatroidElement> TransveralMatroid<T> {
    pub fn new(topics: Vec<u32>) -> Self {
        Self {
            topics,
            _marker: PhantomData,
        }
    }

    // Return the indices of the elements in `set` that form a maximum matching
    // wrt the ground topics
    fn maximum_matching(&self, set: &[T]) -> impl Iterator<Item = usize> {
        let mut visited = vec![false; self.topics.len()];
        let mut representatives = vec![None; self.topics.len()];

        for idx in 0..set.len() {
            // reset the flags
            for flag in visited.iter_mut() {
                *flag = false;
            }
            // try to accomodate the new element
            self.find_matching_for(set, idx, &mut representatives, &mut visited);
        }

        representatives.into_iter().filter_map(|opt| opt)
    }

    fn find_matching_for(
        &self,
        set: &[T],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for (topic_idx, topic) in self.topics.iter().enumerate() {
            if set[idx].topics().find(|t| t == topic).is_some() && !visited[topic_idx] {
                visited[topic_idx] = true;
                let can_set = if let Some(displacing_idx) = representatives[topic_idx] {
                    // try to move the representative to another set
                    self.find_matching_for(set, displacing_idx, representatives, visited)
                } else {
                    true
                };

                if can_set {
                    representatives[topic_idx].replace(idx);
                    return true;
                }
            }
        }

        false
    }
}

/// Element of a set on which we can impose a partition matroid
pub trait PartitionMatroidElement: Clone {
    fn category<'a>(&'a self) -> usize;
}

pub struct PartitionMatroid<T: PartitionMatroidElement> {
    categories: Vec<usize>,
    _marker: PhantomData<T>,
}

impl<T: PartitionMatroidElement> Matroid<T> for PartitionMatroid<T> {
    fn is_independent(&self, set: &[T]) -> bool {
        let mut counts = self.categories.clone();
        for x in set {
            let cat = x.category();
            if counts[cat] == 0 {
                return false;
            } else {
                counts[cat] -= 1;
            }
        }
        true
    }
}

pub trait Weight {
    fn weight(&self) -> u32;
}

impl Weight for (usize, &Vec<usize>) {
    fn weight(&self) -> u32 {
        self.1.len() as u32
    }
}

pub fn weighted_matroid_intersection<'a, V: Clone + Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &'a [V],
    m1: &M1,
    m2: &M2,
) -> impl Iterator<Item = &'a V> + 'a {
    let mut independent_set = vec![false; set.len()];
    let mut last = 0;
    while augment(set, m1, m2, &mut independent_set) {
        let current_size = independent_set.iter().filter(|included| **included).count();
        println!(
            "      Independent set of size {} and weight {}",
            current_size,
            independent_set
                .iter()
                .enumerate()
                .filter(|(_, included)| **included)
                .map(|(i, _)| set[i].weight())
                .sum::<u32>()
        );
        assert!(current_size > last);
        last = current_size;
    }
    independent_set
        .into_iter()
        .zip(set.iter())
        .filter(|p| p.0)
        .map(|p| p.1)
}

/// Augment the given independent set in place. If there is no common independent set larger than the given one,
/// return false, otherwise return true
fn augment<'a, V: Clone + Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &[V],
    m1: &M1,
    m2: &M2,
    independent_set: &mut [bool],
) -> bool {
    let n = set.len();
    let graph = ExchangeGraph::new(set, m1, m2, independent_set);

    let mut independent_set_elements: Vec<V> = independent_set
        .iter()
        .zip(set.iter())
        .filter(|p| *p.0)
        .map(|p| p.1.clone())
        .collect();

    // define the source and destination sets.
    // When the input independent set is empty, it makes sense that the sets
    // x1 and x2 are equal, and corresponding to the full set.
    let x1: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(set[*i].clone());
                let b = m1.is_independent(&independent_set_elements);
                independent_set_elements.pop();
                b
            }
        })
        .map(|p| p.0)
        .collect();
    let x2: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(set[*i].clone());
                let b = m2.is_independent(&independent_set_elements);
                independent_set_elements.pop();
                b
            }
        })
        .map(|p| p.0)
        .collect();

    // find the best path, if any
    if let Some((_, path)) = x1
        .iter()
        .flat_map(|i| graph.bellman_ford(*i, &x2))
        .min_by_key(|(d, path)| (*d, path.len()))
    {
        println!("     Augmenting path: {:?}", path);
        for i in path {
            assert!(
                !independent_set[i],
                "nodes in the augmenting path should not be already in the independent set"
            );
            independent_set[i] = true;
        }
        true
    } else {
        false
    }
}

struct ExchangeGraph {
    length: Vec<i32>,
    edges: Vec<(usize, usize)>,
}

impl ExchangeGraph {
    fn new<'a, V: Clone + Weight, M1: Matroid<V>, M2: Matroid<V>>(
        set: &[V],
        m1: &M1,
        m2: &M2,
        independent_set: &mut [bool],
    ) -> Self {
        let length = set
            .iter()
            .zip(independent_set.iter())
            .map(|(v, included)| {
                if *included {
                    v.weight() as i32
                } else {
                    -(v.weight() as i32)
                }
            })
            .collect();

        let mut edges = Vec::new();

        // y is an element in the independent set, x is an element outside of the independent set
        for (y, _) in independent_set.iter().enumerate().filter(|p| *p.1) {
            // The independent set without J
            let mut scratch: Vec<V> = independent_set
                .iter()
                .enumerate()
                .filter(|p| *p.1 && p.0 != y)
                .map(|p| set[p.0].clone())
                .collect();
            for (x, _) in independent_set.iter().enumerate().filter(|p| !p.1) {
                scratch.push(set[x].clone());
                if m1.is_independent(&scratch) {
                    edges.push((y, x));
                }
                if m2.is_independent(&scratch) {
                    edges.push((x, y));
                }
                scratch.pop();
            }
        }
        edges.sort_by_key(|(u, v)| pair_to_zorder((*u as u32, *v as u32)));

        Self { length, edges }
    }

    /// return the shortest of all the shortest paths starting from `src`
    /// and going to `dsts` as a sequence of indices.
    /// Ties are broken by picking the one with fewest edges.
    ///
    /// If no path exist, then None is returned
    fn bellman_ford(&self, src: usize, dsts: &[usize]) -> Option<(i32, Vec<usize>)> {
        let n = self.length.len();
        let mut distance: Vec<Option<i32>> = vec![None; n];
        let mut predecessor: Vec<Option<usize>> = vec![None; n];

        distance[src].replace(self.length[src]);

        // compute shortest paths
        for _ in 0..n {
            let mut updated = false;
            for &(u, v) in &self.edges {
                // edge relaxation
                if let Some(du) = distance[u] {
                    if let Some(dv) = distance[v] {
                        if du + self.length[v] < dv {
                            updated = true;
                            distance[v].replace(du + self.length[v]);
                            predecessor[v].replace(u);
                        }
                    } else {
                        updated = true;
                        distance[v].replace(du + self.length[v]);
                        predecessor[v].replace(u);
                    }
                }
            }
            if !updated {
                // Early break if no nodes are updated: it means we explored all paths.
                break;
            }
        }

        // Iterator on the paths reaching `i`
        let iter_path = |i: usize| {
            let mut current = Some(i);
            std::iter::from_fn(move || {
                if let Some(i) = current {
                    let toret = current;
                    current = predecessor[i];
                    toret
                } else {
                    None
                }
            })
        };

        // Check the lengths of the paths
        #[cfg(debug_assertions)]
        for dst in dsts.iter() {
            if let Some(d) = distance[*dst] {
                let path: Vec<usize> = iter_path.clone()(*dst).collect();
                let weights: Vec<i32> = path.iter().map(|v| self.length[*v]).collect();
                let w = weights.iter().sum::<i32>();
                assert!(w == d);
            }
        }

        // using flat map we filter out unreachable destinations
        if let Some(shortest_dist) = dsts.iter().flat_map(|i| distance[*i]).min() {
            // Look, among the destinations
            dsts.iter()
                // for the ones at minimum distance
                .filter(|i| distance[**i].is_some() && distance[**i].unwrap() == shortest_dist)
                // and reached with minimum number of steps
                .min_by_key(|i| iter_path.clone()(**i).count())
                // for that one, materialize the path
                .map(|i| {
                    let path: Vec<usize> = iter_path.clone()(*i).collect();
                    assert!(path.len() > 0);
                    (distance[*i].unwrap(), path)
                })
        } else {
            None
        }
    }
}

/// Interleave the bits of the pairs. Using the resulting
/// number as a sorting key improves cache locality
#[inline]
pub fn pair_to_zorder((mut x, mut y): (u32, u32)) -> u64 {
    let mut z = 0;
    let msb_mask = 1_u32 << 31;
    for _ in 0..32 {
        if x & msb_mask == 0 {
            z = z << 1;
        } else {
            z = (z << 1) | 1;
        }
        if y & msb_mask == 0 {
            z = z << 1;
        } else {
            z = (z << 1) | 1;
        }
        x = x << 1;
        y = y << 1;
    }
    z
}
