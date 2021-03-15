use std::{collections::HashMap, marker::PhantomData};

pub trait Matroid<T> {
    fn is_independent(&self, set: &[&T]) -> bool;

    fn rank(&self) -> usize;

    /// Implementation of the (unweighted) greedy algorithm
    /// for the maximal independent set:
    ///
    ///  - start with an empty set
    ///  - while there are elements to add
    ///    - add the next element to the set if this keeps the set independent
    fn maximal_independent_set<'a>(&self, set: &[&'a T]) -> Vec<&'a T> {
        let mut is = Vec::new();

        for x in set {
            is.push(*x);
            if !self.is_independent(&is) {
                is.pop();
            }
        }

        is
    }

    fn is_maximal(&self, is: &[T], set: &[T]) -> bool {
        use std::iter::FromIterator;
        let mut is = Vec::from_iter(is.iter());
        for x in set {
            is.push(x);
            if self.is_independent(&is) {
                return false;
            }
        }
        true
    }
}

pub fn augment<T: Clone + PartialEq>(
    matroid: &Box<dyn Matroid<T>>,
    independent_set: &[T],
    set: &[T],
) -> Vec<T> {
    use std::iter::FromIterator;
    let mut is = Vec::from_iter(independent_set.iter());
    for x in set {
        if !is.contains(&x) {
            is.push(x);
            if !matroid.is_independent(&is) {
                is.pop();
            }
        }
    }

    is.into_iter().cloned().collect()
}

/// Element of a set on which we can impose a transversal matroid
pub trait TransveralMatroidElement {
    fn topics<'a>(&'a self) -> &'a [u32];
}

pub struct TransveralMatroid<T> {
    topics: Vec<u32>,
    _marker: PhantomData<T>,
}

impl<T: TransveralMatroidElement> Matroid<T> for TransveralMatroid<T> {
    fn rank(&self) -> usize {
        self.topics.len()
    }

    fn is_independent(&self, set: &[&T]) -> bool {
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
    fn maximum_matching(&self, set: &[&T]) -> impl Iterator<Item = usize> {
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
        set: &[&T],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for (topic_idx, topic) in self.topics.iter().enumerate() {
            if set[idx].topics().iter().find(|t| *t == topic).is_some() && !visited[topic_idx] {
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
pub trait PartitionMatroidElement {
    fn category<'a>(&'a self) -> &'a String;
}

pub struct PartitionMatroid<T: PartitionMatroidElement> {
    categories: HashMap<String, u32>,
    _marker: PhantomData<T>,
}

impl<T: PartitionMatroidElement> PartitionMatroid<T> {
    pub fn new(categories: HashMap<String, u32>) -> Self {
        Self {
            categories,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: PartitionMatroidElement> Matroid<T> for PartitionMatroid<T> {
    fn rank(&self) -> usize {
        self.categories.values().sum::<u32>() as usize
    }

    fn is_independent(&self, set: &[&T]) -> bool {
        let mut counts = self.categories.clone();
        for x in set {
            let cat = x.category();
            // Categories not explicitly mentioned in the matroid
            // default to a limit of 0. This makes for a less verbose specification
            // of constraints
            if counts.get(cat).unwrap_or(&0) == &0 {
                return false;
            } else {
                counts.get_mut(cat).map(|c| *c -= 1);
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

pub fn weighted_matroid_intersection<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &'a [V],
    m1: &M1,
    m2: &M2,
) -> impl Iterator<Item = &'a V> + 'a {
    let mut independent_set = vec![false; set.len()];
    let mut last = 0;
    while augment_intersection(set, m1, m2, &mut independent_set) {
        // All of the statements in this while body are for debug purposes
        let current_size = independent_set.iter().filter(|included| **included).count();
        #[cfg(debug_assertions)]
        {
            let current_items: Vec<&V> = independent_set
                .iter()
                .enumerate()
                .filter(|(_, included)| **included)
                .map(|(i, _)| &set[i])
                .collect();
            debug_assert!(m1.is_independent(&current_items));
            debug_assert!(m2.is_independent(&current_items));
        }
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
fn augment_intersection<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &[V],
    m1: &M1,
    m2: &M2,
    independent_set: &mut [bool],
) -> bool {
    let mut graph = ExchangeGraph::new(set, m1, m2, independent_set);

    let mut independent_set_elements: Vec<&V> = independent_set
        .iter()
        .zip(set.iter())
        .filter(|p| *p.0)
        .map(|p| p.1)
        .collect();
    debug_assert!(m1.is_independent(&independent_set_elements));
    debug_assert!(m2.is_independent(&independent_set_elements));

    // define the source and destination sets.
    // When the input independent set is empty, it makes sense that the sets
    // x1 and x2 are equal, and corresponding to the full set.
    let x1: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(&set[*i]);
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
                independent_set_elements.push(&set[*i]);
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
            // Computing the xor on the flags array is equivalent to computing the
            // symmetric difference of the path and the independent set
            independent_set[i] ^= true;
        }
        true
    } else {
        false
    }
}

struct ExchangeGraph {
    length: Vec<i32>,
    edges: Vec<(usize, usize)>,
    distance: Vec<Option<i32>>,
    predecessor: Vec<Option<usize>>,
}

impl ExchangeGraph {
    fn new<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
        set: &[V],
        m1: &M1,
        m2: &M2,
        independent_set: &mut [bool],
    ) -> Self {
        let n = set.len();
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
            let mut scratch: Vec<&V> = independent_set
                .iter()
                .enumerate()
                .filter(|p| *p.1 && p.0 != y)
                .map(|p| &set[p.0])
                .collect();
            for (x, _) in independent_set.iter().enumerate().filter(|p| !p.1) {
                scratch.push(&set[x]);
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

        let distance: Vec<Option<i32>> = vec![None; n];
        let predecessor: Vec<Option<usize>> = vec![None; n];

        Self {
            length,
            edges,
            distance,
            predecessor,
        }
    }

    /// Iterator on the paths reaching `i`
    fn iter_path<'a>(&'a self, i: usize) -> impl Iterator<Item = usize> + 'a {
        let mut current = Some(i);
        std::iter::from_fn(move || {
            if let Some(i) = current {
                let toret = current;
                current = self.predecessor[i];
                toret
            } else {
                None
            }
        })
    }

    /// return the shortest of all the shortest paths starting from `src`
    /// and going to `dsts` as a sequence of indices.
    /// Ties are broken by picking the one with fewest edges.
    ///
    /// If no path exist, then None is returned
    fn bellman_ford(&mut self, src: usize, dsts: &[usize]) -> Option<(i32, Vec<usize>)> {
        let n = self.length.len();

        // reset the support arrays
        self.distance.fill(None);
        self.predecessor.fill(None);

        self.distance[src].replace(self.length[src]);

        // compute shortest paths
        for _ in 0..n {
            let mut updated = false;
            for &(u, v) in &self.edges {
                // edge relaxation
                if let Some(du) = self.distance[u] {
                    if let Some(dv) = self.distance[v] {
                        if du + self.length[v] < dv {
                            updated = true;
                            self.distance[v].replace(du + self.length[v]);
                            self.predecessor[v].replace(u);
                        }
                    } else {
                        updated = true;
                        self.distance[v].replace(du + self.length[v]);
                        self.predecessor[v].replace(u);
                    }
                }
            }
            if !updated {
                // Early break if no nodes are updated: it means we explored all paths.
                break;
            }
        }

        // Check the lengths of the paths
        #[cfg(debug_assertions)]
        for dst in dsts.iter() {
            if let Some(d) = self.distance[*dst] {
                let path: Vec<usize> = self.iter_path(*dst).collect();
                let weights: Vec<i32> = path.iter().map(|v| self.length[*v]).collect();
                let w = weights.iter().sum::<i32>();
                assert!(w == d);
            }
        }

        // using flat map we filter out unreachable destinations
        if let Some(shortest_dist) = dsts.iter().flat_map(|i| self.distance[*i]).min() {
            // Look, among the destinations
            dsts.iter()
                // for the ones at minimum distance
                .filter(|i| {
                    self.distance[**i].is_some() && self.distance[**i].unwrap() == shortest_dist
                })
                // and reached with minimum number of steps
                .min_by_key(|i| self.iter_path(**i).count())
                // for that one, materialize the path
                .map(|i| {
                    let path: Vec<usize> = self.iter_path(*i).collect();
                    assert!(path.len() > 0);
                    (self.distance[*i].unwrap(), path)
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
