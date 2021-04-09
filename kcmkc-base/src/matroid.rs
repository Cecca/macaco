use log::*;
use rayon::prelude::*;
use std::time::Instant;
use std::{cell::RefCell, rc::Rc, sync::Arc};
use std::{collections::HashMap, marker::PhantomData};
use thread_local::ThreadLocal;

use crate::perf_counters;

pub trait Matroid<T> {
    fn is_independent(&self, set: &[T]) -> bool;
    fn is_independent_ref(&self, set: &[&T]) -> bool;

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
            if !self.is_independent_ref(&is) {
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
            if self.is_independent_ref(&is) {
                return false;
            }
        }
        true
    }
}

pub fn augment<T: Clone + PartialEq>(
    matroid: Rc<dyn Matroid<T>>,
    independent_set: &[T],
    set: &[&T],
) -> Vec<T> {
    use std::iter::FromIterator;
    let mut is = Vec::from_iter(independent_set.iter());
    for x in set {
        if !is.contains(&x) {
            is.push(x);
            if !matroid.is_independent_ref(&is) {
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
    scratch_visited: ThreadLocal<RefCell<Vec<bool>>>,
    scratch_representatives: ThreadLocal<RefCell<Vec<Option<usize>>>>,
}

impl<T: TransveralMatroidElement> Matroid<T> for TransveralMatroid<T> {
    fn rank(&self) -> usize {
        self.topics.len()
    }

    fn is_independent(&self, set: &[T]) -> bool {
        perf_counters::inc_matroid_oracle_count();
        // FIXME: find a way to remove this allocation, if it is a bottleneck
        let set: Vec<&T> = set.iter().collect();
        set.len() < self.topics.len() && self.maximum_matching_size(&set) == set.len()
    }

    fn is_independent_ref(&self, set: &[&T]) -> bool {
        perf_counters::inc_matroid_oracle_count();
        set.len() < self.topics.len() && self.maximum_matching_size(set) == set.len()
    }
}

impl<T: TransveralMatroidElement> TransveralMatroid<T> {
    pub fn new(topics: Vec<u32>) -> Self {
        Self {
            topics,
            _marker: PhantomData,
            scratch_visited: ThreadLocal::new(),
            scratch_representatives: ThreadLocal::new(),
        }
    }

    fn maximum_matching_size(&self, set: &[&T]) -> usize {
        let n_topics = self.topics.len();
        let mut visited = self
            .scratch_visited
            .get_or(|| RefCell::new(vec![false; n_topics]))
            .borrow_mut();
        visited.fill(false);
        let mut representatives = self
            .scratch_representatives
            .get_or(|| RefCell::new(vec![None; self.topics.len()]))
            .borrow_mut();
        representatives.fill(None);

        for idx in 0..set.len() {
            // reset the flags
            visited.fill(false);
            // try to accomodate the new element
            self.find_matching_for(set, idx, &mut representatives, &mut visited);
        }

        representatives.iter().filter(|opt| opt.is_some()).count()
    }

    fn find_matching_for(
        &self,
        set: &[&T],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for (topic_idx, topic) in self.topics.iter().enumerate() {
            if !visited[topic_idx] && set[idx].topics().iter().find(|t| *t == topic).is_some() {
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

    fn is_independent(&self, set: &[T]) -> bool {
        perf_counters::inc_matroid_oracle_count();
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

    fn is_independent_ref(&self, set: &[&T]) -> bool {
        perf_counters::inc_matroid_oracle_count();
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
    let mut graph = ExchangeGraph::default();
    let mut last = 0;
    while augment_intersection(set, m1, m2, &mut independent_set, &mut graph) {
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
            debug_assert!(m1.is_independent_ref(&current_items));
            debug_assert!(m2.is_independent_ref(&current_items));
        }
        debug!(
            "Independent set of size {} and weight {}",
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
    graph: &mut ExchangeGraph,
) -> bool {
    graph.reset_to(set, m1, m2, independent_set);

    let timer = Instant::now();
    let mut independent_set_elements: Vec<&V> = independent_set
        .iter()
        .zip(set.iter())
        .filter(|p| *p.0)
        .map(|p| p.1)
        .collect();
    debug_assert!(m1.is_independent_ref(&independent_set_elements));
    debug_assert!(m2.is_independent_ref(&independent_set_elements));
    debug!(
        "Retrieved references to independent set elements in {:?}",
        timer.elapsed()
    );

    let timer = Instant::now();
    // define the source and destination sets.
    // When the input independent set is empty, it makes sense that the sets
    // x1 and x2 are equal, and corresponding to the full set.
    let x1: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(&set[*i]);
                let b = m1.is_independent_ref(&independent_set_elements);
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
                let b = m2.is_independent_ref(&independent_set_elements);
                independent_set_elements.pop();
                b
            }
        })
        .map(|p| p.0)
        .collect();

    // let tl_graph = Arc::new(thread_local::ThreadLocal::new());

    debug!(
        "looking for best paths from {} sources to {} destinations (defined in {:?})",
        x1.len(),
        x2.len(),
        timer.elapsed()
    );
    let timer = Instant::now();
    // find the best path, if any
    let res = if x1.len() < x2.len() {
        // more destinations than sources
        if let Some((_, path)) = x1
            .iter()
            .flat_map(|i| graph.bellman_ford(*i, &x2))
            .min_by_key(|(d, path)| (*d, path.len()))
        {
            for i in path {
                // Computing the xor on the flags array is equivalent to computing the
                // symmetric difference of the path and the independent set
                independent_set[i] ^= true;
            }
            true
        } else {
            false
        }
    } else {
        // more sources than destinations
        debug!("more sources than destinations, flipping the edges");
        graph.reverse(); // flip the edges first
        if let Some((_, path)) = x2
            .iter()
            .flat_map(|i| graph.bellman_ford(*i, &x1))
            .min_by_key(|(d, path)| (*d, path.len()))
        {
            for i in path {
                // Computing the xor on the flags array is equivalent to computing the
                // symmetric difference of the path and the independent set
                independent_set[i] ^= true;
            }
            true
        } else {
            false
        }
    };
    debug!("augmenting path found (if any) in {:?}", timer.elapsed());
    res
}

#[derive(Clone, Default)]
struct ExchangeGraph {
    reversed: bool,
    length: Vec<i32>,
    edges: Vec<(usize, usize)>,
    distance: Vec<Option<i32>>,
    predecessor: Vec<Option<usize>>,
}

impl ExchangeGraph {
    fn reset_to<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
        &mut self,
        set: &[V],
        m1: &M1,
        m2: &M2,
        independent_set: &mut [bool],
    ) {
        self.reversed = false;
        self.length.clear();
        self.edges.clear();
        self.distance.clear();
        self.predecessor.clear();
        let timer = std::time::Instant::now();
        let n = set.len();

        // reuse already allocated space, if any
        self.length
            .extend(set.iter().zip(independent_set.iter()).map(|(v, included)| {
                if *included {
                    v.weight() as i32
                } else {
                    -(v.weight() as i32)
                }
            }));
        debug!("computed lengths in {:?}", timer.elapsed());

        // edge building. This is the costly operation.
        //
        // edge invariants, where I is the independent set:
        //  - (y, x) is in the graph iff I - y + x is independent in m1
        //  - (x, y) is in the graph iff I - y + x is independent in m2
        let timer = std::time::Instant::now();
        // y is an element in the independent set, x is an element outside of the independent set
        for (y, _) in independent_set.iter().enumerate().filter(|p| *p.1) {
            // The independent set without y
            let mut scratch: Vec<&V> = independent_set
                .iter()
                .enumerate()
                .filter(|p| *p.1 && p.0 != y)
                .map(|p| &set[p.0])
                .collect();
            for (x, _) in independent_set.iter().enumerate().filter(|p| !p.1) {
                scratch.push(&set[x]);
                if m1.is_independent_ref(&scratch) {
                    self.edges.push((y, x));
                }
                if m2.is_independent_ref(&scratch) {
                    self.edges.push((x, y));
                }
                scratch.pop();
            }
        }
        debug!("created edges in {:?}", timer.elapsed());
        let timer = std::time::Instant::now();
        self.edges
            .sort_by_key(|(u, v)| pair_to_zorder((*u as u32, *v as u32)));
        debug!("sorted edges in z-order in {:?}", timer.elapsed());
        debug!(
            "Created exchange graph with {} edges and {} nodes in {:?}",
            self.edges.len(),
            n,
            timer.elapsed()
        );

        self.distance.resize(n, None);
        self.predecessor.resize(n, None);
    }

    /// reverse the arcs
    fn flip_edges(&mut self) {
        for edge in self.edges.iter_mut() {
            *edge = (edge.1, edge.0);
        }
        self.reversed = !self.reversed;
    }

    fn reverse(&mut self) {
        if !self.reversed {
            self.flip_edges();
        }
    }

    fn forward(&mut self) {
        if self.reversed {
            self.flip_edges();
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
    /// If no path exist, then None is returned.
    /// Otherwise, return the weight of the path and the sequence of its nodes
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
