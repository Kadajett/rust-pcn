//! Buffer pool for reusing pre-allocated `State` objects.
//!
//! # Motivation
//!
//! During training, each sample requires a `State` with 3*(L+1) `Array1<f32>` vectors.
//! Without pooling, this means ~5 allocations per sample per step. With the buffer pool,
//! we pre-allocate a fixed number of States and reuse them across epochs, reducing
//! allocations from ~5 per step to ~0 per sample (after warmup).
//!
//! # Thread Safety
//!
//! The pool uses `Mutex<Vec<State>>` for thread-safe access from Rayon worker threads.
//! Each thread can `get()` a state, use it, and `return_to_pool()` it when done.

use crate::core::State;
use ndarray::Array1;
use std::sync::Mutex;

/// Pre-allocated pool of `State` buffers for zero-allocation training loops.
///
/// The pool holds a stack of ready-to-use `State` objects. When a caller
/// requests a state via `get()`, the pool either pops one from the stack
/// (zero allocation) or creates a new one if the pool is empty.
///
/// Returned states have their vectors zeroed but the underlying memory is
/// reused, avoiding heap allocation/deallocation churn.
pub struct BufferPool {
    /// Stack of available pre-allocated states (LIFO for cache locality)
    states: Mutex<Vec<State>>,
    /// Network layer dimensions for allocation
    dims: Vec<usize>,
    /// Total states ever created (for diagnostics)
    total_allocated: Mutex<usize>,
    /// Number of times `get()` was satisfied from pool (cache hits)
    pool_hits: Mutex<usize>,
    /// Number of times `get()` required a new allocation (cache misses)
    pool_misses: Mutex<usize>,
}

/// Statistics from the buffer pool for performance diagnostics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total states created over the pool's lifetime
    pub total_allocated: usize,
    /// Number of `get()` calls satisfied from the pool
    pub hits: usize,
    /// Number of `get()` calls that required new allocation
    pub misses: usize,
    /// Current number of states available in the pool
    pub available: usize,
    /// Hit rate as a fraction `[0.0, 1.0]`
    pub hit_rate: f32,
}

#[allow(clippy::must_use_candidate)]
impl BufferPool {
    /// Create a new buffer pool pre-allocated with `capacity` `State` objects.
    ///
    /// # Performance
    /// Pre-allocating enough states for the batch size eliminates all
    /// per-sample allocations during training.
    pub fn new(dims: &[usize], capacity: usize) -> Self {
        let mut states = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            states.push(Self::allocate_state(dims));
        }

        Self {
            states: Mutex::new(states),
            dims: dims.to_vec(),
            total_allocated: Mutex::new(capacity),
            pool_hits: Mutex::new(0),
            pool_misses: Mutex::new(0),
        }
    }

    /// Create a new empty pool (no pre-allocation).
    ///
    /// States will be allocated on first `get()` and reused after `return_to_pool()`.
    pub fn empty(dims: &[usize]) -> Self {
        Self {
            states: Mutex::new(Vec::new()),
            dims: dims.to_vec(),
            total_allocated: Mutex::new(0),
            pool_hits: Mutex::new(0),
            pool_misses: Mutex::new(0),
        }
    }

    /// Get a zeroed `State` from the pool, or allocate a new one if empty.
    ///
    /// The returned state has all vectors (x, mu, eps) zeroed and
    /// `steps_taken` / `final_energy` reset to 0.
    pub fn get(&self) -> State {
        let maybe_state = {
            let mut pool = self
                .states
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            pool.pop()
        };

        #[allow(clippy::option_if_let_else, clippy::single_match_else)]
        match maybe_state {
            Some(mut state) => {
                Self::zero_state(&mut state);
                if let Ok(mut hits) = self.pool_hits.lock() {
                    *hits += 1;
                }
                state
            }
            None => {
                if let Ok(mut total) = self.total_allocated.lock() {
                    *total += 1;
                }
                if let Ok(mut misses) = self.pool_misses.lock() {
                    *misses += 1;
                }
                Self::allocate_state(&self.dims)
            }
        }
    }

    /// Return a used `State` to the pool for reuse.
    pub fn return_to_pool(&self, state: State) {
        let mut pool = self
            .states
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pool.push(state);
    }

    /// Return multiple states to the pool at once (batch return).
    pub fn return_batch(&self, states: Vec<State>) {
        let mut pool = self
            .states
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pool.extend(states);
    }

    /// Get pool performance statistics.
    pub fn stats(&self) -> PoolStats {
        let total = self.total_allocated.lock().map_or(0, |g| *g);
        let hits = self.pool_hits.lock().map_or(0, |g| *g);
        let misses = self.pool_misses.lock().map_or(0, |g| *g);
        let available = self.states.lock().map_or(0, |g| g.len());
        let total_gets = hits + misses;
        #[allow(clippy::cast_precision_loss)]
        let hit_rate = if total_gets > 0 {
            hits as f32 / total_gets as f32
        } else {
            0.0
        };

        PoolStats {
            total_allocated: total,
            hits,
            misses,
            available,
            hit_rate,
        }
    }

    /// Current number of available states in the pool.
    pub fn available(&self) -> usize {
        self.states.lock().map_or(0, |g| g.len())
    }

    /// Allocate a fresh zeroed `State` for the given dimensions.
    fn allocate_state(dims: &[usize]) -> State {
        let l_max = dims.len() - 1;
        State {
            x: (0..=l_max).map(|l| Array1::zeros(dims[l])).collect(),
            mu: (0..=l_max).map(|l| Array1::zeros(dims[l])).collect(),
            eps: (0..=l_max).map(|l| Array1::zeros(dims[l])).collect(),
            steps_taken: 0,
            final_energy: 0.0,
        }
    }

    /// Zero all vectors in a state without deallocating the underlying memory.
    fn zero_state(state: &mut State) {
        for x in &mut state.x {
            x.fill(0.0);
        }
        for mu in &mut state.mu {
            mu.fill(0.0);
        }
        for eps in &mut state.eps {
            eps.fill(0.0);
        }
        state.steps_taken = 0;
        state.final_energy = 0.0;
    }
}

// `BufferPool` is automatically `Send` + `Sync` because:
// - `Mutex<Vec<State>>` is `Send` + `Sync` (`State` contains `Vec<Array1<f32>>` which is `Send`)
// - `Vec<usize>` is `Send` + `Sync`
// - `Mutex<usize>` is `Send` + `Sync`
// No unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = BufferPool::new(&[2, 4, 1], 8);
        assert_eq!(pool.available(), 8);
    }

    #[test]
    fn test_pool_get_returns_zeroed_state() {
        let pool = BufferPool::new(&[2, 3, 1], 4);
        let state = pool.get();

        assert_eq!(state.x.len(), 3);
        assert_eq!(state.x[0].len(), 2);
        assert_eq!(state.x[1].len(), 3);
        assert_eq!(state.x[2].len(), 1);

        for x in &state.x {
            for &v in x.iter() {
                assert_eq!(v, 0.0);
            }
        }
        assert_eq!(state.steps_taken, 0);
        assert_eq!(state.final_energy, 0.0);
    }

    #[test]
    fn test_pool_get_and_return() {
        let pool = BufferPool::new(&[2, 3, 1], 4);
        assert_eq!(pool.available(), 4);

        let state1 = pool.get();
        assert_eq!(pool.available(), 3);

        let state2 = pool.get();
        assert_eq!(pool.available(), 2);

        pool.return_to_pool(state1);
        assert_eq!(pool.available(), 3);

        pool.return_to_pool(state2);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_exhaustion_and_reallocation() {
        let pool = BufferPool::new(&[2, 1], 2);

        let s1 = pool.get();
        let s2 = pool.get();
        assert_eq!(pool.available(), 0);

        // Pool exhausted: should allocate new
        let s3 = pool.get();
        assert_eq!(s3.x[0].len(), 2);

        pool.return_to_pool(s1);
        pool.return_to_pool(s2);
        pool.return_to_pool(s3);
        assert_eq!(pool.available(), 3);
    }

    #[test]
    fn test_pool_stats() {
        let pool = BufferPool::new(&[2, 1], 2);

        let s1 = pool.get();
        let s2 = pool.get();
        let s3 = pool.get(); // miss

        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 3);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 1e-5);

        pool.return_to_pool(s1);
        pool.return_to_pool(s2);
        pool.return_to_pool(s3);
    }

    #[test]
    fn test_pool_return_batch() {
        let pool = BufferPool::new(&[2, 1], 4);

        let states: Vec<State> = (0..4).map(|_| pool.get()).collect();
        assert_eq!(pool.available(), 0);

        pool.return_batch(states);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_reuse_zeroes_state() {
        let pool = BufferPool::new(&[2, 3], 1);

        let mut state = pool.get();
        state.x[0].fill(999.0);
        state.x[1].fill(-999.0);
        state.eps[0].fill(42.0);
        state.steps_taken = 100;
        state.final_energy = 50.0;

        pool.return_to_pool(state);
        let clean_state = pool.get();

        for x in &clean_state.x {
            for &v in x.iter() {
                assert_eq!(v, 0.0, "State should be zeroed on reuse");
            }
        }
        assert_eq!(clean_state.steps_taken, 0);
        assert_eq!(clean_state.final_energy, 0.0);
    }

    #[test]
    fn test_empty_pool() {
        let pool = BufferPool::empty(&[3, 2, 1]);
        assert_eq!(pool.available(), 0);

        let state = pool.get();
        assert_eq!(state.x[0].len(), 3);

        let stats = pool.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        pool.return_to_pool(state);

        let _state2 = pool.get();
        let stats2 = pool.stats();
        assert_eq!(stats2.hits, 1);
    }
}
