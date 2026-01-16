//! Fast XORShift64* random number generator.
//!
//! This RNG is not cryptographically secure but is very fast and has
//! good statistical properties for game simulations.

/// A fast XORShift64* random number generator
#[derive(Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    /// Create a new RNG with the given seed
    #[inline]
    pub fn new(seed: u64) -> Self {
        // Ensure seed is not zero
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Create a new RNG seeded from system time
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);
        Self::new(seed)
    }

    /// Generate the next random u64
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // XORShift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Generate a random u32
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate a random number in range [0, max)
    #[inline]
    pub fn gen_range(&mut self, min: u32, max: u32) -> u32 {
        debug_assert!(max > min);
        let range = max - min;
        min + (self.next_u32() % range)
    }

    /// Generate a random float in [0, 1)
    #[inline]
    pub fn gen_float(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Shuffle a slice in place (Fisher-Yates)
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.gen_range(0, (i + 1) as u32) as usize;
            slice.swap(i, j);
        }
    }
}

impl Default for FastRng {
    fn default() -> Self {
        Self::from_entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = FastRng::new(12345);
        let mut rng2 = FastRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = FastRng::new(12345);

        for _ in 0..1000 {
            let val = rng.gen_range(0, 52);
            assert!(val < 52);
        }
    }

    #[test]
    fn test_rng_distribution() {
        let mut rng = FastRng::new(12345);
        let mut counts = [0u32; 10];

        for _ in 0..10000 {
            let val = rng.gen_range(0, 10);
            counts[val as usize] += 1;
        }

        // Each bucket should have roughly 1000 hits
        for count in counts {
            assert!(count > 800 && count < 1200);
        }
    }

    #[test]
    fn test_shuffle() {
        let mut rng = FastRng::new(12345);
        let mut arr: Vec<u32> = (0..52).collect();
        rng.shuffle(&mut arr);

        // Should still contain all elements
        let mut sorted = arr.clone();
        sorted.sort();
        assert_eq!(sorted, (0..52).collect::<Vec<_>>());

        // Should be shuffled (not in original order)
        assert_ne!(arr, (0..52).collect::<Vec<_>>());
    }
}
