//! Hand evaluation for Texas Hold'em.
//!
//! Uses a combination of lookup tables and direct calculation for fast
//! 5-card and 7-card hand evaluation.
//!
//! Hand ranks: 1 (Royal Flush) to 7462 (7-high), lower is better.

use crate::cards::{Card, Rank, Suit};
use std::sync::OnceLock;

/// Hand strength rank (1-7462, lower is stronger)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandRank(u16);

impl HandRank {
    // Hand category boundaries (inclusive upper bounds)
    pub const STRAIGHT_FLUSH_MAX: u16 = 10;
    pub const FOUR_OF_A_KIND_MAX: u16 = 166;
    pub const FULL_HOUSE_MAX: u16 = 322;
    pub const FLUSH_MAX: u16 = 1599;
    pub const STRAIGHT_MAX: u16 = 1609;
    pub const THREE_OF_A_KIND_MAX: u16 = 2467;
    pub const TWO_PAIR_MAX: u16 = 3325;
    pub const ONE_PAIR_MAX: u16 = 6185;
    pub const HIGH_CARD_MAX: u16 = 7462;

    #[inline]
    pub fn new(value: u16) -> Self {
        Self(value)
    }

    #[inline]
    pub fn value(self) -> u16 {
        self.0
    }

    /// Get the hand category
    pub fn category(self) -> HandCategory {
        match self.0 {
            1..=10 => HandCategory::StraightFlush,
            11..=166 => HandCategory::FourOfAKind,
            167..=322 => HandCategory::FullHouse,
            323..=1599 => HandCategory::Flush,
            1600..=1609 => HandCategory::Straight,
            1610..=2467 => HandCategory::ThreeOfAKind,
            2468..=3325 => HandCategory::TwoPair,
            3326..=6185 => HandCategory::OnePair,
            _ => HandCategory::HighCard,
        }
    }
}

/// Hand category enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandCategory {
    StraightFlush,
    FourOfAKind,
    FullHouse,
    Flush,
    Straight,
    ThreeOfAKind,
    TwoPair,
    OnePair,
    HighCard,
}

impl HandCategory {
    pub fn name(&self) -> &'static str {
        match self {
            Self::StraightFlush => "Straight Flush",
            Self::FourOfAKind => "Four of a Kind",
            Self::FullHouse => "Full House",
            Self::Flush => "Flush",
            Self::Straight => "Straight",
            Self::ThreeOfAKind => "Three of a Kind",
            Self::TwoPair => "Two Pair",
            Self::OnePair => "One Pair",
            Self::HighCard => "High Card",
        }
    }
}

// Lookup tables for hand evaluation
static FLUSH_LOOKUP: OnceLock<Box<[u16; 8192]>> = OnceLock::new();
static UNIQUE5_LOOKUP: OnceLock<Box<[u16; 8192]>> = OnceLock::new();
static PRIME_LOOKUP: OnceLock<std::collections::HashMap<u32, u16>> = OnceLock::new();

/// Initialize the lookup tables
pub fn init_evaluator() {
    FLUSH_LOOKUP.get_or_init(generate_flush_table);
    UNIQUE5_LOOKUP.get_or_init(generate_unique5_table);
    PRIME_LOOKUP.get_or_init(generate_prime_table);
}

/// The main evaluator struct
pub struct Evaluator;

impl Evaluator {
    /// Evaluate a 7-card hand, returning the best 5-card hand rank
    #[inline]
    pub fn evaluate_7(cards: &[Card; 7]) -> HandRank {
        init_evaluator();

        // Count cards per suit and collect rank bits per suit
        let mut suit_counts = [0u8; 4];
        let mut suit_bits = [0u16; 4];

        for card in cards {
            let suit_idx = card.suit().to_index() as usize;
            let rank_bit = 1u16 << (card.rank() as u16);
            suit_counts[suit_idx] += 1;
            suit_bits[suit_idx] |= rank_bit;
        }

        // Check for flush
        for i in 0..4 {
            if suit_counts[i] >= 5 {
                return Self::evaluate_flush_7(suit_bits[i], suit_counts[i]);
            }
        }

        // Non-flush: evaluate all 21 5-card combinations
        Self::evaluate_non_flush_7(cards)
    }

    /// Evaluate when we have 5+ cards of the same suit
    fn evaluate_flush_7(bits: u16, count: u8) -> HandRank {
        let flush_table = FLUSH_LOOKUP.get().unwrap();

        if count == 5 {
            return HandRank::new(flush_table[bits as usize]);
        }

        // For 6 or 7 flush cards, find the best 5
        let mut best = u16::MAX;

        // Generate all 5-card combinations from the flush cards
        let indices: Vec<u8> = (0..13).filter(|&i| bits & (1 << i) != 0).collect();

        for combo in combinations(&indices, 5) {
            let mut bits5 = 0u16;
            for &idx in &combo {
                bits5 |= 1 << idx;
            }
            let rank = flush_table[bits5 as usize];
            if rank < best {
                best = rank;
            }
        }

        HandRank::new(best)
    }

    /// Evaluate non-flush hands
    fn evaluate_non_flush_7(cards: &[Card; 7]) -> HandRank {
        let mut best = HandRank::new(u16::MAX);

        // 7C5 = 21 combinations
        const COMBOS: [[usize; 5]; 21] = [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 4, 5],
            [0, 1, 2, 4, 6],
            [0, 1, 2, 5, 6],
            [0, 1, 3, 4, 5],
            [0, 1, 3, 4, 6],
            [0, 1, 3, 5, 6],
            [0, 1, 4, 5, 6],
            [0, 2, 3, 4, 5],
            [0, 2, 3, 4, 6],
            [0, 2, 3, 5, 6],
            [0, 2, 4, 5, 6],
            [0, 3, 4, 5, 6],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 5, 6],
            [1, 2, 4, 5, 6],
            [1, 3, 4, 5, 6],
            [2, 3, 4, 5, 6],
        ];

        for combo in &COMBOS {
            let hand = [
                cards[combo[0]],
                cards[combo[1]],
                cards[combo[2]],
                cards[combo[3]],
                cards[combo[4]],
            ];
            let rank = Self::evaluate_5(&hand);
            if rank < best {
                best = rank;
            }
        }

        best
    }

    /// Evaluate a 5-card hand
    pub fn evaluate_5(cards: &[Card; 5]) -> HandRank {
        init_evaluator();

        // Get rank bits and check for flush
        let mut bits: u16 = 0;
        let mut suit_mask: u8 = 0x0F;
        let mut prime_product: u32 = 1;

        for card in cards {
            bits |= 1 << (card.rank() as u16);
            suit_mask &= card.suit_bit();
            prime_product = prime_product.wrapping_mul(card.prime() as u32);
        }

        let is_flush = suit_mask != 0;
        let is_unique = bits.count_ones() == 5;

        if is_flush {
            let flush_table = FLUSH_LOOKUP.get().unwrap();
            return HandRank::new(flush_table[bits as usize]);
        }

        if is_unique {
            // Straight or high card
            let unique5_table = UNIQUE5_LOOKUP.get().unwrap();
            return HandRank::new(unique5_table[bits as usize]);
        }

        // Pair, two pair, trips, full house, or quads
        let prime_table = PRIME_LOOKUP.get().unwrap();
        HandRank::new(*prime_table.get(&prime_product).unwrap_or(&7462))
    }
}

/// Generate the flush lookup table
fn generate_flush_table() -> Box<[u16; 8192]> {
    let mut table = Box::new([0u16; 8192]);

    // Straight flush patterns (A-high to 5-high)
    let straights: [u16; 10] = [
        0b1111100000000, // A-K-Q-J-T (Royal)
        0b0111110000000, // K-Q-J-T-9
        0b0011111000000, // Q-J-T-9-8
        0b0001111100000, // J-T-9-8-7
        0b0000111110000, // T-9-8-7-6
        0b0000011111000, // 9-8-7-6-5
        0b0000001111100, // 8-7-6-5-4
        0b0000000111110, // 7-6-5-4-3
        0b0000000011111, // 6-5-4-3-2
        0b1000000001111, // 5-4-3-2-A (Wheel)
    ];

    // Assign straight flush ranks (1-10)
    for (rank, &pattern) in straights.iter().enumerate() {
        table[pattern as usize] = (rank + 1) as u16;
    }

    // Generate all other 5-card flush hands
    let mut rank = 323u16; // After straights (1-10) and non-flush straights (1600-1609)

    // All 5-card combinations with same suit (non-straight)
    for bits in (0u16..8192).rev() {
        if bits.count_ones() == 5 {
            // Check if it's a straight
            let is_straight = straights.contains(&bits);
            if !is_straight && table[bits as usize] == 0 {
                table[bits as usize] = rank;
                rank += 1;
                if rank > 1599 {
                    rank = 1599; // Cap at flush max
                }
            }
        }
    }

    table
}

/// Generate the unique-5 lookup table (straights and high cards)
fn generate_unique5_table() -> Box<[u16; 8192]> {
    let mut table = Box::new([0u16; 8192]);

    // Straight patterns
    let straights: [u16; 10] = [
        0b1111100000000, // A-K-Q-J-T
        0b0111110000000, // K-Q-J-T-9
        0b0011111000000, // Q-J-T-9-8
        0b0001111100000, // J-T-9-8-7
        0b0000111110000, // T-9-8-7-6
        0b0000011111000, // 9-8-7-6-5
        0b0000001111100, // 8-7-6-5-4
        0b0000000111110, // 7-6-5-4-3
        0b0000000011111, // 6-5-4-3-2
        0b1000000001111, // 5-4-3-2-A (Wheel)
    ];

    // Assign straight ranks (1600-1609)
    for (i, &pattern) in straights.iter().enumerate() {
        table[pattern as usize] = 1600 + i as u16;
    }

    // High card hands (6186-7462)
    let mut rank = 6186u16;

    for bits in (0u16..8192).rev() {
        if bits.count_ones() == 5 && table[bits as usize] == 0 {
            table[bits as usize] = rank;
            rank += 1;
            if rank > 7462 {
                rank = 7462;
            }
        }
    }

    table
}

/// Generate the prime product lookup table for pair hands
fn generate_prime_table() -> std::collections::HashMap<u32, u16> {
    use std::collections::HashMap;

    let primes: [u32; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
    let mut table = HashMap::new();

    // Four of a kind (11-166)
    let mut rank = 11u16;
    for quad in (0..13).rev() {
        for kicker in (0..13).rev() {
            if kicker != quad {
                let product = primes[quad].pow(4) * primes[kicker];
                table.insert(product, rank);
                rank += 1;
            }
        }
    }

    // Full house (167-322)
    rank = 167;
    for trips in (0..13).rev() {
        for pair in (0..13).rev() {
            if pair != trips {
                let product = primes[trips].pow(3) * primes[pair].pow(2);
                table.insert(product, rank);
                rank += 1;
            }
        }
    }

    // Three of a kind (1610-2467)
    rank = 1610;
    for trips in (0..13).rev() {
        for k1 in (0..13).rev() {
            if k1 == trips {
                continue;
            }
            for k2 in (0..k1).rev() {
                if k2 == trips {
                    continue;
                }
                let product = primes[trips].pow(3) * primes[k1] * primes[k2];
                table.insert(product, rank);
                rank += 1;
            }
        }
    }

    // Two pair (2468-3325)
    rank = 2468;
    for p1 in (0..13).rev() {
        for p2 in (0..p1).rev() {
            for kicker in (0..13).rev() {
                if kicker != p1 && kicker != p2 {
                    let product = primes[p1].pow(2) * primes[p2].pow(2) * primes[kicker];
                    table.insert(product, rank);
                    rank += 1;
                }
            }
        }
    }

    // One pair (3326-6185)
    rank = 3326;
    for pair in (0..13).rev() {
        for k1 in (0..13).rev() {
            if k1 == pair {
                continue;
            }
            for k2 in (0..k1).rev() {
                if k2 == pair {
                    continue;
                }
                for k3 in (0..k2).rev() {
                    if k3 == pair {
                        continue;
                    }
                    let product = primes[pair].pow(2) * primes[k1] * primes[k2] * primes[k3];
                    table.insert(product, rank);
                    rank += 1;
                }
            }
        }
    }

    table
}

/// Helper function to generate combinations
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();

    for (i, item) in items.iter().enumerate() {
        for mut combo in combinations(&items[i + 1..], k - 1) {
            combo.insert(0, item.clone());
            result.push(combo);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(rank: u8, suit: u8) -> Card {
        Card::new(Rank::ALL[rank as usize], Suit::from_index(suit))
    }

    #[test]
    fn test_royal_flush() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(11, 0), // Kc
            make_card(10, 0), // Qc
            make_card(9, 0),  // Jc
            make_card(8, 0),  // Tc
            make_card(0, 1),  // 2d
            make_card(1, 2),  // 3h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.value(), 1);
        assert_eq!(rank.category(), HandCategory::StraightFlush);
    }

    #[test]
    fn test_wheel_straight_flush() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(0, 0),  // 2c
            make_card(1, 0),  // 3c
            make_card(2, 0),  // 4c
            make_card(3, 0),  // 5c
            make_card(6, 1),  // 8d
            make_card(7, 2),  // 9h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.value(), 10); // Wheel straight flush
        assert_eq!(rank.category(), HandCategory::StraightFlush);
    }

    #[test]
    fn test_four_of_a_kind() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(12, 3), // As
            make_card(11, 0), // Kc
            make_card(0, 1),  // 2d
            make_card(1, 2),  // 3h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::FourOfAKind);
    }

    #[test]
    fn test_full_house() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(11, 0), // Kc
            make_card(11, 1), // Kd
            make_card(0, 2),  // 2h
            make_card(1, 3),  // 3s
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::FullHouse);
    }

    #[test]
    fn test_flush() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(10, 0), // Qc
            make_card(8, 0),  // Tc
            make_card(6, 0),  // 8c
            make_card(4, 0),  // 6c
            make_card(2, 1),  // 4d
            make_card(0, 2),  // 2h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::Flush);
    }

    #[test]
    fn test_straight() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(11, 1), // Kd
            make_card(10, 2), // Qh
            make_card(9, 3),  // Js
            make_card(8, 0),  // Tc
            make_card(2, 1),  // 4d
            make_card(0, 2),  // 2h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::Straight);
    }

    #[test]
    fn test_three_of_a_kind() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(12, 2), // Ah
            make_card(11, 0), // Kc
            make_card(9, 1),  // Jd
            make_card(2, 2),  // 4h
            make_card(0, 3),  // 2s
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::ThreeOfAKind);
    }

    #[test]
    fn test_two_pair() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(11, 0), // Kc
            make_card(11, 1), // Kd
            make_card(9, 2),  // Jh
            make_card(2, 2),  // 4h
            make_card(0, 3),  // 2s
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::TwoPair);
    }

    #[test]
    fn test_one_pair() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(12, 1), // Ad
            make_card(11, 0), // Kc
            make_card(9, 1),  // Jd
            make_card(7, 2),  // 9h
            make_card(2, 2),  // 4h
            make_card(0, 3),  // 2s
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::OnePair);
    }

    #[test]
    fn test_high_card() {
        let cards = [
            make_card(12, 0), // Ac
            make_card(10, 1), // Qd
            make_card(8, 2),  // Th
            make_card(6, 3),  // 8s
            make_card(4, 0),  // 6c
            make_card(2, 1),  // 4d
            make_card(0, 2),  // 2h
        ];

        let rank = Evaluator::evaluate_7(&cards);
        assert_eq!(rank.category(), HandCategory::HighCard);
    }

    #[test]
    fn test_hand_comparison() {
        // Full house beats flush
        let full_house = [
            make_card(10, 0),
            make_card(10, 1),
            make_card(10, 2),
            make_card(5, 0),
            make_card(5, 1),
            make_card(0, 2),
            make_card(1, 3),
        ];
        let flush = [
            make_card(12, 0),
            make_card(10, 0),
            make_card(8, 0),
            make_card(6, 0),
            make_card(4, 0),
            make_card(2, 1),
            make_card(0, 2),
        ];

        let fh_rank = Evaluator::evaluate_7(&full_house);
        let fl_rank = Evaluator::evaluate_7(&flush);

        assert!(fh_rank < fl_rank); // Lower rank = stronger
    }
}
