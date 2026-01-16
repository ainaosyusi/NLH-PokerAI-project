//! Card and Deck representation using bitboards for ultra-fast operations.
//!
//! Card encoding (32-bit):
//! ```text
//! +--------+--------+--------+--------+
//! |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
//! +--------+--------+--------+--------+
//! bits 16-28: rank bitmask (2=bit16, A=bit28)
//! bits 12-15: suit flag (c=8, d=4, h=2, s=1)
//! bits 8-11:  rank index (0-12)
//! bits 0-5:   prime number for hand evaluation
//! ```

use crate::rng::FastRng;
use std::fmt;

/// Prime numbers for each rank (2-A), used in hand evaluation
const PRIMES: [u8; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// Suit representation with bitflag values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Suit {
    Clubs = 0b1000,    // bit 15
    Diamonds = 0b0100, // bit 14
    Hearts = 0b0010,   // bit 13
    Spades = 0b0001,   // bit 12
}

impl Suit {
    pub const ALL: [Suit; 4] = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];

    #[inline]
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Suit::Clubs,
            1 => Suit::Diamonds,
            2 => Suit::Hearts,
            3 => Suit::Spades,
            _ => panic!("Invalid suit index: {}", index),
        }
    }

    #[inline]
    pub fn to_index(self) -> u8 {
        match self {
            Suit::Clubs => 0,
            Suit::Diamonds => 1,
            Suit::Hearts => 2,
            Suit::Spades => 3,
        }
    }

    #[inline]
    pub fn to_char(self) -> char {
        match self {
            Suit::Clubs => 'c',
            Suit::Diamonds => 'd',
            Suit::Hearts => 'h',
            Suit::Spades => 's',
        }
    }
}

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Rank representation (2-A = 0-12)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Rank {
    Two = 0,
    Three = 1,
    Four = 2,
    Five = 3,
    Six = 4,
    Seven = 5,
    Eight = 6,
    Nine = 7,
    Ten = 8,
    Jack = 9,
    Queen = 10,
    King = 11,
    Ace = 12,
}

impl Rank {
    pub const ALL: [Rank; 13] = [
        Rank::Two,
        Rank::Three,
        Rank::Four,
        Rank::Five,
        Rank::Six,
        Rank::Seven,
        Rank::Eight,
        Rank::Nine,
        Rank::Ten,
        Rank::Jack,
        Rank::Queen,
        Rank::King,
        Rank::Ace,
    ];

    #[inline]
    pub fn from_index(index: u8) -> Self {
        Self::ALL[index as usize]
    }

    #[inline]
    pub fn prime(self) -> u8 {
        PRIMES[self as usize]
    }

    #[inline]
    pub fn bit_mask(self) -> u32 {
        1u32 << (16 + self as u32)
    }

    #[inline]
    pub fn to_char(self) -> char {
        match self {
            Rank::Two => '2',
            Rank::Three => '3',
            Rank::Four => '4',
            Rank::Five => '5',
            Rank::Six => '6',
            Rank::Seven => '7',
            Rank::Eight => '8',
            Rank::Nine => '9',
            Rank::Ten => 'T',
            Rank::Jack => 'J',
            Rank::Queen => 'Q',
            Rank::King => 'K',
            Rank::Ace => 'A',
        }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A single playing card encoded as a 32-bit integer
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Card(u32);

impl Card {
    /// Create a new card from rank and suit
    #[inline]
    pub const fn new(rank: Rank, suit: Suit) -> Self {
        let r = rank as u32;
        let s = suit as u32;
        let p = PRIMES[rank as usize] as u32;
        let bit_rank = 1u32 << (16 + r);

        // |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
        let value = bit_rank | (s << 12) | (r << 8) | p;
        Card(value)
    }

    /// Create a card from index (0-51)
    /// Index mapping: 0-12 = 2c-Ac, 13-25 = 2d-Ad, 26-38 = 2h-Ah, 39-51 = 2s-As
    #[inline]
    pub fn from_index(index: u8) -> Self {
        debug_assert!(index < 52, "Card index must be 0-51");
        let rank = Rank::from_index(index % 13);
        let suit = Suit::from_index(index / 13);
        Self::new(rank, suit)
    }

    /// Convert card to index (0-51)
    #[inline]
    pub fn to_index(self) -> u8 {
        let rank = self.rank() as u8;
        let suit = self.suit().to_index();
        suit * 13 + rank
    }

    /// Get the rank of this card
    #[inline]
    pub fn rank(self) -> Rank {
        Rank::from_index(((self.0 >> 8) & 0x0F) as u8)
    }

    /// Get the suit of this card
    #[inline]
    pub fn suit(self) -> Suit {
        match (self.0 >> 12) & 0x0F {
            0b1000 => Suit::Clubs,
            0b0100 => Suit::Diamonds,
            0b0010 => Suit::Hearts,
            0b0001 => Suit::Spades,
            _ => unreachable!(),
        }
    }

    /// Get the prime number for this card's rank
    #[inline]
    pub fn prime(self) -> u8 {
        (self.0 & 0x3F) as u8
    }

    /// Get the bit mask for this card's rank
    #[inline]
    pub fn bit_rank(self) -> u32 {
        self.0 >> 16
    }

    /// Get the suit bit (for flush detection)
    #[inline]
    pub fn suit_bit(self) -> u8 {
        ((self.0 >> 12) & 0x0F) as u8
    }

    /// Get the raw 32-bit value
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Parse a card from string like "As", "Kh", "2c"
    pub fn from_str(s: &str) -> Option<Self> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 2 {
            return None;
        }

        let rank = match chars[0] {
            '2' => Rank::Two,
            '3' => Rank::Three,
            '4' => Rank::Four,
            '5' => Rank::Five,
            '6' => Rank::Six,
            '7' => Rank::Seven,
            '8' => Rank::Eight,
            '9' => Rank::Nine,
            'T' | 't' => Rank::Ten,
            'J' | 'j' => Rank::Jack,
            'Q' | 'q' => Rank::Queen,
            'K' | 'k' => Rank::King,
            'A' | 'a' => Rank::Ace,
            _ => return None,
        };

        let suit = match chars[1] {
            'c' | 'C' => Suit::Clubs,
            'd' | 'D' => Suit::Diamonds,
            'h' | 'H' => Suit::Hearts,
            's' | 'S' => Suit::Spades,
            _ => return None,
        };

        Some(Self::new(rank, suit))
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit())
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit())
    }
}

/// A deck of 52 cards represented as a 64-bit bitboard
#[derive(Clone, Copy)]
pub struct Deck {
    /// Each bit represents a card (bit i = Card::from_index(i))
    cards: u64,
    /// Number of remaining cards
    count: u8,
}

impl Deck {
    /// Create a new full deck (52 cards)
    #[inline]
    pub fn new() -> Self {
        Self {
            cards: (1u64 << 52) - 1, // Lower 52 bits all set
            count: 52,
        }
    }

    /// Create an empty deck
    #[inline]
    pub fn empty() -> Self {
        Self { cards: 0, count: 0 }
    }

    /// Draw a random card from the deck
    #[inline]
    pub fn draw(&mut self, rng: &mut FastRng) -> Option<Card> {
        if self.count == 0 {
            return None;
        }

        // Pick a random position among remaining cards
        let n = rng.gen_range(0, self.count as u32);
        let index = self.nth_set_bit(n);

        // Remove the card
        self.cards &= !(1u64 << index);
        self.count -= 1;

        Some(Card::from_index(index as u8))
    }

    /// Remove a specific card from the deck
    #[inline]
    pub fn remove(&mut self, card: Card) -> bool {
        let index = card.to_index() as u64;
        let mask = 1u64 << index;
        if self.cards & mask != 0 {
            self.cards &= !mask;
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Check if a card is in the deck
    #[inline]
    pub fn contains(&self, card: Card) -> bool {
        let index = card.to_index() as u64;
        self.cards & (1u64 << index) != 0
    }

    /// Get the number of remaining cards
    #[inline]
    pub fn remaining(&self) -> u8 {
        self.count
    }

    /// Deal n cards from the deck
    pub fn deal(&mut self, n: usize, rng: &mut FastRng) -> Vec<Card> {
        (0..n).filter_map(|_| self.draw(rng)).collect()
    }

    /// Find the position of the nth set bit
    #[inline]
    fn nth_set_bit(&self, n: u32) -> u32 {
        let mut remaining = n;
        let mut bits = self.cards;

        for i in 0..64 {
            if bits & 1 != 0 {
                if remaining == 0 {
                    return i;
                }
                remaining -= 1;
            }
            bits >>= 1;
        }
        unreachable!()
    }
}

impl Default for Deck {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of cards (for hole cards or community cards)
#[derive(Clone, Copy, Default)]
pub struct Hand {
    /// Bitmask of cards in hand
    cards: u64,
    /// Product of prime numbers (for hand evaluation)
    prime_product: u32,
    /// Number of cards
    count: u8,
}

impl Hand {
    /// Create a new empty hand
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a card to the hand
    #[inline]
    pub fn add(&mut self, card: Card) {
        let index = card.to_index() as u64;
        self.cards |= 1u64 << index;
        self.prime_product = self.prime_product.saturating_mul(card.prime() as u32);
        self.count += 1;
    }

    /// Check if hand contains a card
    #[inline]
    pub fn contains(&self, card: Card) -> bool {
        let index = card.to_index() as u64;
        self.cards & (1u64 << index) != 0
    }

    /// Get the number of cards in hand
    #[inline]
    pub fn count(&self) -> u8 {
        self.count
    }

    /// Get the bitmask of cards
    #[inline]
    pub fn cards_mask(&self) -> u64 {
        self.cards
    }

    /// Get the prime product (for hand evaluation)
    #[inline]
    pub fn prime_product(&self) -> u32 {
        self.prime_product
    }

    /// Clear the hand
    #[inline]
    pub fn clear(&mut self) {
        self.cards = 0;
        self.prime_product = 1;
        self.count = 0;
    }

    /// Iterate over cards in hand
    pub fn iter(&self) -> impl Iterator<Item = Card> + '_ {
        (0..52).filter_map(move |i| {
            if self.cards & (1u64 << i) != 0 {
                Some(Card::from_index(i))
            } else {
                None
            }
        })
    }

    /// Convert to a vector of cards
    pub fn to_vec(&self) -> Vec<Card> {
        self.iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_creation() {
        let card = Card::new(Rank::Ace, Suit::Spades);
        assert_eq!(card.rank(), Rank::Ace);
        assert_eq!(card.suit(), Suit::Spades);
        assert_eq!(card.prime(), 41);
    }

    #[test]
    fn test_card_index_roundtrip() {
        for i in 0..52 {
            let card = Card::from_index(i);
            assert_eq!(card.to_index(), i);
        }
    }

    #[test]
    fn test_card_display() {
        assert_eq!(format!("{}", Card::new(Rank::Ace, Suit::Spades)), "As");
        assert_eq!(format!("{}", Card::new(Rank::Ten, Suit::Hearts)), "Th");
        assert_eq!(format!("{}", Card::new(Rank::Two, Suit::Clubs)), "2c");
    }

    #[test]
    fn test_card_from_str() {
        assert_eq!(Card::from_str("As"), Some(Card::new(Rank::Ace, Suit::Spades)));
        assert_eq!(Card::from_str("Th"), Some(Card::new(Rank::Ten, Suit::Hearts)));
        assert_eq!(Card::from_str("2c"), Some(Card::new(Rank::Two, Suit::Clubs)));
        assert_eq!(Card::from_str("invalid"), None);
    }

    #[test]
    fn test_deck_new() {
        let deck = Deck::new();
        assert_eq!(deck.remaining(), 52);
    }

    #[test]
    fn test_deck_draw_all() {
        let mut deck = Deck::new();
        let mut rng = FastRng::new(12345);
        let mut drawn = std::collections::HashSet::new();

        for _ in 0..52 {
            let card = deck.draw(&mut rng).unwrap();
            assert!(drawn.insert(card.to_index()));
        }

        assert_eq!(deck.remaining(), 0);
        assert!(deck.draw(&mut rng).is_none());
    }

    #[test]
    fn test_deck_remove() {
        let mut deck = Deck::new();
        let card = Card::new(Rank::Ace, Suit::Spades);

        assert!(deck.contains(card));
        assert!(deck.remove(card));
        assert!(!deck.contains(card));
        assert!(!deck.remove(card));
        assert_eq!(deck.remaining(), 51);
    }

    #[test]
    fn test_hand() {
        let mut hand = Hand::new();
        let c1 = Card::new(Rank::Ace, Suit::Spades);
        let c2 = Card::new(Rank::King, Suit::Spades);

        hand.add(c1);
        hand.add(c2);

        assert_eq!(hand.count(), 2);
        assert!(hand.contains(c1));
        assert!(hand.contains(c2));
        assert!(!hand.contains(Card::new(Rank::Queen, Suit::Spades)));
    }
}
