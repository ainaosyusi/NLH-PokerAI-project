"""Card and action abstraction for poker AI.

This module provides:
1. Hand strength calculation (equity estimation)
2. Card bucketing (clustering similar hands)
3. Action abstraction (discretizing bet sizes)
4. Feature extraction for neural networks
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from enum import IntEnum

import poker_engine


class Street(IntEnum):
    """Betting round enumeration."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class HandFeatures:
    """Features extracted from a poker hand situation."""
    # Hand strength features
    hand_strength: float  # Current equity (0-1)
    hand_potential: float  # Probability of improving

    # Board texture features
    board_wetness: float  # How coordinated the board is
    board_high_card: int  # Highest card rank on board (0-12)
    flush_possible: bool  # 3+ of same suit on board
    straight_possible: bool  # Connected cards on board
    paired_board: bool  # Pair on board

    # Position features
    position: int  # 0 = early, 1 = middle, 2 = late, 3 = blinds
    players_to_act: int  # Players yet to act

    # Pot odds features
    pot_odds: float  # Amount to call / pot size
    stack_to_pot: float  # Effective stack / pot size

    # Action history (simplified)
    raises_this_street: int
    is_facing_bet: bool


class HandStrengthCalculator:
    """Calculate hand strength through Monte Carlo simulation."""

    def __init__(self, num_simulations: int = 1000):
        """Initialize with number of Monte Carlo simulations."""
        self.num_simulations = num_simulations

    def calculate_equity(
        self,
        hole_cards: List[Tuple[int, int]],  # List of (rank, suit) for our cards
        board: List[Tuple[int, int]],  # Community cards
        num_opponents: int = 1,
    ) -> float:
        """Calculate hand equity through Monte Carlo simulation.

        Args:
            hole_cards: Our hole cards as (rank, suit) tuples
            board: Community cards as (rank, suit) tuples
            num_opponents: Number of opponents

        Returns:
            Equity as a float between 0 and 1
        """
        if len(board) == 5:
            # River - exact calculation possible but we use sampling
            return self._monte_carlo_equity(hole_cards, board, num_opponents)
        else:
            # Pre-river - use Monte Carlo
            return self._monte_carlo_equity(hole_cards, board, num_opponents)

    def _monte_carlo_equity(
        self,
        hole_cards: List[Tuple[int, int]],
        board: List[Tuple[int, int]],
        num_opponents: int,
    ) -> float:
        """Monte Carlo equity calculation."""
        wins = 0
        ties = 0

        # Convert to card indices (0-51)
        our_cards = [self._to_index(r, s) for r, s in hole_cards]
        board_cards = [self._to_index(r, s) for r, s in board]
        used_cards = set(our_cards + board_cards)

        # Available cards for dealing
        available = [i for i in range(52) if i not in used_cards]

        for _ in range(self.num_simulations):
            # Shuffle and deal
            np.random.shuffle(available)
            idx = 0

            # Complete the board
            sim_board = board_cards.copy()
            cards_needed = 5 - len(board_cards)
            for _ in range(cards_needed):
                sim_board.append(available[idx])
                idx += 1

            # Deal opponent hands
            opponent_hands = []
            for _ in range(num_opponents):
                opp_hand = [available[idx], available[idx + 1]]
                opponent_hands.append(opp_hand)
                idx += 2

            # Evaluate hands
            our_7 = our_cards + sim_board
            our_rank = self._evaluate_7(our_7)

            best_opp_rank = 10000
            for opp_hand in opponent_hands:
                opp_7 = opp_hand + sim_board
                opp_rank = self._evaluate_7(opp_7)
                best_opp_rank = min(best_opp_rank, opp_rank)

            if our_rank < best_opp_rank:
                wins += 1
            elif our_rank == best_opp_rank:
                ties += 1

        return (wins + ties * 0.5) / self.num_simulations

    def _to_index(self, rank: int, suit: int) -> int:
        """Convert (rank, suit) to card index (0-51)."""
        return suit * 13 + rank

    def _evaluate_7(self, card_indices: List[int]) -> int:
        """Evaluate 7 cards and return hand rank (lower is better)."""
        # Use the Rust evaluator through Python bindings
        cards = [poker_engine.make_card(idx % 13, idx // 13) for idx in card_indices]
        rank, _ = poker_engine.evaluate_hand(cards[:2], cards[2:7])
        return rank


class PreflopBucketing:
    """Preflop hand bucketing based on hand categories."""

    # Chen formula inspired buckets
    BUCKET_NAMES = [
        "Premium",      # AA, KK, QQ, AKs
        "Strong",       # JJ, TT, AQs, AKo, AJs
        "Good",         # 99, 88, ATs, KQs, AQo
        "Playable",     # 77, 66, KJs, QJs, AJo, KQo
        "Speculative",  # 55, 44, 33, 22, suited connectors, small pairs
        "Marginal",     # Weak hands that can sometimes be played
        "Fold",         # Trash
    ]

    def __init__(self):
        """Initialize preflop buckets."""
        self._build_lookup_table()

    def _build_lookup_table(self):
        """Build lookup table for all 169 starting hands."""
        self.bucket_table = {}

        # Premium hands (bucket 0)
        premium = ["AA", "KK", "QQ", "AKs"]
        # Strong hands (bucket 1)
        strong = ["JJ", "TT", "AQs", "AKo", "AJs", "KQs"]
        # Good hands (bucket 2)
        good = ["99", "88", "ATs", "KJs", "QJs", "AQo", "KQo"]
        # Playable hands (bucket 3)
        playable = ["77", "66", "A9s", "A8s", "KTs", "QTs", "JTs", "AJo", "KJo"]
        # Speculative hands (bucket 4)
        speculative = ["55", "44", "33", "22", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
                       "K9s", "Q9s", "J9s", "T9s", "98s", "87s", "76s", "65s", "54s"]

        for hand in premium:
            self.bucket_table[hand] = 0
        for hand in strong:
            self.bucket_table[hand] = 1
        for hand in good:
            self.bucket_table[hand] = 2
        for hand in playable:
            self.bucket_table[hand] = 3
        for hand in speculative:
            self.bucket_table[hand] = 4

    def get_bucket(self, rank1: int, suit1: int, rank2: int, suit2: int) -> int:
        """Get bucket index for a preflop hand.

        Args:
            rank1, suit1: First card (rank 0-12, suit 0-3)
            rank2, suit2: Second card

        Returns:
            Bucket index (0-6)
        """
        # Normalize hand (higher card first)
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1

        suited = suit1 == suit2

        # Build hand string
        rank_chars = "23456789TJQKA"
        hand_str = rank_chars[rank1] + rank_chars[rank2]
        if rank1 != rank2:
            hand_str += "s" if suited else "o"

        return self.bucket_table.get(hand_str, 5 if suited else 6)


class PostflopBucketing:
    """Postflop hand bucketing based on hand strength percentiles."""

    NUM_BUCKETS = 10

    def __init__(self, equity_calculator: Optional[HandStrengthCalculator] = None):
        """Initialize postflop bucketing."""
        self.equity_calc = equity_calculator or HandStrengthCalculator(num_simulations=500)

    def get_bucket(
        self,
        hole_cards: List[Tuple[int, int]],
        board: List[Tuple[int, int]],
        num_opponents: int = 1,
    ) -> int:
        """Get bucket index based on equity.

        Args:
            hole_cards: Our hole cards
            board: Community cards
            num_opponents: Number of opponents

        Returns:
            Bucket index (0-9, higher is stronger)
        """
        equity = self.equity_calc.calculate_equity(hole_cards, board, num_opponents)

        # Map equity to bucket (0-9)
        # Non-linear mapping to give more granularity to strong hands
        if equity < 0.2:
            return 0
        elif equity < 0.35:
            return 1
        elif equity < 0.45:
            return 2
        elif equity < 0.50:
            return 3
        elif equity < 0.55:
            return 4
        elif equity < 0.60:
            return 5
        elif equity < 0.70:
            return 6
        elif equity < 0.80:
            return 7
        elif equity < 0.90:
            return 8
        else:
            return 9


class ActionAbstraction:
    """Action abstraction for bet sizing."""

    # Discrete bet sizes as fractions of pot
    BET_SIZES = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]  # All-in handled separately

    def __init__(self, include_all_in: bool = True):
        """Initialize action abstraction.

        Args:
            include_all_in: Whether to include all-in as a separate action
        """
        self.include_all_in = include_all_in

    def get_abstract_actions(
        self,
        pot: int,
        to_call: int,
        stack: int,
        min_raise: int,
    ) -> List[Tuple[str, int]]:
        """Get list of abstract actions with bet amounts.

        Args:
            pot: Current pot size
            to_call: Amount to call
            stack: Our remaining stack
            min_raise: Minimum raise amount

        Returns:
            List of (action_name, amount) tuples
        """
        actions = []

        # Fold is always available
        actions.append(("fold", 0))

        # Check/Call
        if to_call == 0:
            actions.append(("check", 0))
        elif to_call <= stack:
            actions.append(("call", to_call))

        # Bet/Raise sizes
        pot_after_call = pot + to_call
        for frac in self.BET_SIZES:
            bet_amount = int(pot_after_call * frac)

            if to_call == 0:
                # Betting
                if bet_amount >= min_raise and bet_amount <= stack:
                    actions.append(("bet", bet_amount))
            else:
                # Raising
                raise_to = to_call + bet_amount
                if raise_to >= min_raise and raise_to <= stack:
                    actions.append(("raise", raise_to))

        # All-in
        if self.include_all_in and stack > 0:
            actions.append(("all_in", stack))

        return actions

    def map_to_abstract_action(
        self,
        action_type: str,
        amount: int,
        pot: int,
        to_call: int,
    ) -> int:
        """Map a concrete action to the nearest abstract action index.

        Args:
            action_type: "fold", "check", "call", "bet", "raise", "all_in"
            amount: Bet/raise amount
            pot: Current pot
            to_call: Amount to call

        Returns:
            Abstract action index
        """
        if action_type == "fold":
            return 0
        elif action_type in ("check", "call"):
            return 1
        elif action_type == "all_in":
            return len(self.BET_SIZES) + 2
        else:
            # Find nearest bet size
            pot_after_call = pot + to_call
            if pot_after_call == 0:
                return 2  # Default to first bet size

            actual_frac = amount / pot_after_call

            best_idx = 0
            best_diff = float('inf')
            for i, frac in enumerate(self.BET_SIZES):
                diff = abs(actual_frac - frac)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            return 2 + best_idx


class FeatureExtractor:
    """Extract features for neural network input."""

    def __init__(
        self,
        num_players: int = 6,
        preflop_buckets: int = 7,
        postflop_buckets: int = 10,
    ):
        """Initialize feature extractor.

        Args:
            num_players: Maximum number of players
            preflop_buckets: Number of preflop hand buckets
            postflop_buckets: Number of postflop hand buckets
        """
        self.num_players = num_players
        self.preflop_buckets = preflop_buckets
        self.postflop_buckets = postflop_buckets

        self.preflop_bucketing = PreflopBucketing()
        self.postflop_bucketing = PostflopBucketing()

        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = 0
        dim += self.preflop_buckets  # Preflop hand bucket (one-hot)
        dim += self.postflop_buckets  # Postflop bucket (one-hot)
        dim += 4  # Street (one-hot)
        dim += self.num_players  # Position (one-hot)
        dim += self.num_players  # Active players (binary)
        dim += 5  # Pot odds, stack/pot, raises, facing bet, etc.
        dim += 5  # Board texture features
        return dim

    def extract_features(
        self,
        hole_cards: List[Tuple[int, int]],
        board: List[Tuple[int, int]],
        street: int,
        position: int,
        pot: int,
        to_call: int,
        stack: int,
        active_players: List[bool],
        raises_this_street: int = 0,
    ) -> np.ndarray:
        """Extract feature vector from game state.

        Args:
            hole_cards: Our hole cards as (rank, suit) tuples
            board: Community cards
            street: Current street (0-3)
            position: Our position (0 to num_players-1)
            pot: Current pot size
            to_call: Amount to call
            stack: Our stack size
            active_players: List of active player flags
            raises_this_street: Number of raises this street

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Preflop bucket (one-hot)
        preflop_bucket = self.preflop_bucketing.get_bucket(
            hole_cards[0][0], hole_cards[0][1],
            hole_cards[1][0], hole_cards[1][1]
        )
        preflop_onehot = np.zeros(self.preflop_buckets)
        preflop_onehot[preflop_bucket] = 1.0
        features.append(preflop_onehot)

        # Postflop bucket (one-hot) - only if we have board cards
        postflop_onehot = np.zeros(self.postflop_buckets)
        if len(board) >= 3:
            postflop_bucket = self.postflop_bucketing.get_bucket(
                hole_cards, board, num_opponents=sum(active_players) - 1
            )
            postflop_onehot[postflop_bucket] = 1.0
        features.append(postflop_onehot)

        # Street (one-hot)
        street_onehot = np.zeros(4)
        street_onehot[min(street, 3)] = 1.0
        features.append(street_onehot)

        # Position (one-hot)
        position_onehot = np.zeros(self.num_players)
        position_onehot[position] = 1.0
        features.append(position_onehot)

        # Active players (binary)
        active_array = np.array(active_players[:self.num_players], dtype=np.float32)
        if len(active_array) < self.num_players:
            active_array = np.pad(active_array, (0, self.num_players - len(active_array)))
        features.append(active_array)

        # Numeric features (normalized)
        pot_odds = to_call / max(pot + to_call, 1)
        stack_pot_ratio = min(stack / max(pot, 1), 10.0) / 10.0
        raises_normalized = min(raises_this_street, 4) / 4.0
        facing_bet = 1.0 if to_call > 0 else 0.0
        num_active = sum(active_players) / self.num_players

        numeric_features = np.array([
            pot_odds,
            stack_pot_ratio,
            raises_normalized,
            facing_bet,
            num_active,
        ], dtype=np.float32)
        features.append(numeric_features)

        # Board texture features
        board_features = self._extract_board_features(board)
        features.append(board_features)

        return np.concatenate(features).astype(np.float32)

    def _extract_board_features(self, board: List[Tuple[int, int]]) -> np.ndarray:
        """Extract board texture features."""
        features = np.zeros(5, dtype=np.float32)

        if not board:
            return features

        ranks = [r for r, s in board]
        suits = [s for r, s in board]

        # High card (normalized)
        features[0] = max(ranks) / 12.0

        # Flush possible (3+ same suit)
        suit_counts = [suits.count(s) for s in range(4)]
        features[1] = 1.0 if max(suit_counts) >= 3 else 0.0

        # Straight possible (check for connected cards)
        sorted_ranks = sorted(set(ranks))
        max_connected = 1
        current_connected = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] - sorted_ranks[i-1] <= 2:
                current_connected += 1
                max_connected = max(max_connected, current_connected)
            else:
                current_connected = 1
        features[2] = 1.0 if max_connected >= 3 else 0.0

        # Paired board
        features[3] = 1.0 if len(ranks) != len(set(ranks)) else 0.0

        # Board wetness (combination of above)
        features[4] = (features[1] + features[2] + features[3]) / 3.0

        return features


# Convenience functions
def create_default_extractor(num_players: int = 6) -> FeatureExtractor:
    """Create a feature extractor with default settings."""
    return FeatureExtractor(num_players=num_players)


def create_action_abstraction() -> ActionAbstraction:
    """Create action abstraction with default bet sizes."""
    return ActionAbstraction()
