"""Multi-player feature extraction for poker AI.

This module provides:
1. Enhanced feature extraction for 2-9 player games
2. Relative position encoding
3. Per-opponent feature vectors
4. Opponent modeling statistics tracking
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from .abstraction import (
        FeatureExtractor,
        PreflopBucketing,
        PostflopBucketing,
        ActionAbstraction,
    )
except ImportError:
    from abstraction import (
        FeatureExtractor,
        PreflopBucketing,
        PostflopBucketing,
        ActionAbstraction,
    )


@dataclass
class OpponentStats:
    """Statistics for modeling an opponent's play style."""
    hands_played: int = 0
    vpip_hands: int = 0  # Voluntarily put $ in pot (excluding blinds)
    pfr_hands: int = 0   # Preflop raise
    threebets: int = 0   # 3-bet preflop
    cbet_opportunities: int = 0
    cbets_made: int = 0  # Continuation bets
    folds_to_cbet: int = 0
    cbet_faced: int = 0
    aggression_actions: int = 0  # Bets + raises
    passive_actions: int = 0     # Calls + checks
    showdown_hands: int = 0
    showdown_wins: int = 0
    total_bets: List[float] = field(default_factory=list)

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot percentage."""
        return self.vpip_hands / max(self.hands_played, 1)

    @property
    def pfr(self) -> float:
        """Preflop Raise percentage."""
        return self.pfr_hands / max(self.hands_played, 1)

    @property
    def threebet_pct(self) -> float:
        """3-bet percentage."""
        return self.threebets / max(self.pfr_hands, 1)

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        return self.cbets_made / max(self.cbet_opportunities, 1)

    @property
    def fold_to_cbet(self) -> float:
        """Fold to c-bet percentage."""
        return self.folds_to_cbet / max(self.cbet_faced, 1)

    @property
    def aggression_factor(self) -> float:
        """Aggression factor (bets+raises / calls)."""
        return self.aggression_actions / max(self.passive_actions, 1)

    @property
    def wtsd(self) -> float:
        """Went to showdown percentage."""
        return self.showdown_hands / max(self.hands_played, 1)

    @property
    def won_at_showdown(self) -> float:
        """Won at showdown percentage."""
        return self.showdown_wins / max(self.showdown_hands, 1)

    @property
    def avg_bet_size(self) -> float:
        """Average bet size as fraction of pot."""
        return np.mean(self.total_bets) if self.total_bets else 0.5

    def to_feature_vector(self) -> np.ndarray:
        """Convert stats to feature vector."""
        return np.array([
            self.vpip,
            self.pfr,
            self.threebet_pct,
            self.cbet_pct,
            self.fold_to_cbet,
            min(self.aggression_factor, 5.0) / 5.0,  # Normalized
            self.wtsd,
            self.won_at_showdown,
            self.avg_bet_size,
            min(self.hands_played, 100) / 100.0,  # Confidence indicator
        ], dtype=np.float32)


@dataclass
class PlayerProfile:
    """Profile for categorizing player types."""
    MANIAC = "maniac"      # High VPIP, high aggression
    LAG = "lag"            # Loose-aggressive
    TAG = "tag"            # Tight-aggressive
    ROCK = "rock"          # Very tight, passive
    FISH = "fish"          # Loose, passive
    UNKNOWN = "unknown"    # Not enough data

    @staticmethod
    def classify(stats: OpponentStats) -> str:
        """Classify player type from statistics."""
        if stats.hands_played < 20:
            return PlayerProfile.UNKNOWN

        vpip = stats.vpip
        pfr = stats.pfr
        af = stats.aggression_factor

        # Classification based on standard HUD categories
        if vpip > 0.40 and af > 2.0:
            return PlayerProfile.MANIAC
        elif vpip > 0.25 and af > 1.5:
            return PlayerProfile.LAG
        elif vpip < 0.20 and af > 1.5:
            return PlayerProfile.TAG
        elif vpip < 0.15 and af < 1.0:
            return PlayerProfile.ROCK
        elif vpip > 0.35 and af < 1.0:
            return PlayerProfile.FISH
        else:
            # Default to closest match
            if af > 1.2:
                return PlayerProfile.TAG if vpip < 0.25 else PlayerProfile.LAG
            else:
                return PlayerProfile.ROCK if vpip < 0.25 else PlayerProfile.FISH


class OpponentTracker:
    """Track opponent statistics across multiple hands."""

    def __init__(self, max_players: int = 9):
        """Initialize opponent tracker.

        Args:
            max_players: Maximum number of players to track
        """
        self.max_players = max_players
        self.stats: Dict[int, OpponentStats] = defaultdict(OpponentStats)
        self._current_hand_data: Dict[str, any] = {}

    def reset_hand(self):
        """Reset per-hand tracking data."""
        self._current_hand_data = {
            "preflop_raiser": None,
            "preflop_actions": defaultdict(list),
            "postflop_aggressor": None,
            "vpip_players": set(),
        }

    def record_action(
        self,
        player_id: int,
        action_type: str,
        amount: int,
        pot_size: int,
        street: int,
        is_facing_raise: bool = False,
    ):
        """Record an action for opponent modeling.

        Args:
            player_id: Player who took the action
            action_type: Type of action (fold, check, call, bet, raise, all_in)
            amount: Bet/raise amount
            pot_size: Pot size before action
            street: Current street (0=preflop, 1=flop, etc.)
            is_facing_raise: Whether player faced a raise
        """
        stats = self.stats[player_id]

        # Track VPIP (any voluntary money preflop)
        if street == 0 and action_type in ("call", "bet", "raise", "all_in"):
            self._current_hand_data["vpip_players"].add(player_id)

        # Track PFR
        if street == 0 and action_type in ("bet", "raise", "all_in"):
            if self._current_hand_data["preflop_raiser"] is None:
                self._current_hand_data["preflop_raiser"] = player_id
                stats.pfr_hands += 1
            elif is_facing_raise:
                # This is a 3-bet
                stats.threebets += 1

        # Track aggression
        if action_type in ("bet", "raise", "all_in"):
            stats.aggression_actions += 1
            if pot_size > 0:
                stats.total_bets.append(amount / pot_size)
        elif action_type in ("call", "check"):
            stats.passive_actions += 1

        # Track c-bet
        if street == 1:  # Flop
            if self._current_hand_data.get("postflop_aggressor") is None:
                if action_type in ("bet", "raise", "all_in"):
                    self._current_hand_data["postflop_aggressor"] = player_id
                    # Check if this player was preflop raiser
                    if self._current_hand_data["preflop_raiser"] == player_id:
                        stats.cbet_opportunities += 1
                        stats.cbets_made += 1
            else:
                # Facing c-bet
                if action_type == "fold":
                    stats.folds_to_cbet += 1
                    stats.cbet_faced += 1
                elif action_type in ("call", "raise"):
                    stats.cbet_faced += 1

    def end_hand(
        self,
        active_players: List[int],
        went_to_showdown: bool,
        winner_ids: List[int],
    ):
        """Finalize hand statistics.

        Args:
            active_players: Players who were in the hand
            went_to_showdown: Whether hand went to showdown
            winner_ids: IDs of winning players
        """
        for player_id in active_players:
            self.stats[player_id].hands_played += 1

        for player_id in self._current_hand_data.get("vpip_players", set()):
            self.stats[player_id].vpip_hands += 1

        if went_to_showdown:
            for player_id in active_players:
                self.stats[player_id].showdown_hands += 1
            for winner_id in winner_ids:
                self.stats[winner_id].showdown_wins += 1

        # Check for missed c-bet opportunities
        preflop_raiser = self._current_hand_data.get("preflop_raiser")
        postflop_aggressor = self._current_hand_data.get("postflop_aggressor")
        if preflop_raiser is not None and postflop_aggressor != preflop_raiser:
            # Preflop raiser didn't c-bet
            if preflop_raiser in active_players:
                self.stats[preflop_raiser].cbet_opportunities += 1

        self.reset_hand()

    def get_opponent_features(self, opponent_id: int) -> np.ndarray:
        """Get feature vector for an opponent.

        Args:
            opponent_id: ID of the opponent

        Returns:
            10-dimensional feature vector
        """
        return self.stats[opponent_id].to_feature_vector()

    def get_opponent_profile(self, opponent_id: int) -> str:
        """Get player type classification.

        Args:
            opponent_id: ID of the opponent

        Returns:
            Player type string
        """
        return PlayerProfile.classify(self.stats[opponent_id])


class MultiPlayerFeatureExtractor:
    """Enhanced feature extractor for multi-player games."""

    # Feature dimensions
    OPPONENT_FEATURE_DIM = 10  # OpponentStats features
    POSITION_FEATURE_DIM = 5   # Relative position features
    STACK_FEATURE_DIM = 9      # Stack-related features per player

    def __init__(
        self,
        max_players: int = 9,
        preflop_buckets: int = 7,
        postflop_buckets: int = 10,
        use_opponent_modeling: bool = True,
    ):
        """Initialize multi-player feature extractor.

        Args:
            max_players: Maximum number of players supported
            preflop_buckets: Number of preflop hand buckets
            postflop_buckets: Number of postflop hand buckets
            use_opponent_modeling: Whether to include opponent modeling features
        """
        self.max_players = max_players
        self.preflop_buckets = preflop_buckets
        self.postflop_buckets = postflop_buckets
        self.use_opponent_modeling = use_opponent_modeling

        self.preflop_bucketing = PreflopBucketing()
        self.postflop_bucketing = PostflopBucketing()
        self.opponent_tracker = OpponentTracker(max_players) if use_opponent_modeling else None

        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimension."""
        dim = 0

        # Hand strength features
        dim += self.preflop_buckets  # Preflop bucket one-hot
        dim += self.postflop_buckets  # Postflop bucket one-hot

        # Street encoding
        dim += 4  # Street one-hot

        # Position features
        dim += self.max_players  # Absolute position one-hot
        dim += self.POSITION_FEATURE_DIM  # Relative position features

        # Player states (for each seat)
        dim += self.max_players * 4  # stack, active, acted_this_round, total_bet

        # Pot and betting features
        dim += 8  # Pot odds, stack/pot, to_call/pot, raises, etc.

        # Board texture
        dim += 5  # Board texture features

        # Opponent modeling (if enabled)
        if self.use_opponent_modeling:
            dim += self.max_players * self.OPPONENT_FEATURE_DIM

        return dim

    def extract_features(
        self,
        env,
        obs: Dict,
        player_id: int,
        opponent_features: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Extract features from game state for a specific player.

        Args:
            env: Poker environment instance
            obs: Observation dictionary from environment
            player_id: ID of the player to extract features for
            opponent_features: Pre-computed opponent features (optional)

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Get game state info
        raw_hole_cards = obs.get("hole_cards", [(0, 0), (0, 0)])
        board = obs.get("board", [])
        street = obs.get("street", 0)
        pot = obs.get("pot", 0)
        to_call = obs.get("to_call", 0)
        stacks = obs.get("stacks", [10000] * self.max_players)
        active = obs.get("active_players", [True] * self.max_players)
        acted_this_round = obs.get("acted_this_round", [False] * self.max_players)
        bets_this_round = obs.get("bets_this_round", [0] * self.max_players)
        button_pos = obs.get("button", 0)
        num_players = obs.get("num_players", len(stacks))
        raises = obs.get("raises_this_street", 0)

        # Parse hole cards to handle different formats
        hole_cards = [self._parse_card(c) for c in raw_hole_cards]

        # Ensure lists are the right size
        stacks = self._pad_list(stacks, self.max_players, 0)
        active = self._pad_list(active, self.max_players, False)
        acted_this_round = self._pad_list(acted_this_round, self.max_players, False)
        bets_this_round = self._pad_list(bets_this_round, self.max_players, 0)

        # 1. Hand strength features
        # Preflop bucket (one-hot)
        preflop_bucket = self.preflop_bucketing.get_bucket(
            hole_cards[0][0], hole_cards[0][1],
            hole_cards[1][0], hole_cards[1][1]
        )
        preflop_onehot = np.zeros(self.preflop_buckets, dtype=np.float32)
        preflop_onehot[preflop_bucket] = 1.0
        features.append(preflop_onehot)

        # Postflop bucket (one-hot)
        # Note: Full postflop bucketing uses Monte Carlo which is slow
        # For fast inference, we use a simplified version based on preflop bucket
        postflop_onehot = np.zeros(self.postflop_buckets, dtype=np.float32)
        if len(board) >= 3:
            # Simple approximation: use preflop bucket as base
            # and adjust based on board connectivity
            base_bucket = min(preflop_bucket, self.postflop_buckets - 1)
            postflop_onehot[base_bucket] = 1.0
        features.append(postflop_onehot)

        # 2. Street encoding (one-hot)
        street_onehot = np.zeros(4, dtype=np.float32)
        street_onehot[min(street, 3)] = 1.0
        features.append(street_onehot)

        # 3. Position features
        # Absolute position (one-hot)
        position_onehot = np.zeros(self.max_players, dtype=np.float32)
        position_onehot[player_id] = 1.0
        features.append(position_onehot)

        # Relative position features
        relative_pos = self._compute_relative_position(
            player_id, button_pos, num_players, active[:num_players]
        )
        features.append(relative_pos)

        # 4. Player state features (per seat)
        player_features = []
        max_stack = max(stacks[:num_players]) if num_players > 0 else 10000
        for i in range(self.max_players):
            if i < num_players:
                player_features.extend([
                    stacks[i] / max(max_stack, 1),  # Normalized stack
                    1.0 if active[i] else 0.0,
                    1.0 if acted_this_round[i] else 0.0,
                    bets_this_round[i] / max(pot, 1) if pot > 0 else 0.0,
                ])
            else:
                player_features.extend([0.0, 0.0, 0.0, 0.0])
        features.append(np.array(player_features, dtype=np.float32))

        # 5. Pot and betting features
        my_stack = stacks[player_id]
        pot_features = np.array([
            to_call / max(pot + to_call, 1),  # Pot odds
            min(my_stack / max(pot, 1), 10.0) / 10.0,  # Stack-to-pot ratio
            to_call / max(my_stack, 1),  # To call as % of stack
            min(raises, 4) / 4.0,  # Raises this street
            1.0 if to_call > 0 else 0.0,  # Facing bet
            sum(1 for a in active[:num_players] if a) / num_players,  # % active
            pot / max(sum(stacks[:num_players]), 1),  # Pot as % of total chips
            min(sum(bets_this_round[:num_players]), pot) / max(pot, 1),  # Action this round
        ], dtype=np.float32)
        features.append(pot_features)

        # 6. Board texture features
        board_features = self._extract_board_features(board)
        features.append(board_features)

        # 7. Opponent modeling features (if enabled)
        if self.use_opponent_modeling:
            opp_features = []
            for i in range(self.max_players):
                if i != player_id and i < num_players:
                    if opponent_features and i in opponent_features:
                        opp_features.append(opponent_features[i])
                    elif self.opponent_tracker:
                        opp_features.append(self.opponent_tracker.get_opponent_features(i))
                    else:
                        opp_features.append(np.zeros(self.OPPONENT_FEATURE_DIM, dtype=np.float32))
                else:
                    opp_features.append(np.zeros(self.OPPONENT_FEATURE_DIM, dtype=np.float32))
            features.append(np.concatenate(opp_features))

        return np.concatenate(features).astype(np.float32)

    def _pad_list(self, lst: List, target_len: int, default) -> List:
        """Pad a list to target length."""
        if len(lst) >= target_len:
            return lst[:target_len]
        return lst + [default] * (target_len - len(lst))

    def _compute_relative_position(
        self,
        player_id: int,
        button_pos: int,
        num_players: int,
        active: List[bool],
    ) -> np.ndarray:
        """Compute relative position features.

        Returns:
            5-dimensional position feature vector:
            - Distance from button (normalized)
            - Is button
            - Is small blind
            - Is big blind
            - Players to act before us (normalized)
        """
        features = np.zeros(self.POSITION_FEATURE_DIM, dtype=np.float32)

        # Distance from button (0 = button, higher = earlier position)
        distance = (player_id - button_pos) % num_players
        features[0] = distance / max(num_players - 1, 1)

        # Position flags
        features[1] = 1.0 if distance == 0 else 0.0  # Is button
        features[2] = 1.0 if distance == 1 else 0.0  # Is small blind
        features[3] = 1.0 if distance == 2 else 0.0  # Is big blind

        # Count active players to act before us
        players_to_act = 0
        for i in range(1, num_players):
            check_pos = (player_id + i) % num_players
            if check_pos < len(active) and active[check_pos]:
                players_to_act += 1
        features[4] = players_to_act / max(num_players - 1, 1)

        return features

    def _parse_card(self, card) -> Tuple[int, int]:
        """Parse a card into (rank, suit) tuple.

        Handles multiple formats from the Rust engine.
        """
        if isinstance(card, (list, tuple)) and len(card) >= 2:
            r, s = card[0], card[1]
            # Handle string ranks
            if isinstance(r, str):
                rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                           '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                r = rank_map.get(r, 0)
            if isinstance(s, str):
                suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3,
                           'clubs': 0, 'diamonds': 1, 'hearts': 2, 'spades': 3}
                s = suit_map.get(s.lower(), 0)
            return (int(r), int(s))
        elif hasattr(card, 'rank') and hasattr(card, 'suit'):
            r = card.rank
            s = card.suit
            if isinstance(r, str):
                rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                           '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                r = rank_map.get(r, 0)
            if isinstance(s, str):
                suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3,
                           'clubs': 0, 'diamonds': 1, 'hearts': 2, 'spades': 3}
                s = suit_map.get(s.lower(), 0)
            return (int(r), int(s))
        return (0, 0)

    def _extract_board_features(self, board: List[Tuple[int, int]]) -> np.ndarray:
        """Extract board texture features."""
        features = np.zeros(5, dtype=np.float32)

        if not board:
            return features

        # Parse cards to handle different formats
        parsed_board = [self._parse_card(c) for c in board]
        ranks = [r for r, s in parsed_board]
        suits = [s for r, s in parsed_board]

        # High card (normalized)
        features[0] = max(ranks) / 12.0 if ranks else 0.0

        # Flush possible (3+ same suit)
        suit_counts = [suits.count(s) for s in range(4)]
        features[1] = 1.0 if max(suit_counts) >= 3 else 0.0

        # Straight possible (connected cards)
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

        # Board wetness (combination)
        features[4] = (features[1] + features[2] + features[3]) / 3.0

        return features


def create_multi_player_extractor(
    max_players: int = 9,
    use_opponent_modeling: bool = True,
) -> MultiPlayerFeatureExtractor:
    """Create a multi-player feature extractor.

    Args:
        max_players: Maximum players (2-9)
        use_opponent_modeling: Include opponent modeling features

    Returns:
        Configured MultiPlayerFeatureExtractor
    """
    return MultiPlayerFeatureExtractor(
        max_players=max_players,
        use_opponent_modeling=use_opponent_modeling,
    )


# Testing
if __name__ == "__main__":
    print("=== Multi-Player Feature Extractor Test ===\n")

    # Create extractor
    extractor = create_multi_player_extractor(max_players=6)
    print(f"Feature dimension: {extractor.feature_dim}")

    # Test opponent tracking
    tracker = OpponentTracker(max_players=6)
    tracker.reset_hand()

    # Simulate some actions
    tracker.record_action(1, "raise", 300, 150, street=0)
    tracker.record_action(2, "call", 300, 450, street=0)
    tracker.record_action(1, "bet", 400, 600, street=1)  # c-bet
    tracker.record_action(2, "fold", 0, 1000, street=1)

    tracker.end_hand(active_players=[1, 2], went_to_showdown=False, winner_ids=[1])

    # Check stats
    stats1 = tracker.stats[1]
    stats2 = tracker.stats[2]

    print(f"\nPlayer 1 stats:")
    print(f"  VPIP: {stats1.vpip:.2f}")
    print(f"  PFR: {stats1.pfr:.2f}")
    print(f"  C-bet: {stats1.cbet_pct:.2f}")
    print(f"  AF: {stats1.aggression_factor:.2f}")

    print(f"\nPlayer 2 stats:")
    print(f"  VPIP: {stats2.vpip:.2f}")
    print(f"  Fold to c-bet: {stats2.fold_to_cbet:.2f}")

    # Test feature extraction with mock observation
    mock_obs = {
        "hole_cards": [(12, 0), (12, 1)],  # AA
        "board": [(10, 0), (5, 1), (2, 2)],  # Flop
        "street": 1,
        "pot": 600,
        "to_call": 300,
        "stacks": [9700, 9400, 10000, 10000, 10000, 10000],
        "active_players": [True, True, False, False, False, False],
        "acted_this_round": [True, False, False, False, False, False],
        "bets_this_round": [300, 0, 0, 0, 0, 0],
        "button": 0,
        "num_players": 6,
        "raises_this_street": 1,
    }

    features = extractor.extract_features(None, mock_obs, player_id=1)
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Expected shape: ({extractor.feature_dim},)")
    assert features.shape[0] == extractor.feature_dim, "Feature dimension mismatch!"

    print("\n✓ All tests passed!")
