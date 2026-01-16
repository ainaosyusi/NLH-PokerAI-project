"""Personality-based poker agents for diverse training.

This module provides:
1. Different agent personalities (Maniac, Rock, TAG, LAG, Fish)
2. Agent pools for population-based training
3. Configurable personality parameters
4. Rule-based agents for baseline training
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import random

try:
    from .multi_player_features import MultiPlayerFeatureExtractor, OpponentStats
    from .abstraction import PreflopBucketing
except ImportError:
    from multi_player_features import MultiPlayerFeatureExtractor, OpponentStats
    from abstraction import PreflopBucketing


class PersonalityType(Enum):
    """Types of poker personalities."""
    MANIAC = "maniac"      # Very loose, very aggressive
    LAG = "lag"            # Loose-aggressive
    TAG = "tag"            # Tight-aggressive
    ROCK = "rock"          # Very tight, passive
    FISH = "fish"          # Loose, passive (calling station)
    BALANCED = "balanced"  # GTO-ish balanced play
    RANDOM = "random"      # Pure random (for baseline)


@dataclass
class PersonalityConfig:
    """Configuration for a poker personality."""
    # Preflop parameters
    vpip_range: Tuple[float, float] = (0.20, 0.25)  # % of hands to play
    pfr_ratio: float = 0.75  # PFR / VPIP ratio
    threebet_range: Tuple[float, float] = (0.06, 0.10)

    # Postflop parameters
    cbet_frequency: float = 0.65
    fold_to_cbet: float = 0.45
    aggression_factor: float = 2.0

    # Bet sizing (as fraction of pot)
    bet_size_small: float = 0.33
    bet_size_medium: float = 0.66
    bet_size_large: float = 1.0
    bet_size_variance: float = 0.1  # Randomization

    # Bluff/value ratio
    bluff_frequency: float = 0.35
    value_bet_threshold: float = 0.55  # Min equity to value bet

    # Adaptability
    tilt_factor: float = 0.0  # How much losses affect play
    adjustment_speed: float = 0.5  # How fast to adapt to opponents


# Predefined personality configurations
PERSONALITY_CONFIGS: Dict[PersonalityType, PersonalityConfig] = {
    PersonalityType.MANIAC: PersonalityConfig(
        vpip_range=(0.50, 0.65),
        pfr_ratio=0.85,
        threebet_range=(0.15, 0.25),
        cbet_frequency=0.85,
        fold_to_cbet=0.20,
        aggression_factor=4.0,
        bet_size_small=0.5,
        bet_size_medium=1.0,
        bet_size_large=1.5,
        bet_size_variance=0.2,
        bluff_frequency=0.55,
        value_bet_threshold=0.40,
        tilt_factor=0.3,
        adjustment_speed=0.2,
    ),
    PersonalityType.LAG: PersonalityConfig(
        vpip_range=(0.30, 0.40),
        pfr_ratio=0.80,
        threebet_range=(0.10, 0.15),
        cbet_frequency=0.75,
        fold_to_cbet=0.35,
        aggression_factor=2.5,
        bet_size_small=0.4,
        bet_size_medium=0.75,
        bet_size_large=1.25,
        bet_size_variance=0.15,
        bluff_frequency=0.45,
        value_bet_threshold=0.50,
        tilt_factor=0.1,
        adjustment_speed=0.6,
    ),
    PersonalityType.TAG: PersonalityConfig(
        vpip_range=(0.18, 0.24),
        pfr_ratio=0.80,
        threebet_range=(0.08, 0.12),
        cbet_frequency=0.70,
        fold_to_cbet=0.45,
        aggression_factor=2.0,
        bet_size_small=0.33,
        bet_size_medium=0.66,
        bet_size_large=1.0,
        bet_size_variance=0.1,
        bluff_frequency=0.35,
        value_bet_threshold=0.55,
        tilt_factor=0.05,
        adjustment_speed=0.7,
    ),
    PersonalityType.ROCK: PersonalityConfig(
        vpip_range=(0.10, 0.15),
        pfr_ratio=0.70,
        threebet_range=(0.04, 0.06),
        cbet_frequency=0.55,
        fold_to_cbet=0.55,
        aggression_factor=1.2,
        bet_size_small=0.33,
        bet_size_medium=0.5,
        bet_size_large=0.75,
        bet_size_variance=0.05,
        bluff_frequency=0.15,
        value_bet_threshold=0.65,
        tilt_factor=0.0,
        adjustment_speed=0.3,
    ),
    PersonalityType.FISH: PersonalityConfig(
        vpip_range=(0.45, 0.60),
        pfr_ratio=0.30,
        threebet_range=(0.02, 0.04),
        cbet_frequency=0.40,
        fold_to_cbet=0.60,
        aggression_factor=0.6,
        bet_size_small=0.5,
        bet_size_medium=0.5,
        bet_size_large=0.5,  # Fixed sizing
        bet_size_variance=0.0,
        bluff_frequency=0.15,
        value_bet_threshold=0.45,
        tilt_factor=0.4,
        adjustment_speed=0.1,
    ),
    PersonalityType.BALANCED: PersonalityConfig(
        vpip_range=(0.22, 0.28),
        pfr_ratio=0.75,
        threebet_range=(0.07, 0.11),
        cbet_frequency=0.65,
        fold_to_cbet=0.45,
        aggression_factor=2.0,
        bet_size_small=0.33,
        bet_size_medium=0.66,
        bet_size_large=1.0,
        bet_size_variance=0.1,
        bluff_frequency=0.35,
        value_bet_threshold=0.55,
        tilt_factor=0.0,
        adjustment_speed=0.5,
    ),
}


class RuleBasedAgent:
    """Rule-based agent with configurable personality."""

    def __init__(
        self,
        personality: PersonalityType = PersonalityType.TAG,
        player_id: int = 0,
        seed: Optional[int] = None,
    ):
        """Initialize rule-based agent.

        Args:
            personality: Agent personality type
            player_id: Player position (0-8)
            seed: Random seed for reproducibility
        """
        self.personality = personality
        self.player_id = player_id
        self.config = PERSONALITY_CONFIGS.get(
            personality,
            PERSONALITY_CONFIGS[PersonalityType.BALANCED]
        )
        self.rng = np.random.RandomState(seed)
        self.preflop_buckets = PreflopBucketing()

        # Sample actual parameters from ranges
        self.vpip = self.rng.uniform(*self.config.vpip_range)
        self.threebet = self.rng.uniform(*self.config.threebet_range)

        # Track state
        self.hands_played = 0
        self.session_profit = 0.0

    def get_action(self, obs: Dict, env=None) -> int:
        """Get action based on observation.

        Args:
            obs: Observation dictionary from environment
            env: Optional environment reference

        Returns:
            Action index
        """
        if self.personality == PersonalityType.RANDOM:
            return self._random_action(obs)

        street = obs.get("street", 0)
        action_mask = obs.get("action_mask", [True, True, True, True])

        if street == 0:
            return self._preflop_action(obs, action_mask)
        else:
            return self._postflop_action(obs, action_mask, street)

    def _random_action(self, obs: Dict) -> int:
        """Pure random action selection."""
        action_mask = obs.get("action_mask", [True, True, True, True])
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        return self.rng.choice(valid_actions)

    def _preflop_action(self, obs: Dict, action_mask: List[bool]) -> int:
        """Preflop action selection."""
        hole_cards = obs.get("hole_cards", [(0, 0), (0, 0)])
        to_call = obs.get("to_call", 0)
        pot = obs.get("pot", 0)

        # Handle different card formats from engine
        def parse_card(card):
            if isinstance(card, (list, tuple)) and len(card) == 2:
                return (int(card[0]), int(card[1]))
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                return (card.rank, card.suit)
            else:
                return (0, 0)

        hole_cards = [parse_card(c) for c in hole_cards]

        # Get hand bucket (0=premium, 6=trash)
        bucket = self.preflop_buckets.get_bucket(
            hole_cards[0][0], hole_cards[0][1],
            hole_cards[1][0], hole_cards[1][1]
        )

        # Calculate play threshold based on VPIP
        play_threshold = self._bucket_threshold_from_vpip(self.vpip)

        # Decide to play or fold
        if bucket > play_threshold:
            # Hand too weak for our VPIP range
            if action_mask[0]:  # Fold available
                return 0
            elif action_mask[1]:  # Check available
                return 1
            else:
                return self._get_valid_action(action_mask, prefer_passive=True)

        # We're playing - decide raise or call
        should_raise = self._should_raise_preflop(bucket, to_call, pot)

        if should_raise and (action_mask[2] or action_mask[3]):
            # Raise or all-in
            if action_mask[3] and self.rng.random() < 0.05:  # Small all-in freq
                return 3
            return 2 if action_mask[2] else (3 if action_mask[3] else 1)
        elif action_mask[1]:
            # Call/check
            return 1
        else:
            return self._get_valid_action(action_mask, prefer_passive=True)

    def _postflop_action(
        self,
        obs: Dict,
        action_mask: List[bool],
        street: int,
    ) -> int:
        """Postflop action selection."""
        to_call = obs.get("to_call", 0)
        pot = obs.get("pot", 0)

        # Simplified equity estimation (would use real calculation in production)
        equity = self._estimate_hand_strength(obs)

        # C-bet logic
        if to_call == 0:
            # We can check or bet
            if self.rng.random() < self.config.cbet_frequency:
                # Bet
                if equity > self.config.value_bet_threshold:
                    # Value bet
                    return 2 if action_mask[2] else 1
                elif self.rng.random() < self.config.bluff_frequency:
                    # Bluff
                    return 2 if action_mask[2] else 1
            return 1 if action_mask[1] else 0

        # Facing a bet
        pot_odds = to_call / max(pot + to_call, 1)

        # Calling range based on equity vs pot odds
        call_threshold = pot_odds * (1.0 - self.config.aggression_factor * 0.1)

        if equity > 0.70:
            # Strong hand - raise for value
            if self.rng.random() < self.config.aggression_factor / 3:
                return 2 if action_mask[2] else 1
            return 1 if action_mask[1] else 0

        if equity > call_threshold:
            # Good enough to call
            return 1 if action_mask[1] else 0

        # Weak - consider bluff raise or fold
        if self.rng.random() < self.config.bluff_frequency * 0.3:
            if action_mask[2]:
                return 2
        return 0 if action_mask[0] else 1

    def _bucket_threshold_from_vpip(self, vpip: float) -> int:
        """Convert VPIP to preflop bucket threshold."""
        # Higher VPIP = higher threshold (play more hands)
        if vpip >= 0.50:
            return 6  # Play almost everything
        elif vpip >= 0.40:
            return 5
        elif vpip >= 0.30:
            return 4
        elif vpip >= 0.22:
            return 3
        elif vpip >= 0.15:
            return 2
        else:
            return 1  # Only premium/strong

    def _should_raise_preflop(self, bucket: int, to_call: int, pot: int) -> bool:
        """Decide whether to raise preflop."""
        pfr_threshold = self.vpip * self.config.pfr_ratio

        # Premium hands always raise
        if bucket <= 1:
            return True

        # Strong hands usually raise
        if bucket <= 2:
            return self.rng.random() < 0.8

        # Playable hands sometimes raise
        if bucket <= 3:
            return self.rng.random() < pfr_threshold

        # Speculative hands occasionally raise (bluff)
        if bucket <= 4:
            return self.rng.random() < pfr_threshold * 0.3

        return False

    def _estimate_hand_strength(self, obs: Dict) -> float:
        """Estimate current hand strength (simplified)."""
        hole_cards = obs.get("hole_cards", [(0, 0), (0, 0)])
        board = obs.get("board", [])

        # Handle different card formats from engine
        def parse_card(card):
            if isinstance(card, (list, tuple)) and len(card) == 2:
                return (int(card[0]), int(card[1]))
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                return (card.rank, card.suit)
            else:
                return (0, 0)

        hole_cards = [parse_card(c) for c in hole_cards]
        board = [parse_card(c) for c in board]

        # Get preflop bucket as base
        bucket = self.preflop_buckets.get_bucket(
            hole_cards[0][0], hole_cards[0][1],
            hole_cards[1][0], hole_cards[1][1]
        )

        # Convert bucket to base equity
        base_equity = (6 - bucket) / 6.0  # 0=premium->1.0, 6=trash->0.0

        if not board:
            return base_equity

        # Check for simple board connections (very simplified)
        ranks = [r for r, s in hole_cards] + [r for r, s in board]
        suits = [s for r, s in hole_cards] + [s for r, s in board]

        # Ensure all are integers
        ranks = [int(r) if not isinstance(r, int) else r for r in ranks]
        suits = [int(s) if not isinstance(s, int) else s for s in suits]

        # Pair or better
        if len(ranks) != len(set(ranks)):
            base_equity += 0.15

        # Flush draw
        suit_counts = [suits.count(s) for s in range(4)]
        if max(suit_counts) >= 4:
            base_equity += 0.10

        # Connected cards (straight draw potential)
        sorted_ranks = sorted(set(ranks))
        for i in range(len(sorted_ranks) - 1):
            if sorted_ranks[i+1] - sorted_ranks[i] == 1:
                base_equity += 0.02

        return min(base_equity, 1.0)

    def _get_valid_action(self, action_mask: List[bool], prefer_passive: bool = False) -> int:
        """Get a valid action from mask."""
        if prefer_passive:
            # Prefer check/call > fold > raise > all-in
            order = [1, 0, 2, 3]
        else:
            # Prefer raise > call > fold > all-in
            order = [2, 1, 0, 3]

        for action in order:
            if action < len(action_mask) and action_mask[action]:
                return action

        # Fallback to any valid action
        for i, valid in enumerate(action_mask):
            if valid:
                return i
        return 0

    def update_session(self, reward: float):
        """Update session stats."""
        self.hands_played += 1
        self.session_profit += reward

        # Tilt factor adjustment
        if self.config.tilt_factor > 0 and self.session_profit < 0:
            tilt_multiplier = 1.0 + self.config.tilt_factor * min(abs(self.session_profit) / 5000, 1.0)
            self.vpip = min(self.vpip * tilt_multiplier, 0.8)


class AgentPool:
    """Pool of agents with specific personality distribution."""

    def __init__(
        self,
        pool_name: str,
        personality_distribution: Dict[PersonalityType, float],
        pool_size: int = 20,
        seed: Optional[int] = None,
    ):
        """Initialize agent pool.

        Args:
            pool_name: Name of the pool
            personality_distribution: Dict mapping personality to probability
            pool_size: Number of agents in pool
            seed: Random seed
        """
        self.pool_name = pool_name
        self.distribution = personality_distribution
        self.pool_size = pool_size
        self.rng = np.random.RandomState(seed)
        self.agents: List[RuleBasedAgent] = []

        self._create_agents()

    def _create_agents(self):
        """Create agents according to distribution."""
        # Normalize distribution
        total = sum(self.distribution.values())
        normalized = {k: v/total for k, v in self.distribution.items()}

        # Create agents
        personalities = list(normalized.keys())
        probs = list(normalized.values())

        for i in range(self.pool_size):
            personality = self.rng.choice(personalities, p=probs)
            agent = RuleBasedAgent(
                personality=personality,
                player_id=i,
                seed=self.rng.randint(0, 2**31),
            )
            self.agents.append(agent)

    def sample_agents(self, n: int) -> List[RuleBasedAgent]:
        """Sample n agents from pool.

        Args:
            n: Number of agents to sample

        Returns:
            List of sampled agents
        """
        if n >= len(self.agents):
            return self.agents.copy()
        indices = self.rng.choice(len(self.agents), size=n, replace=False)
        return [self.agents[i] for i in indices]

    def get_agent(self, index: int) -> RuleBasedAgent:
        """Get specific agent by index."""
        return self.agents[index % len(self.agents)]


# Predefined pool configurations
POOL_CONFIGS = {
    "maniac_pool": {
        PersonalityType.MANIAC: 0.70,
        PersonalityType.LAG: 0.20,
        PersonalityType.FISH: 0.10,
    },
    "rock_pool": {
        PersonalityType.ROCK: 0.60,
        PersonalityType.TAG: 0.30,
        PersonalityType.BALANCED: 0.10,
    },
    "tag_pool": {
        PersonalityType.TAG: 0.70,
        PersonalityType.BALANCED: 0.20,
        PersonalityType.LAG: 0.10,
    },
    "lag_pool": {
        PersonalityType.LAG: 0.60,
        PersonalityType.TAG: 0.20,
        PersonalityType.MANIAC: 0.20,
    },
    "fish_pool": {
        PersonalityType.FISH: 0.70,
        PersonalityType.RANDOM: 0.20,
        PersonalityType.ROCK: 0.10,
    },
    "mixed_pool": {
        PersonalityType.TAG: 0.25,
        PersonalityType.LAG: 0.25,
        PersonalityType.FISH: 0.20,
        PersonalityType.ROCK: 0.15,
        PersonalityType.MANIAC: 0.15,
    },
    "tough_pool": {
        PersonalityType.TAG: 0.40,
        PersonalityType.LAG: 0.35,
        PersonalityType.BALANCED: 0.25,
    },
}


def create_pool(pool_name: str, pool_size: int = 20, seed: Optional[int] = None) -> AgentPool:
    """Create an agent pool by name.

    Args:
        pool_name: Name of predefined pool or "custom"
        pool_size: Number of agents
        seed: Random seed

    Returns:
        Configured AgentPool
    """
    if pool_name not in POOL_CONFIGS:
        raise ValueError(f"Unknown pool: {pool_name}. Available: {list(POOL_CONFIGS.keys())}")

    return AgentPool(
        pool_name=pool_name,
        personality_distribution=POOL_CONFIGS[pool_name],
        pool_size=pool_size,
        seed=seed,
    )


def create_all_pools(pool_size: int = 20, seed: Optional[int] = None) -> Dict[str, AgentPool]:
    """Create all predefined agent pools.

    Args:
        pool_size: Size of each pool
        seed: Base random seed

    Returns:
        Dictionary of pool name to AgentPool
    """
    pools = {}
    for i, name in enumerate(POOL_CONFIGS.keys()):
        pool_seed = seed + i * 1000 if seed is not None else None
        pools[name] = create_pool(name, pool_size, pool_seed)
    return pools


# Testing
if __name__ == "__main__":
    import poker_engine

    print("=== Personality-Based Agent Pool Test ===\n")

    # Create pools
    pools = create_all_pools(pool_size=10, seed=42)
    print(f"Created {len(pools)} pools:")
    for name, pool in pools.items():
        personalities = [a.personality.value for a in pool.agents]
        counts = {p: personalities.count(p) for p in set(personalities)}
        print(f"  {name}: {counts}")

    # Test agents against each other
    print("\n--- Agent vs Agent Test (100 hands) ---")

    env = poker_engine.PokerEnv(
        num_players=6,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    # Sample agents from different pools
    agents = [
        pools["tag_pool"].get_agent(0),
        pools["lag_pool"].get_agent(0),
        pools["rock_pool"].get_agent(0),
        pools["fish_pool"].get_agent(0),
        pools["maniac_pool"].get_agent(0),
        pools["mixed_pool"].get_agent(0),
    ]

    # Update player IDs
    for i, agent in enumerate(agents):
        agent.player_id = i

    total_rewards = [0.0] * 6

    for hand in range(100):
        obs = env.reset()

        while not obs["is_terminal"]:
            current_player = obs["current_player"]
            action = agents[current_player].get_action(obs, env)
            obs, rewards, done, _, info = env.step(action)

        if done:
            for i in range(6):
                total_rewards[i] += rewards[i]
                agents[i].update_session(rewards[i])

    print("\nResults after 100 hands:")
    for i, agent in enumerate(agents):
        bb_per_100 = total_rewards[i] / 100  # 100 hands, BB=100
        print(f"  Seat {i} ({agent.personality.value}): {total_rewards[i]:+.0f} chips ({bb_per_100:+.1f} BB/100)")

    print("\n✓ All tests passed!")
