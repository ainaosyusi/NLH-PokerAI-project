"""Tournament and league system for poker AI evaluation.

This module provides:
1. Round-robin tournament play
2. ELO rating calculation
3. League tables and rankings
4. Match history and statistics
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os
from datetime import datetime

try:
    from .personality_agents import (
        RuleBasedAgent,
        AgentPool,
        PersonalityType,
        create_pool,
    )
except ImportError:
    from personality_agents import (
        RuleBasedAgent,
        AgentPool,
        PersonalityType,
        create_pool,
    )


@dataclass
class MatchResult:
    """Result of a single match between agents."""
    agent1_id: str
    agent2_id: str
    agent1_profit: float
    agent2_profit: float
    num_hands: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def agent1_bb100(self) -> float:
        """Agent 1's BB/100."""
        return self.agent1_profit / (self.num_hands / 100) / 100

    @property
    def agent2_bb100(self) -> float:
        """Agent 2's BB/100."""
        return self.agent2_profit / (self.num_hands / 100) / 100

    @property
    def winner(self) -> Optional[str]:
        """Winner of the match (or None for tie)."""
        if self.agent1_profit > self.agent2_profit:
            return self.agent1_id
        elif self.agent2_profit > self.agent1_profit:
            return self.agent2_id
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "agent1_id": self.agent1_id,
            "agent2_id": self.agent2_id,
            "agent1_profit": self.agent1_profit,
            "agent2_profit": self.agent2_profit,
            "agent1_bb100": self.agent1_bb100,
            "agent2_bb100": self.agent2_bb100,
            "num_hands": self.num_hands,
            "winner": self.winner,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentStats:
    """Statistics for an agent in the league."""
    agent_id: str
    elo_rating: float = 1200.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_profit: float = 0.0
    total_hands: int = 0
    peak_elo: float = 1200.0
    elo_history: List[float] = field(default_factory=lambda: [1200.0])

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played

    @property
    def bb_per_100(self) -> float:
        """BB/100 overall."""
        if self.total_hands == 0:
            return 0.0
        return self.total_profit / (self.total_hands / 100) / 100

    def update(self, result: MatchResult, is_agent1: bool, new_elo: float):
        """Update stats from match result.

        Args:
            result: Match result
            is_agent1: Whether this agent was agent1
            new_elo: New ELO rating
        """
        profit = result.agent1_profit if is_agent1 else result.agent2_profit
        other_profit = result.agent2_profit if is_agent1 else result.agent1_profit

        self.matches_played += 1
        self.total_profit += profit
        self.total_hands += result.num_hands

        if profit > other_profit:
            self.wins += 1
        elif profit < other_profit:
            self.losses += 1
        else:
            self.draws += 1

        self.elo_rating = new_elo
        self.peak_elo = max(self.peak_elo, new_elo)
        self.elo_history.append(new_elo)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "elo_rating": self.elo_rating,
            "matches_played": self.matches_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "total_profit": self.total_profit,
            "total_hands": self.total_hands,
            "bb_per_100": self.bb_per_100,
            "peak_elo": self.peak_elo,
        }


class ELOCalculator:
    """Calculate ELO ratings for matches."""

    def __init__(self, k_factor: float = 32.0, base_rating: float = 1200.0):
        """Initialize ELO calculator.

        Args:
            k_factor: How much ratings change per match
            base_rating: Starting rating for new players
        """
        self.k_factor = k_factor
        self.base_rating = base_rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A.

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def calculate_new_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,  # 1 for win, 0.5 for draw, 0 for loss
    ) -> Tuple[float, float]:
        """Calculate new ratings after a match.

        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            score_a: Actual score for player A (1/0.5/0)

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = rating_a + self.k_factor * (score_a - expected_a)
        new_b = rating_b + self.k_factor * (score_b - expected_b)

        return new_a, new_b

    def calculate_from_result(
        self,
        rating_a: float,
        rating_b: float,
        result: MatchResult,
        is_a_agent1: bool = True,
    ) -> Tuple[float, float]:
        """Calculate new ratings from a match result.

        Args:
            rating_a: Rating of agent A
            rating_b: Rating of agent B
            result: Match result
            is_a_agent1: Whether agent A was agent1 in the result

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        profit_a = result.agent1_profit if is_a_agent1 else result.agent2_profit
        profit_b = result.agent2_profit if is_a_agent1 else result.agent1_profit

        if profit_a > profit_b:
            score_a = 1.0
        elif profit_a < profit_b:
            score_a = 0.0
        else:
            score_a = 0.5

        return self.calculate_new_ratings(rating_a, rating_b, score_a)


class Tournament:
    """Round-robin tournament system."""

    def __init__(
        self,
        name: str,
        hands_per_match: int = 500,
        num_rounds: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize tournament.

        Args:
            name: Tournament name
            hands_per_match: Hands to play per match
            num_rounds: Number of round-robin rounds
            seed: Random seed
        """
        self.name = name
        self.hands_per_match = hands_per_match
        self.num_rounds = num_rounds
        self.rng = np.random.RandomState(seed)

        self.participants: Dict[str, Any] = {}  # agent_id -> agent object
        self.stats: Dict[str, AgentStats] = {}
        self.matches: List[MatchResult] = []
        self.elo_calc = ELOCalculator()

    def add_participant(self, agent_id: str, agent: Any, initial_elo: float = 1200.0):
        """Add a participant to the tournament.

        Args:
            agent_id: Unique identifier
            agent: Agent object (must have get_action method)
            initial_elo: Starting ELO rating
        """
        self.participants[agent_id] = agent
        self.stats[agent_id] = AgentStats(agent_id=agent_id, elo_rating=initial_elo)

    def run(
        self,
        env,
        get_action_fn: Callable[[Any, Dict, Any], int],
        num_players: int = 2,
        verbose: bool = True,
    ) -> Dict[str, AgentStats]:
        """Run the tournament.

        Args:
            env: Poker environment
            get_action_fn: Function to get action from agent
            num_players: Players per table (2 for heads-up)
            verbose: Print progress

        Returns:
            Final statistics for all participants
        """
        agent_ids = list(self.participants.keys())

        if len(agent_ids) < 2:
            raise ValueError("Need at least 2 participants")

        # Generate round-robin pairings
        pairings = self._generate_pairings(agent_ids)

        total_matches = len(pairings) * self.num_rounds
        match_num = 0

        for round_num in range(self.num_rounds):
            if verbose:
                print(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")

            for agent1_id, agent2_id in pairings:
                match_num += 1

                if verbose:
                    print(f"Match {match_num}/{total_matches}: {agent1_id} vs {agent2_id}")

                # Play match
                result = self._play_match(
                    env,
                    agent1_id, self.participants[agent1_id],
                    agent2_id, self.participants[agent2_id],
                    get_action_fn,
                    num_players,
                )

                # Update ELO
                elo1 = self.stats[agent1_id].elo_rating
                elo2 = self.stats[agent2_id].elo_rating
                new_elo1, new_elo2 = self.elo_calc.calculate_from_result(
                    elo1, elo2, result, is_a_agent1=True
                )

                # Update stats
                self.stats[agent1_id].update(result, is_agent1=True, new_elo=new_elo1)
                self.stats[agent2_id].update(result, is_agent1=False, new_elo=new_elo2)
                self.matches.append(result)

                if verbose:
                    winner_str = result.winner if result.winner else "Draw"
                    print(
                        f"  Result: {result.agent1_bb100:+.1f} vs {result.agent2_bb100:+.1f} BB/100 | "
                        f"Winner: {winner_str}"
                    )

        return self.stats

    def _generate_pairings(self, agent_ids: List[str]) -> List[Tuple[str, str]]:
        """Generate round-robin pairings.

        Args:
            agent_ids: List of agent IDs

        Returns:
            List of (agent1_id, agent2_id) pairs
        """
        pairings = []
        n = len(agent_ids)

        for i in range(n):
            for j in range(i + 1, n):
                pairings.append((agent_ids[i], agent_ids[j]))

        # Shuffle for variety
        self.rng.shuffle(pairings)
        return pairings

    def _play_match(
        self,
        env,
        agent1_id: str,
        agent1: Any,
        agent2_id: str,
        agent2: Any,
        get_action_fn: Callable,
        num_players: int,
    ) -> MatchResult:
        """Play a match between two agents.

        Args:
            env: Poker environment
            agent1_id: ID of agent 1
            agent1: Agent 1 object
            agent2_id: ID of agent 2
            agent2: Agent 2 object
            get_action_fn: Action function
            num_players: Number of players

        Returns:
            Match result
        """
        profits = {0: 0.0, 1: 0.0}

        for hand in range(self.hands_per_match):
            obs = env.reset()

            while not obs["is_terminal"]:
                current = obs["current_player"]

                if current == 0:
                    action = get_action_fn(agent1, obs, env)
                elif current == 1:
                    action = get_action_fn(agent2, obs, env)
                else:
                    # Extra players - random action
                    mask = obs.get("action_mask", [True] * 4)
                    valid = [i for i, v in enumerate(mask) if v]
                    action = self.rng.choice(valid)

                obs, rewards, done, _, _ = env.step(action)

            if done:
                for i in range(min(2, len(rewards))):
                    profits[i] += rewards[i]

        return MatchResult(
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            agent1_profit=profits[0],
            agent2_profit=profits[1],
            num_hands=self.hands_per_match,
        )

    def get_standings(self) -> List[Dict]:
        """Get current standings sorted by ELO.

        Returns:
            List of agent stats as dicts, sorted by ELO
        """
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s.elo_rating,
            reverse=True
        )
        return [s.to_dict() for s in sorted_stats]

    def save(self, path: str):
        """Save tournament results.

        Args:
            path: Path to save file
        """
        data = {
            "name": self.name,
            "hands_per_match": self.hands_per_match,
            "num_rounds": self.num_rounds,
            "standings": self.get_standings(),
            "matches": [m.to_dict() for m in self.matches],
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class League:
    """Ongoing league with seasons and promotions."""

    def __init__(
        self,
        name: str,
        tiers: List[str] = None,
        matches_per_season: int = 100,
        promotion_slots: int = 2,
        relegation_slots: int = 2,
    ):
        """Initialize league.

        Args:
            name: League name
            tiers: Tier names (e.g., ["Diamond", "Gold", "Silver", "Bronze"])
            matches_per_season: Matches per season per agent
            promotion_slots: Agents promoted per tier
            relegation_slots: Agents relegated per tier
        """
        self.name = name
        self.tiers = tiers or ["Diamond", "Gold", "Silver", "Bronze"]
        self.matches_per_season = matches_per_season
        self.promotion_slots = promotion_slots
        self.relegation_slots = relegation_slots

        self.season = 1
        self.tier_members: Dict[str, List[str]] = {tier: [] for tier in self.tiers}
        self.all_stats: Dict[str, AgentStats] = {}
        self.season_history: List[Dict] = []

    def add_agent(self, agent_id: str, tier: Optional[str] = None, elo: float = 1200.0):
        """Add agent to league.

        Args:
            agent_id: Agent identifier
            tier: Starting tier (default: lowest)
            elo: Starting ELO
        """
        if tier is None:
            tier = self.tiers[-1]  # Lowest tier

        if tier not in self.tiers:
            raise ValueError(f"Unknown tier: {tier}")

        self.tier_members[tier].append(agent_id)
        self.all_stats[agent_id] = AgentStats(agent_id=agent_id, elo_rating=elo)

    def get_tier_standings(self, tier: str) -> List[Dict]:
        """Get standings for a specific tier.

        Args:
            tier: Tier name

        Returns:
            Sorted list of agent stats
        """
        members = self.tier_members[tier]
        stats = [self.all_stats[aid] for aid in members]
        sorted_stats = sorted(stats, key=lambda s: s.elo_rating, reverse=True)
        return [s.to_dict() for s in sorted_stats]

    def process_season_end(self):
        """Process end of season: promotions and relegations."""
        self.season += 1

        for i, tier in enumerate(self.tiers):
            members = self.tier_members[tier]
            if not members:
                continue

            # Sort by ELO
            sorted_members = sorted(
                members,
                key=lambda aid: self.all_stats[aid].elo_rating,
                reverse=True
            )

            # Promotions (to higher tier)
            if i > 0:  # Not top tier
                higher_tier = self.tiers[i - 1]
                promoted = sorted_members[:self.promotion_slots]
                for aid in promoted:
                    self.tier_members[tier].remove(aid)
                    self.tier_members[higher_tier].append(aid)

            # Relegations (to lower tier)
            if i < len(self.tiers) - 1:  # Not bottom tier
                lower_tier = self.tiers[i + 1]
                relegated = sorted_members[-self.relegation_slots:]
                for aid in relegated:
                    if aid in self.tier_members[tier]:  # May have been promoted
                        self.tier_members[tier].remove(aid)
                        self.tier_members[lower_tier].append(aid)

        # Record history
        self.season_history.append({
            "season": self.season - 1,
            "tiers": {
                tier: self.get_tier_standings(tier)
                for tier in self.tiers
            }
        })

    def to_dict(self) -> Dict:
        """Convert league state to dictionary."""
        return {
            "name": self.name,
            "season": self.season,
            "tiers": self.tiers,
            "tier_members": self.tier_members,
            "standings": {
                tier: self.get_tier_standings(tier)
                for tier in self.tiers
            },
            "season_history": self.season_history,
        }


def default_get_action(agent: Any, obs: Dict, env: Any) -> int:
    """Default action getter that works with RuleBasedAgent.

    Args:
        agent: Agent object
        obs: Observation dictionary
        env: Environment

    Returns:
        Action index
    """
    if hasattr(agent, 'get_action'):
        return agent.get_action(obs, env)
    else:
        # Random fallback
        mask = obs.get("action_mask", [True] * 4)
        valid = [i for i, v in enumerate(mask) if v]
        return np.random.choice(valid)


# Testing
if __name__ == "__main__":
    import poker_engine

    print("=== Tournament System Test ===\n")

    # Create environment
    env = poker_engine.PokerEnv(
        num_players=2,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    # Create tournament
    tournament = Tournament(
        name="Test Tournament",
        hands_per_match=100,
        num_rounds=1,
        seed=42,
    )

    # Add participants from different pools
    tag_agent = RuleBasedAgent(PersonalityType.TAG, player_id=0, seed=1)
    lag_agent = RuleBasedAgent(PersonalityType.LAG, player_id=1, seed=2)
    rock_agent = RuleBasedAgent(PersonalityType.ROCK, player_id=2, seed=3)
    fish_agent = RuleBasedAgent(PersonalityType.FISH, player_id=3, seed=4)

    tournament.add_participant("TAG_1", tag_agent, initial_elo=1200)
    tournament.add_participant("LAG_1", lag_agent, initial_elo=1200)
    tournament.add_participant("ROCK_1", rock_agent, initial_elo=1200)
    tournament.add_participant("FISH_1", fish_agent, initial_elo=1200)

    print(f"Tournament: {tournament.name}")
    print(f"Participants: {list(tournament.participants.keys())}")
    print(f"Hands per match: {tournament.hands_per_match}")

    # Run tournament
    print("\nRunning tournament...")
    stats = tournament.run(
        env,
        get_action_fn=default_get_action,
        num_players=2,
        verbose=True,
    )

    # Print standings
    print("\n=== Final Standings ===")
    for i, standing in enumerate(tournament.get_standings()):
        print(
            f"{i+1}. {standing['agent_id']:10} | "
            f"ELO: {standing['elo_rating']:.0f} | "
            f"W-L-D: {standing['wins']}-{standing['losses']}-{standing['draws']} | "
            f"BB/100: {standing['bb_per_100']:+.1f}"
        )

    # Save results
    tournament.save("/tmp/tournament_test.json")
    print(f"\nResults saved to /tmp/tournament_test.json")

    # Test League
    print("\n=== League System Test ===")
    league = League(
        name="Poker League",
        tiers=["Gold", "Silver", "Bronze"],
    )

    # Add agents to league
    for i in range(9):
        tier_idx = i // 3
        tier = league.tiers[tier_idx]
        league.add_agent(f"agent_{i}", tier=tier, elo=1200 + (2-tier_idx) * 100)

    print(f"League: {league.name}")
    print(f"Tiers: {league.tiers}")

    for tier in league.tiers:
        print(f"\n{tier} tier:")
        for standing in league.get_tier_standings(tier):
            print(f"  {standing['agent_id']}: ELO {standing['elo_rating']:.0f}")

    print("\n✓ All tests passed!")
