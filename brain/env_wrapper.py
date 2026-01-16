"""Gymnasium and PettingZoo compatible wrappers for the poker engine."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore

import poker_engine


class PokerGymEnv(gym.Env if gym else object):
    """Gymnasium-compatible poker environment.

    This wrapper provides a standard Gym interface for the poker engine.

    Action space:
        Discrete(4):
        - 0: Fold
        - 1: Check/Call
        - 2: Min Bet/Raise
        - 3: All-in

    Observation space:
        Dict containing game state information
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        num_players: int = 6,
        small_blind: int = 50,
        big_blind: int = 100,
        starting_stack: int = 10000,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the poker environment.

        Args:
            num_players: Number of players (2-9)
            small_blind: Small blind amount
            big_blind: Big blind amount
            starting_stack: Starting stack for each player
            seed: Random seed for reproducibility
            render_mode: Render mode ('ansi' or None)
        """
        super().__init__()

        self.num_players = num_players
        self.render_mode = render_mode

        self._env = poker_engine.PokerEnv(
            num_players=num_players,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            seed=seed,
        )

        # Action space: 0=fold, 1=check/call, 2=min raise, 3=all-in
        self.action_space = spaces.Discrete(4)

        # Observation space
        self.observation_space = spaces.Dict({
            "current_player": spaces.Discrete(num_players),
            "street": spaces.Discrete(5),
            "pot": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "current_bet": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "stacks": spaces.Box(low=0, high=np.inf, shape=(num_players,), dtype=np.float32),
            "bets": spaces.Box(low=0, high=np.inf, shape=(num_players,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(4),
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset the environment and start a new hand.

        Returns:
            Tuple of (observation, info)
        """
        # Note: seed is currently ignored as the underlying env was seeded at init
        obs = self._env.reset()
        return self._process_obs(obs), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take an action in the environment.

        Args:
            action: Action index (0-3)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, rewards, done, truncated, info = self._env.step(action)

        # Return reward for current player (before action)
        current_player = self._last_player if hasattr(self, "_last_player") else 0
        reward = float(rewards[current_player]) if done else 0.0

        self._last_player = obs.get("current_player", 0)

        return self._process_obs(obs), reward, done, truncated, dict(info)

    def _process_obs(self, obs: Dict) -> Dict:
        """Process raw observation into Gym-compatible format."""
        return {
            "current_player": obs["current_player"],
            "street": obs["street"],
            "pot": np.array([obs["pot"]], dtype=np.float32),
            "current_bet": np.array([obs["current_bet"]], dtype=np.float32),
            "stacks": np.array(
                [p["stack"] for p in obs["players"]], dtype=np.float32
            ),
            "bets": np.array(
                [p["bet"] for p in obs["players"]], dtype=np.float32
            ),
            "action_mask": np.array(obs["action_mask"], dtype=np.int8),
        }

    def get_hole_cards(self, player: int) -> List:
        """Get hole cards for a specific player."""
        return self._env.get_hole_cards(player)

    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode != "ansi":
            return None

        obs = self._env.get_observation()
        lines = [
            f"=== Hand #{self._env.hand_number} ===",
            f"Street: {['Preflop', 'Flop', 'Turn', 'River', 'Showdown'][obs['street']]}",
            f"Board: {' '.join(obs['board']) if obs['board'] else '(none)'}",
            f"Pot: {obs['pot']}",
            f"Current bet: {obs['current_bet']}",
            "",
        ]

        for i, p in enumerate(obs["players"]):
            status = ["Active", "Folded", "All-in", "Out"][p["status"]]
            marker = " <--" if i == obs["current_player"] else ""
            lines.append(f"Player {i}: {p['stack']} chips, bet {p['bet']} ({status}){marker}")

        return "\n".join(lines)


class PokerAECEnv:
    """PettingZoo AEC (Agent Environment Cycle) compatible environment.

    This provides a multi-agent environment where agents take turns
    in a cycle, suitable for training multi-agent RL systems.
    """

    def __init__(
        self,
        num_players: int = 6,
        small_blind: int = 50,
        big_blind: int = 100,
        starting_stack: int = 10000,
        seed: Optional[int] = None,
    ):
        """Initialize the AEC environment."""
        self.num_players = num_players

        self._env = poker_engine.PokerEnv(
            num_players=num_players,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
            seed=seed,
        )

        # Agent names
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents.copy()
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        self._obs: Optional[Dict] = None
        self._rewards = {agent: 0.0 for agent in self.agents}
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        self._infos: Dict[str, Dict] = {agent: {} for agent in self.agents}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> None:
        """Reset the environment."""
        self._obs = self._env.reset()
        self._rewards = {agent: 0.0 for agent in self.agents}
        self._terminations = {agent: False for agent in self.agents}
        self._truncations = {agent: False for agent in self.agents}
        self._infos = {agent: {} for agent in self.agents}

    def step(self, action: int) -> None:
        """Take an action for the current agent."""
        obs, rewards, done, truncated, info = self._env.step(action)
        self._obs = obs

        for i, agent in enumerate(self.agents):
            self._rewards[agent] = float(rewards[i])
            self._terminations[agent] = done
            self._truncations[agent] = truncated

    @property
    def agent_selection(self) -> str:
        """Get the current agent to act."""
        if self._obs is None:
            return self.agents[0]
        return f"player_{self._obs['current_player']}"

    def observe(self, agent: str) -> Dict:
        """Get observation for a specific agent."""
        if self._obs is None:
            return {}
        return self._obs

    def last(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """Get the last observation, reward, and flags for the current agent."""
        agent = self.agent_selection
        obs = self._obs if self._obs else {}
        return (
            obs,
            self._rewards.get(agent, 0.0),
            self._terminations.get(agent, False),
            self._truncations.get(agent, False),
            self._infos.get(agent, {}),
        )

    def action_space(self, agent: str) -> "spaces.Discrete":
        """Get action space for an agent."""
        return spaces.Discrete(4)

    def observation_space(self, agent: str) -> "spaces.Dict":
        """Get observation space for an agent."""
        return spaces.Dict({
            "street": spaces.Discrete(5),
            "pot": spaces.Box(low=0, high=np.inf, shape=(1,)),
            "action_mask": spaces.MultiBinary(4),
        })

    def get_hole_cards(self, agent: str) -> List:
        """Get hole cards for a specific agent."""
        player_idx = self.agent_name_mapping[agent]
        return self._env.get_hole_cards(player_idx)


def make_env(
    env_type: str = "gym",
    num_players: int = 6,
    **kwargs,
) -> Any:
    """Factory function to create poker environments.

    Args:
        env_type: "gym" for Gymnasium or "aec" for PettingZoo AEC
        num_players: Number of players
        **kwargs: Additional arguments passed to the environment

    Returns:
        Environment instance
    """
    if env_type == "gym":
        return PokerGymEnv(num_players=num_players, **kwargs)
    elif env_type == "aec":
        return PokerAECEnv(num_players=num_players, **kwargs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
