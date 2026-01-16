"""Deep Counterfactual Regret Minimization (Deep CFR) implementation.

This implements Single Deep CFR (SD-CFR) which is a neural network-based
version of CFR for large extensive-form games like poker.

Key concepts:
- Regret: How much better an action would have been vs the current strategy
- Strategy: Probability distribution over actions, proportional to positive regrets
- Counterfactual Value: Expected value when reaching a state for a specific player
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time

import poker_engine
try:
    from .abstraction import FeatureExtractor, ActionAbstraction, create_default_extractor
    from .networks import RegretNetwork, StrategyNetwork, create_regret_network
except ImportError:
    from abstraction import FeatureExtractor, ActionAbstraction, create_default_extractor
    from networks import RegretNetwork, StrategyNetwork, create_regret_network


@dataclass
class TraversalSample:
    """A sample collected during CFR traversal."""
    features: np.ndarray  # State features
    action_mask: np.ndarray  # Valid actions
    regrets: np.ndarray  # Computed regrets
    iteration: int  # CFR iteration number
    player: int  # Which player this sample is for


@dataclass
class StrategySample:
    """A sample for training the average strategy network."""
    features: np.ndarray
    action_mask: np.ndarray
    strategy: np.ndarray  # Action probabilities
    reach_prob: float  # Probability of reaching this state
    iteration: int


class ReservoirBuffer:
    """Reservoir sampling buffer for storing training samples.

    Maintains a fixed-size buffer with uniform sampling over all seen samples.
    """

    def __init__(self, capacity: int):
        """Initialize buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: List[Any] = []
        self.count = 0

    def add(self, sample: Any):
        """Add a sample to the buffer using reservoir sampling."""
        self.count += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            # Reservoir sampling
            idx = random.randint(0, self.count - 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

    def sample(self, batch_size: int) -> List[Any]:
        """Sample a batch from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class CFRDataset(Dataset):
    """PyTorch dataset for CFR training samples."""

    def __init__(self, samples: List[TraversalSample]):
        """Initialize dataset."""
        self.features = torch.tensor(
            np.array([s.features for s in samples]), dtype=torch.float32
        )
        self.action_masks = torch.tensor(
            np.array([s.action_mask for s in samples]), dtype=torch.bool
        )
        self.regrets = torch.tensor(
            np.array([s.regrets for s in samples]), dtype=torch.float32
        )
        self.iterations = torch.tensor(
            [s.iteration for s in samples], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.features[idx],
            self.action_masks[idx],
            self.regrets[idx],
            self.iterations[idx],
        )


class DeepCFR:
    """Deep CFR trainer for poker.

    This implementation uses:
    - External sampling: Only traverse for one player at a time
    - Linear CFR weighting: Recent iterations weighted more heavily
    - Reservoir sampling: Maintain fixed-size training buffers
    """

    def __init__(
        self,
        num_players: int = 2,
        num_actions: int = 4,
        feature_dim: int = None,
        hidden_dims: List[int] = [256, 256, 128],
        buffer_size: int = 100000,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        """Initialize Deep CFR.

        Args:
            num_players: Number of players (typically 2 for heads-up)
            num_actions: Number of abstract actions
            feature_dim: Input feature dimension (auto-computed if None)
            hidden_dims: Hidden layer dimensions for networks
            buffer_size: Size of reservoir buffer
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            device: Device to run on ("cpu" or "cuda")
        """
        self.num_players = num_players
        self.num_actions = num_actions
        self.device = torch.device(device)

        # Feature extraction
        self.feature_extractor = create_default_extractor(num_players)
        self.feature_dim = feature_dim or self.feature_extractor.feature_dim

        # Action abstraction
        self.action_abstraction = ActionAbstraction()

        # Networks - one regret network per player
        self.regret_networks = nn.ModuleList([
            RegretNetwork(
                input_dim=self.feature_dim,
                num_actions=num_actions,
                hidden_dims=hidden_dims,
            ).to(self.device)
            for _ in range(num_players)
        ])

        # Average strategy network (shared)
        self.strategy_network = StrategyNetwork(
            input_dim=self.feature_dim,
            num_actions=num_actions,
            hidden_dims=hidden_dims,
        ).to(self.device)

        # Optimizers
        self.regret_optimizers = [
            optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.regret_networks
        ]
        self.strategy_optimizer = optim.Adam(
            self.strategy_network.parameters(), lr=learning_rate
        )

        # Training buffers
        self.regret_buffers = [
            ReservoirBuffer(buffer_size) for _ in range(num_players)
        ]
        self.strategy_buffer = ReservoirBuffer(buffer_size)

        # Training config
        self.batch_size = batch_size
        self.iteration = 0

        # Stats
        self.stats = defaultdict(list)

    def train(
        self,
        num_iterations: int,
        traversals_per_iter: int = 1000,
        train_every: int = 100,
        verbose: bool = True,
    ):
        """Run Deep CFR training.

        Args:
            num_iterations: Number of CFR iterations
            traversals_per_iter: Game traversals per iteration
            train_every: Train networks every N traversals
            verbose: Print progress
        """
        start_time = time.time()

        for iteration in range(num_iterations):
            self.iteration = iteration + 1

            # Collect samples through game traversals
            for t in range(traversals_per_iter):
                # Alternate which player we're computing regrets for
                traverse_player = t % self.num_players

                # Run one game traversal
                self._traverse_game(traverse_player)

                # Train networks periodically
                if (t + 1) % train_every == 0:
                    self._train_networks()

            # End of iteration training
            self._train_networks()

            if verbose and (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_regret = np.mean(self.stats.get("regret_loss", [0])[-100:])
                print(
                    f"Iteration {iteration + 1}/{num_iterations} | "
                    f"Time: {elapsed:.1f}s | "
                    f"Avg Regret Loss: {avg_regret:.4f} | "
                    f"Buffer sizes: {[len(b) for b in self.regret_buffers]}"
                )

    def _traverse_game(self, traverse_player: int):
        """Traverse one game, collecting samples for traverse_player.

        Uses external sampling: opponent actions are sampled from current strategy.
        """
        # Create game environment
        env = poker_engine.PokerEnv(
            num_players=self.num_players,
            small_blind=50,
            big_blind=100,
            starting_stack=10000,
        )
        obs = env.reset()

        # Traverse until terminal
        reach_prob = 1.0

        while not obs["is_terminal"]:
            current_player = obs["current_player"]
            features = self._extract_features(env, obs, current_player)
            action_mask = np.array(obs["action_mask"], dtype=np.float32)

            # Get current strategy from regret network
            with torch.no_grad():
                feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
                strategy = self.regret_networks[current_player].get_strategy(
                    feat_tensor, mask_tensor
                ).squeeze(0).cpu().numpy()

            if current_player == traverse_player:
                # For traverse player, compute counterfactual values for all actions
                action_values = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    if action_mask[action]:
                        # Clone state and take action
                        action_value = self._compute_action_value(
                            env, obs, action, traverse_player
                        )
                        action_values[action] = action_value

                # Compute regrets
                state_value = np.sum(strategy * action_values)
                regrets = action_values - state_value

                # Mask invalid actions
                regrets = regrets * action_mask

                # Add sample to buffer
                sample = TraversalSample(
                    features=features,
                    action_mask=action_mask,
                    regrets=regrets,
                    iteration=self.iteration,
                    player=traverse_player,
                )
                self.regret_buffers[traverse_player].add(sample)

                # Add strategy sample
                strategy_sample = StrategySample(
                    features=features,
                    action_mask=action_mask,
                    strategy=strategy,
                    reach_prob=reach_prob,
                    iteration=self.iteration,
                )
                self.strategy_buffer.add(strategy_sample)

                # Sample action according to strategy
                action = np.random.choice(self.num_actions, p=strategy)
                reach_prob *= strategy[action]

            else:
                # For opponent, just sample from strategy
                action = np.random.choice(self.num_actions, p=strategy)

            # Take action
            obs, rewards, done, _, info = env.step(int(action))

    def _compute_action_value(
        self,
        env: Any,
        obs: Dict,
        action: int,
        player: int,
    ) -> float:
        """Compute the value of taking an action by simulation.

        For simplicity, we run a random rollout to terminal and return
        the player's reward. In full Deep CFR, this would be a recursive
        traversal with proper value computation.
        """
        # Create a new env from current state (simplified - just continue game)
        # In practice, you'd want proper state cloning

        # Take the action
        new_obs, rewards, done, _, _ = env.step(int(action))

        if done:
            return rewards[player] / 100.0  # Normalize rewards

        # Rollout with current strategies until terminal
        while not new_obs["is_terminal"]:
            current_player = new_obs["current_player"]
            action_mask = np.array(new_obs["action_mask"], dtype=np.float32)

            # Sample from strategy
            features = self._extract_features(env, new_obs, current_player)

            with torch.no_grad():
                feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
                strategy = self.regret_networks[current_player].get_strategy(
                    feat_tensor, mask_tensor
                ).squeeze(0).cpu().numpy()

            next_action = np.random.choice(self.num_actions, p=strategy)
            new_obs, rewards, done, _, _ = env.step(int(next_action))

        return rewards[player] / 100.0  # Normalize

    def _extract_features(self, env: Any, obs: Dict, player: int) -> np.ndarray:
        """Extract features from game state."""
        # Get hole cards
        hole_cards = env.get_hole_cards(player)
        hole_cards_tuples = [(c.rank, c.suit) for c in hole_cards]

        # Get board cards
        board = []
        if obs["board"]:
            for card_str in obs["board"]:
                rank_char = card_str[0]
                suit_char = card_str[1]

                rank_map = {
                    "2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6,
                    "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12
                }
                suit_map = {"c": 0, "d": 1, "h": 2, "s": 3}

                board.append((rank_map[rank_char], suit_map[suit_char]))

        # Get other info
        players_info = obs["players"]
        active_players = [p["status"] == 0 or p["status"] == 2 for p in players_info]

        # Compute position
        position = player

        return self.feature_extractor.extract_features(
            hole_cards=hole_cards_tuples,
            board=board,
            street=obs["street"],
            position=position,
            pot=obs["pot"],
            to_call=obs["current_bet"] - players_info[player]["bet"],
            stack=players_info[player]["stack"],
            active_players=active_players,
        )

    def _train_networks(self):
        """Train regret and strategy networks on collected samples."""
        # Train regret networks
        for player in range(self.num_players):
            if len(self.regret_buffers[player]) < self.batch_size:
                continue

            samples = self.regret_buffers[player].sample(self.batch_size)
            loss = self._train_regret_network(player, samples)
            self.stats["regret_loss"].append(loss)

        # Train strategy network
        if len(self.strategy_buffer) >= self.batch_size:
            samples = self.strategy_buffer.sample(self.batch_size)
            loss = self._train_strategy_network(samples)
            self.stats["strategy_loss"].append(loss)

    def _train_regret_network(
        self,
        player: int,
        samples: List[TraversalSample],
    ) -> float:
        """Train regret network on samples."""
        network = self.regret_networks[player]
        optimizer = self.regret_optimizers[player]

        # Prepare batch
        features = torch.tensor(
            np.array([s.features for s in samples]), dtype=torch.float32
        ).to(self.device)
        targets = torch.tensor(
            np.array([s.regrets for s in samples]), dtype=torch.float32
        ).to(self.device)

        # Linear CFR weighting: weight by iteration number
        weights = torch.tensor(
            [s.iteration for s in samples], dtype=torch.float32
        ).to(self.device)
        weights = weights / weights.sum()

        # Forward pass
        optimizer.zero_grad()
        predictions = network(features)

        # MSE loss with iteration weighting
        loss = (weights.unsqueeze(1) * (predictions - targets) ** 2).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()

    def _train_strategy_network(self, samples: List[StrategySample]) -> float:
        """Train average strategy network on samples."""
        # Prepare batch
        features = torch.tensor(
            np.array([s.features for s in samples]), dtype=torch.float32
        ).to(self.device)
        action_masks = torch.tensor(
            np.array([s.action_mask for s in samples]), dtype=torch.bool
        ).to(self.device)
        targets = torch.tensor(
            np.array([s.strategy for s in samples]), dtype=torch.float32
        ).to(self.device)

        # Weighting by reach probability and iteration
        weights = torch.tensor(
            [s.reach_prob * s.iteration for s in samples], dtype=torch.float32
        ).to(self.device)
        weights = weights / weights.sum()

        # Forward pass
        self.strategy_optimizer.zero_grad()
        predictions = self.strategy_network(features, action_masks)

        # Cross-entropy loss
        loss = -(weights.unsqueeze(1) * targets * torch.log(predictions + 1e-8)).sum()

        # Backward pass
        loss.backward()
        self.strategy_optimizer.step()

        return loss.item()

    def get_strategy(
        self,
        features: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Get the current strategy for a state.

        Args:
            features: State features
            action_mask: Valid action mask

        Returns:
            Strategy (probability distribution over actions)
        """
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

            strategy = self.strategy_network(feat_tensor, mask_tensor)

            return strategy.squeeze(0).cpu().numpy()

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            "regret_networks": [net.state_dict() for net in self.regret_networks],
            "strategy_network": self.strategy_network.state_dict(),
            "iteration": self.iteration,
            "config": {
                "num_players": self.num_players,
                "num_actions": self.num_actions,
                "feature_dim": self.feature_dim,
            },
        }, path)

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)

        for i, state_dict in enumerate(checkpoint["regret_networks"]):
            self.regret_networks[i].load_state_dict(state_dict)

        self.strategy_network.load_state_dict(checkpoint["strategy_network"])
        self.iteration = checkpoint["iteration"]


class DeepCFRAgent:
    """Agent that uses trained Deep CFR model for playing poker."""

    def __init__(self, model: DeepCFR, player_id: int = 0):
        """Initialize agent.

        Args:
            model: Trained Deep CFR model
            player_id: Which player this agent controls
        """
        self.model = model
        self.player_id = player_id

    def get_action(
        self,
        env: Any,
        obs: Dict,
        deterministic: bool = False,
    ) -> int:
        """Get action for current state.

        Args:
            env: Poker environment
            obs: Current observation
            deterministic: If True, return best action

        Returns:
            Action index
        """
        features = self.model._extract_features(env, obs, self.player_id)
        action_mask = np.array(obs["action_mask"], dtype=np.float32)

        strategy = self.model.get_strategy(features, action_mask)

        if deterministic:
            return int(np.argmax(strategy))
        else:
            return int(np.random.choice(self.model.num_actions, p=strategy))


def train_deep_cfr(
    num_iterations: int = 100,
    traversals_per_iter: int = 1000,
    save_path: Optional[str] = None,
) -> DeepCFR:
    """Convenience function to train Deep CFR.

    Args:
        num_iterations: Number of CFR iterations
        traversals_per_iter: Game traversals per iteration
        save_path: Optional path to save model

    Returns:
        Trained DeepCFR model
    """
    model = DeepCFR(
        num_players=2,
        num_actions=4,
        hidden_dims=[256, 256, 128],
        buffer_size=100000,
        batch_size=1024,
        learning_rate=1e-3,
    )

    print("Starting Deep CFR training...")
    model.train(
        num_iterations=num_iterations,
        traversals_per_iter=traversals_per_iter,
        verbose=True,
    )

    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model


if __name__ == "__main__":
    # Quick test
    model = train_deep_cfr(num_iterations=10, traversals_per_iter=100)
    print("Training complete!")
