"""Proximal Policy Optimization (PPO) agent for poker.

PPO is a policy gradient method that uses a clipped surrogate objective
for stable training. This implementation supports:
- Self-play training
- Multiple parallel environments
- Generalized Advantage Estimation (GAE)
- Entropy regularization
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import time

import poker_engine
try:
    from .abstraction import FeatureExtractor, create_default_extractor
    from .networks import ActorCriticNetwork, create_actor_critic
except ImportError:
    from abstraction import FeatureExtractor, create_default_extractor
    from networks import ActorCriticNetwork, create_actor_critic


@dataclass
class Experience:
    """Single experience tuple."""
    features: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float
    done: bool
    value: float
    log_prob: float


class RolloutBuffer:
    """Buffer for storing rollout experiences."""

    def __init__(self):
        """Initialize empty buffer."""
        self.experiences: List[Experience] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(self, exp: Experience):
        """Add an experience to the buffer."""
        self.experiences.append(exp)

    def clear(self):
        """Clear the buffer."""
        self.experiences = []
        self.advantages = None
        self.returns = None

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for the state after the last experience
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        n = len(self.experiences)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value_used = last_value

        for t in reversed(range(n)):
            exp = self.experiences[t]

            if exp.done:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = last_value_used

            # TD error
            delta = exp.reward + gamma * next_value - exp.value

            # GAE
            last_gae = delta + gamma * gae_lambda * (1 - exp.done) * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = self.advantages[t] + exp.value

            last_value_used = exp.value

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """Get mini-batches for training.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle experiences

        Returns:
            List of batch dictionaries
        """
        n = len(self.experiences)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            batch = {
                "features": torch.tensor(
                    np.array([self.experiences[i].features for i in batch_indices]),
                    dtype=torch.float32,
                ),
                "actions": torch.tensor(
                    [self.experiences[i].action for i in batch_indices],
                    dtype=torch.long,
                ),
                "action_masks": torch.tensor(
                    np.array([self.experiences[i].action_mask for i in batch_indices]),
                    dtype=torch.bool,
                ),
                "old_log_probs": torch.tensor(
                    [self.experiences[i].log_prob for i in batch_indices],
                    dtype=torch.float32,
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_indices],
                    dtype=torch.float32,
                ),
                "returns": torch.tensor(
                    self.returns[batch_indices],
                    dtype=torch.float32,
                ),
            }
            batches.append(batch)

        return batches

    def __len__(self) -> int:
        return len(self.experiences)


class PPOAgent:
    """PPO agent for poker."""

    def __init__(
        self,
        num_players: int = 2,
        num_actions: int = 4,
        feature_dim: int = None,
        hidden_dims: List[int] = [256, 256],
        actor_dims: List[int] = [128],
        critic_dims: List[int] = [128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """Initialize PPO agent.

        Args:
            num_players: Number of players
            num_actions: Number of actions
            feature_dim: Input feature dimension
            hidden_dims: Shared network hidden dims
            actor_dims: Actor head dims
            critic_dims: Critic head dims
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Gradient clipping
            device: Device to use
        """
        self.num_players = num_players
        self.num_actions = num_actions
        self.device = torch.device(device)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Feature extraction
        self.feature_extractor = create_default_extractor(num_players)
        self.feature_dim = feature_dim or self.feature_extractor.feature_dim

        # Network
        self.network = ActorCriticNetwork(
            input_dim=self.feature_dim,
            num_actions=num_actions,
            shared_dims=hidden_dims,
            actor_dims=actor_dims,
            critic_dims=critic_dims,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Stats
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def get_action(
        self,
        features: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Get action for state.

        Args:
            features: State features
            action_mask: Valid action mask
            deterministic: If True, return best action

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

            action, log_prob, value = self.network.get_action(
                feat_tensor, mask_tensor, deterministic=deterministic
            )

            return (
                action.item(),
                log_prob.item(),
                value.item(),
            )

    def train_step(
        self,
        batch_size: int = 64,
        epochs: int = 4,
    ) -> Dict[str, float]:
        """Perform one training step on the buffer.

        Args:
            batch_size: Mini-batch size
            epochs: Number of epochs to train

        Returns:
            Dictionary of loss values
        """
        if len(self.buffer) == 0:
            return {}

        # Get mini-batches
        batches = self.buffer.get_batches(batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(epochs):
            for batch in batches:
                # Move to device
                features = batch["features"].to(self.device)
                actions = batch["actions"].to(self.device)
                action_masks = batch["action_masks"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy
                log_probs, values, entropy = self.network.evaluate_actions(
                    features, actions, action_masks
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def collect_rollout(
        self,
        env: Any,
        opponent: Optional["PPOAgent"] = None,
        num_steps: int = 2048,
        player_id: int = 0,
    ) -> Dict[str, float]:
        """Collect rollout experiences.

        Args:
            env: Poker environment
            opponent: Opponent agent (self if None for self-play)
            num_steps: Number of steps to collect
            player_id: Which player we control

        Returns:
            Statistics dictionary
        """
        if opponent is None:
            opponent = self

        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0

        steps_collected = 0

        while steps_collected < num_steps:
            current_player = obs["current_player"]

            # Extract features and mask
            features = self._extract_features(env, obs, current_player)
            action_mask = np.array(obs["action_mask"], dtype=np.float32)

            if current_player == player_id:
                # Our turn - collect experience
                action, log_prob, value = self.get_action(features, action_mask)

                # Take action
                new_obs, rewards, done, _, info = env.step(action)

                # Store experience (reward will be updated at end of episode)
                exp = Experience(
                    features=features,
                    action=action,
                    action_mask=action_mask,
                    reward=0.0,  # Will be set later
                    done=done,
                    value=value,
                    log_prob=log_prob,
                )
                self.buffer.add(exp)
                steps_collected += 1

            else:
                # Opponent's turn
                action, _, _ = opponent.get_action(features, action_mask)
                new_obs, rewards, done, _, info = env.step(action)

            episode_length += 1

            if done:
                # Set final reward for all our experiences this episode
                final_reward = rewards[player_id] / 100.0  # Normalize

                # Set reward for last experience
                if len(self.buffer) > 0:
                    self.buffer.experiences[-1] = Experience(
                        features=self.buffer.experiences[-1].features,
                        action=self.buffer.experiences[-1].action,
                        action_mask=self.buffer.experiences[-1].action_mask,
                        reward=final_reward,
                        done=True,
                        value=self.buffer.experiences[-1].value,
                        log_prob=self.buffer.experiences[-1].log_prob,
                    )

                episode_reward = rewards[player_id]
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Reset
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = new_obs

        # Compute advantages
        last_value = 0.0
        if not done:
            with torch.no_grad():
                feat_tensor = torch.tensor(
                    self._extract_features(env, obs, player_id),
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                _, last_value = self.network(feat_tensor)
                last_value = last_value.item()

        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        return {
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }

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

        return self.feature_extractor.extract_features(
            hole_cards=hole_cards_tuples,
            board=board,
            street=obs["street"],
            position=player,
            pot=obs["pot"],
            to_call=obs["current_bet"] - players_info[player]["bet"],
            stack=players_info[player]["stack"],
            active_players=active_players,
        )

    def save(self, path: str):
        """Save model to file."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "num_players": self.num_players,
                "num_actions": self.num_actions,
                "feature_dim": self.feature_dim,
            },
        }, path)

    def load(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def train_ppo(
    num_iterations: int = 1000,
    steps_per_iter: int = 2048,
    batch_size: int = 64,
    epochs: int = 4,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> PPOAgent:
    """Train PPO agent through self-play.

    Args:
        num_iterations: Number of training iterations
        steps_per_iter: Steps to collect per iteration
        batch_size: Training batch size
        epochs: Training epochs per iteration
        save_path: Path to save model
        verbose: Print progress

    Returns:
        Trained PPOAgent
    """
    # Create agent
    agent = PPOAgent(
        num_players=2,
        num_actions=4,
        hidden_dims=[256, 256],
        actor_dims=[128],
        critic_dims=[128],
        learning_rate=3e-4,
    )

    # Create environment
    env = poker_engine.PokerEnv(
        num_players=2,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    print("Starting PPO self-play training...")
    start_time = time.time()

    for iteration in range(num_iterations):
        # Collect rollouts (self-play)
        rollout_stats = agent.collect_rollout(
            env=env,
            opponent=agent,  # Self-play
            num_steps=steps_per_iter,
            player_id=iteration % 2,  # Alternate positions
        )

        # Train on collected data
        train_stats = agent.train_step(batch_size=batch_size, epochs=epochs)

        if verbose and (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Iteration {iteration + 1}/{num_iterations} | "
                f"Time: {elapsed:.1f}s | "
                f"Mean Reward: {rollout_stats['mean_reward']:.1f} | "
                f"Policy Loss: {train_stats.get('policy_loss', 0):.4f} | "
                f"Value Loss: {train_stats.get('value_loss', 0):.4f}"
            )

    if save_path:
        agent.save(save_path)
        print(f"Model saved to {save_path}")

    return agent


if __name__ == "__main__":
    agent = train_ppo(num_iterations=100, steps_per_iter=512, verbose=True)
    print("Training complete!")
