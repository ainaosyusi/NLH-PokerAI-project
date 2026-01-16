"""Distributed training support using Ray.

This module provides:
1. Ray-based parallel training workers
2. Distributed evaluation across multiple tables
3. Centralized parameter server pattern
4. TensorBoard/WandB logging integration
5. Graceful fallback to single-process if Ray unavailable
"""

from typing import List, Dict, Optional, Tuple, Callable, Any, Union
import numpy as np
import os
import time
from dataclasses import dataclass, field
from collections import deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import io

# Try to import Ray
try:
    import ray
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    RayQueue = None

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import logging backends
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from .personality_agents import RuleBasedAgent, AgentPool, create_pool
    from .multi_player_features import MultiPlayerFeatureExtractor
    from .population_training import Population, HyperParameters
    from .networks import ActorCriticNetwork
    from .abstraction import create_default_extractor
except ImportError:
    from personality_agents import RuleBasedAgent, AgentPool, create_pool
    from multi_player_features import MultiPlayerFeatureExtractor
    from population_training import Population, HyperParameters
    from networks import ActorCriticNetwork
    from abstraction import create_default_extractor


# Constants
NUM_ACTIONS = 11  # Pot-geometric action space


@dataclass
class RolloutData:
    """Data collected from a single rollout."""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    action_masks: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    player_ids: List[int] = field(default_factory=list)
    total_profit: float = 0.0
    num_hands: int = 0
    worker_id: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "states": [s.tolist() if isinstance(s, np.ndarray) else s for s in self.states],
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "action_masks": [m.tolist() if isinstance(m, np.ndarray) else m for m in self.action_masks],
            "values": self.values,
            "log_probs": self.log_probs,
            "player_ids": self.player_ids,
            "total_profit": self.total_profit,
            "num_hands": self.num_hands,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RolloutData":
        """Create from dictionary."""
        return cls(
            states=[np.array(s, dtype=np.float32) for s in d["states"]],
            actions=d["actions"],
            rewards=d["rewards"],
            dones=d["dones"],
            action_masks=[np.array(m, dtype=np.float32) for m in d["action_masks"]],
            values=d["values"],
            log_probs=d["log_probs"],
            player_ids=d.get("player_ids", [0] * len(d["actions"])),
            total_profit=d["total_profit"],
            num_hands=d["num_hands"],
            worker_id=d.get("worker_id", 0),
        )


@dataclass
class WorkerConfig:
    """Configuration for a training worker."""
    worker_id: int
    num_players: int
    hands_per_rollout: int
    opponent_pool_name: str
    num_actions: int = NUM_ACTIONS
    small_blind: int = 50
    big_blind: int = 100
    starting_stack: int = 10000
    seed: Optional[int] = None


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    num_workers: int = 4
    num_players: int = 6
    hands_per_rollout: int = 100
    opponent_pool_name: str = "mixed_pool"

    # Game settings
    small_blind: int = 50
    big_blind: int = 100
    starting_stack: int = 10000

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 256
    epochs_per_update: int = 4

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_dims: List[int] = field(default_factory=lambda: [128])
    critic_dims: List[int] = field(default_factory=lambda: [128])

    # Logging
    log_dir: str = "logs"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "poker-ai"
    wandb_entity: Optional[str] = None

    # Training
    total_iterations: int = 1000
    eval_every: int = 50
    save_every: int = 100
    save_dir: str = "models"

    seed: Optional[int] = None


class TrainingLogger:
    """Unified logging interface for TensorBoard and WandB."""

    def __init__(self, config: TrainingConfig):
        """Initialize loggers based on configuration."""
        self.config = config
        self.step = 0
        self.tb_writer = None
        self.wandb_run = None

        # Setup TensorBoard
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            os.makedirs(config.log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(config.log_dir)
            print(f"TensorBoard logging to: {config.log_dir}")

        # Setup WandB
        if config.use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
            )
            print(f"WandB logging initialized: {config.wandb_project}")

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        step = step if step is not None else self.step

        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)

        if self.wandb_run:
            wandb.log({name: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values."""
        step = step if step is not None else self.step

        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)

        if self.wandb_run:
            wandb.log(metrics, step=step)

    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram."""
        step = step if step is not None else self.step

        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)

        if self.wandb_run:
            wandb.log({name: wandb.Histogram(values)}, step=step)

    def increment_step(self):
        """Increment the global step counter."""
        self.step += 1

    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            wandb.finish()


class LocalWorker:
    """Single-process worker for rollout collection."""

    def __init__(self, config: WorkerConfig):
        """Initialize local worker."""
        self.config = config
        self.worker_id = config.worker_id
        self.rng = np.random.RandomState(config.seed)
        self.num_actions = config.num_actions

        # Import poker_engine here to avoid issues with Ray serialization
        import poker_engine
        self.env = poker_engine.PokerEnv(
            num_players=config.num_players,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            starting_stack=config.starting_stack,
        )

        self.opponent_pool = create_pool(
            config.opponent_pool_name,
            pool_size=20,
            seed=config.seed,
        )

        self.feature_extractor = MultiPlayerFeatureExtractor(
            max_players=config.num_players,
            use_opponent_modeling=False,
        )

        # Local network for inference (will be updated with weights from central learner)
        self.network = None
        self.device = torch.device("cpu")

    def set_weights(self, weights: bytes):
        """Set network weights from serialized bytes."""
        if not TORCH_AVAILABLE:
            return

        # Deserialize weights
        buffer = io.BytesIO(weights)
        state_dict = torch.load(buffer, map_location=self.device, weights_only=False)

        # Create network if not exists
        if self.network is None:
            # Find the input dimension from first layer weights
            # The key could be "shared.layers.0.weight" or "shared.output_layer.weight"
            feature_dim = None
            for key in state_dict.keys():
                if "shared" in key and "weight" in key:
                    weight_shape = state_dict[key].shape
                    # First linear layer's weight has shape (out_features, in_features)
                    # We want the in_features of the first layer
                    if len(weight_shape) == 2:
                        feature_dim = weight_shape[1]
                        break

            if feature_dim is None:
                # Fallback: use feature extractor to get feature dim
                feature_dim = self.feature_extractor.feature_dim

            self.network = ActorCriticNetwork(
                input_dim=feature_dim,
                num_actions=self.num_actions,
                shared_dims=[256, 256],
                actor_dims=[128],
                critic_dims=[128],
            ).to(self.device)

        self.network.load_state_dict(state_dict)
        self.network.eval()

    def get_action_from_network(
        self,
        features: np.ndarray,
        action_mask: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Get action using neural network."""
        if self.network is None:
            # Fallback to random if no network
            valid = np.where(action_mask > 0.5)[0]
            action = self.rng.choice(valid)
            return action, 0.0, 0.0

        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

            action, log_prob, value = self.network.get_action(
                feat_tensor, mask_tensor, deterministic=False
            )

            return action.item(), log_prob.item(), value.item()

    def collect_rollout(
        self,
        player_id: int = 0,
        use_network: bool = True,
    ) -> RolloutData:
        """Collect a rollout of experience."""
        rollout = RolloutData(worker_id=self.worker_id)

        # Get opponents for this rollout
        opponents = self.opponent_pool.sample_agents(self.config.num_players - 1)
        for i, opp in enumerate(opponents):
            opp.player_id = i + 1 if i >= player_id else i

        for _ in range(self.config.hands_per_rollout):
            obs = self.env.reset()
            hand_states = []
            hand_actions = []
            hand_values = []
            hand_log_probs = []
            hand_masks = []
            hand_player_ids = []

            while not obs["is_terminal"]:
                current = obs["current_player"]
                # Get action mask (11 actions for pot-geometric sizing)
                mask = np.array(obs.get("action_mask", [True] * self.num_actions), dtype=np.float32)

                if current == player_id:
                    # Learning agent's turn
                    features = self.feature_extractor.extract_features(
                        self.env, obs, current
                    )

                    if use_network:
                        action, log_prob, value = self.get_action_from_network(features, mask)
                    else:
                        valid = np.where(mask > 0.5)[0]
                        action = self.rng.choice(valid)
                        log_prob, value = 0.0, 0.0

                    hand_states.append(features)
                    hand_actions.append(action)
                    hand_values.append(value)
                    hand_log_probs.append(log_prob)
                    hand_masks.append(mask)
                    hand_player_ids.append(current)
                else:
                    # Opponent's turn - use legacy action mask for rule-based agents
                    legacy_mask = np.array(obs.get("legacy_action_mask", [True] * 4), dtype=np.float32)
                    opp_idx = current - 1 if current > player_id else current
                    if opp_idx < len(opponents):
                        action = opponents[opp_idx].get_action(obs, self.env)
                    else:
                        valid = [i for i, v in enumerate(legacy_mask) if v]
                        action = self.rng.choice(valid) if valid else 1  # Default to call

                obs, env_rewards, done, _, _ = self.env.step(action)

            # Process end of hand
            if done and hand_states:
                final_reward = env_rewards[player_id]
                rollout.total_profit += final_reward
                rollout.num_hands += 1

                # Assign rewards (only final step gets reward)
                for i in range(len(hand_states)):
                    rollout.states.append(hand_states[i])
                    rollout.actions.append(hand_actions[i])
                    rollout.values.append(hand_values[i])
                    rollout.log_probs.append(hand_log_probs[i])
                    rollout.action_masks.append(hand_masks[i])
                    rollout.player_ids.append(hand_player_ids[i])

                    if i == len(hand_states) - 1:
                        rollout.rewards.append(final_reward / 100.0)  # Normalize
                        rollout.dones.append(True)
                    else:
                        rollout.rewards.append(0.0)
                        rollout.dones.append(False)

        return rollout

    def evaluate(
        self,
        num_hands: int,
        player_id: int = 0,
    ) -> Tuple[float, int]:
        """Evaluate agent performance."""
        opponents = self.opponent_pool.sample_agents(self.config.num_players - 1)
        for i, opp in enumerate(opponents):
            opp.player_id = i + 1 if i >= player_id else i

        total_profit = 0.0

        for _ in range(num_hands):
            obs = self.env.reset()

            while not obs["is_terminal"]:
                current = obs["current_player"]
                mask = np.array(obs.get("action_mask", [True] * self.num_actions), dtype=np.float32)

                if current == player_id:
                    features = self.feature_extractor.extract_features(
                        self.env, obs, current
                    )
                    action, _, _ = self.get_action_from_network(features, mask)
                else:
                    legacy_mask = np.array(obs.get("legacy_action_mask", [True] * 4), dtype=np.float32)
                    opp_idx = current - 1 if current > player_id else current
                    if opp_idx < len(opponents):
                        action = opponents[opp_idx].get_action(obs, self.env)
                    else:
                        valid = [i for i, v in enumerate(legacy_mask) if v]
                        action = self.rng.choice(valid) if valid else 1

                obs, rewards, done, _, _ = self.env.step(action)

            if done:
                total_profit += rewards[player_id]

        return total_profit, num_hands


# Ray remote worker (only defined if Ray is available)
if RAY_AVAILABLE:
    @ray.remote
    class RayWorker:
        """Ray remote worker for distributed training."""

        def __init__(self, config: WorkerConfig):
            """Initialize Ray worker."""
            self.local_worker = LocalWorker(config)
            self.config = config

        def set_weights(self, weights: bytes):
            """Set network weights."""
            self.local_worker.set_weights(weights)

        def collect_rollout(
            self,
            player_id: int = 0,
            use_network: bool = True,
        ) -> Dict:
            """Collect rollout with current network weights."""
            rollout = self.local_worker.collect_rollout(player_id, use_network)
            return rollout.to_dict()

        def evaluate(
            self,
            num_hands: int,
            player_id: int = 0,
        ) -> Tuple[float, int]:
            """Evaluate with current network weights."""
            return self.local_worker.evaluate(num_hands, player_id)

        def get_worker_id(self) -> int:
            """Get worker ID."""
            return self.config.worker_id


class RayPPOTrainer:
    """Distributed PPO trainer using Ray."""

    def __init__(self, config: TrainingConfig):
        """Initialize distributed trainer."""
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Initialize Ray if needed
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                logging_level=40,  # Warning level
                num_cpus=config.num_workers + 1,  # Workers + learner
            )

        self.use_ray = RAY_AVAILABLE

        # Create workers
        self.workers = []
        self._create_workers()

        # Create central learner network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_network()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # Logger
        self.logger = TrainingLogger(config)

        # Statistics
        self.total_steps = 0
        self.total_hands = 0
        self.episode_rewards = deque(maxlen=100)

        print(f"RayPPOTrainer initialized:")
        print(f"  Workers: {config.num_workers}")
        print(f"  Using Ray: {self.use_ray}")
        print(f"  Device: {self.device}")
        print(f"  Actions: {NUM_ACTIONS} (pot-geometric)")

    def _create_workers(self):
        """Create worker instances."""
        for i in range(self.config.num_workers):
            worker_config = WorkerConfig(
                worker_id=i,
                num_players=self.config.num_players,
                hands_per_rollout=self.config.hands_per_rollout,
                opponent_pool_name=self.config.opponent_pool_name,
                num_actions=NUM_ACTIONS,
                small_blind=self.config.small_blind,
                big_blind=self.config.big_blind,
                starting_stack=self.config.starting_stack,
                seed=self.config.seed + i * 1000 if self.config.seed else None,
            )

            if self.use_ray:
                worker = RayWorker.remote(worker_config)
            else:
                worker = LocalWorker(worker_config)

            self.workers.append(worker)

        print(f"Created {len(self.workers)} {'Ray' if self.use_ray else 'local'} workers")

    def _create_network(self):
        """Create the central learner network."""
        # Create a dummy feature extractor to get feature dim
        extractor = MultiPlayerFeatureExtractor(
            max_players=self.config.num_players,
            use_opponent_modeling=False,
        )
        feature_dim = extractor.feature_dim

        self.network = ActorCriticNetwork(
            input_dim=feature_dim,
            num_actions=NUM_ACTIONS,
            shared_dims=self.config.hidden_dims,
            actor_dims=self.config.actor_dims,
            critic_dims=self.config.critic_dims,
        ).to(self.device)

        print(f"Created network with feature_dim={feature_dim}, num_actions={NUM_ACTIONS}")

    def _get_weights_bytes(self) -> bytes:
        """Serialize network weights to bytes."""
        buffer = io.BytesIO()
        torch.save(self.network.state_dict(), buffer)
        return buffer.getvalue()

    def _broadcast_weights(self):
        """Send current weights to all workers."""
        weights = self._get_weights_bytes()

        if self.use_ray:
            # Put weights in Ray object store once
            weights_ref = ray.put(weights)
            # Send to all workers
            ray.get([w.set_weights.remote(weights_ref) for w in self.workers])
        else:
            # Send directly to local workers
            for w in self.workers:
                w.set_weights(weights)

    def _collect_rollouts(self, player_id: int = 0) -> List[RolloutData]:
        """Collect rollouts from all workers in parallel."""
        if self.use_ray:
            # Ray parallel collection
            futures = [
                w.collect_rollout.remote(player_id, True)
                for w in self.workers
            ]
            results = ray.get(futures)
            rollouts = [RolloutData.from_dict(r) for r in results]
        else:
            # ThreadPool parallel collection for local workers
            rollouts = []
            with ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                futures = [
                    executor.submit(w.collect_rollout, player_id, True)
                    for w in self.workers
                ]
                for future in as_completed(futures):
                    rollouts.append(future.result())

        return rollouts

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = values[t + 1] if t + 1 < n else last_value

            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae

            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _train_step(self, rollouts: List[RolloutData]) -> Dict[str, float]:
        """Perform one training step on collected rollouts."""
        # Merge rollouts
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_masks = []
        all_values = []
        all_log_probs = []

        for r in rollouts:
            all_states.extend(r.states)
            all_actions.extend(r.actions)
            all_rewards.extend(r.rewards)
            all_dones.extend(r.dones)
            all_masks.extend(r.action_masks)
            all_values.extend(r.values)
            all_log_probs.extend(r.log_probs)

        if len(all_states) == 0:
            return {}

        # Convert to numpy arrays
        states = np.array(all_states, dtype=np.float32)
        actions = np.array(all_actions, dtype=np.int64)
        rewards = np.array(all_rewards, dtype=np.float32)
        dones = np.array(all_dones, dtype=np.float32)
        masks = np.array(all_masks, dtype=np.float32)
        values = np.array(all_values, dtype=np.float32)
        old_log_probs = np.array(all_log_probs, dtype=np.float32)

        # Compute advantages
        advantages, returns = self._compute_gae(rewards, values, dones)

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).to(self.device)
        masks_t = torch.tensor(masks, dtype=torch.bool).to(self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Training loop
        n = len(states)
        batch_size = self.config.batch_size
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.epochs_per_update):
            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]

                batch_states = states_t[batch_idx]
                batch_actions = actions_t[batch_idx]
                batch_masks = masks_t[batch_idx]
                batch_old_log_probs = old_log_probs_t[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                # Get current policy
                log_probs, values_pred, entropy = self.network.evaluate_actions(
                    batch_states, batch_actions, batch_masks
                )

                # Policy loss (PPO clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values_pred - batch_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "num_updates": num_updates,
        }

    def _evaluate(self, num_hands_per_worker: int = 100) -> Dict[str, float]:
        """Evaluate current policy."""
        if self.use_ray:
            futures = [
                w.evaluate.remote(num_hands_per_worker, i % self.config.num_players)
                for i, w in enumerate(self.workers)
            ]
            results = ray.get(futures)
        else:
            results = []
            for i, w in enumerate(self.workers):
                results.append(w.evaluate(num_hands_per_worker, i % self.config.num_players))

        total_profit = sum(r[0] for r in results)
        total_hands = sum(r[1] for r in results)

        bb_per_100 = (total_profit / total_hands) / self.config.big_blind * 100 if total_hands > 0 else 0

        return {
            "eval/total_profit": total_profit,
            "eval/total_hands": total_hands,
            "eval/bb_per_100": bb_per_100,
        }

    def train(self):
        """Run the full training loop."""
        print(f"\nStarting distributed training for {self.config.total_iterations} iterations...")
        start_time = time.time()

        for iteration in range(self.config.total_iterations):
            iter_start = time.time()

            # Broadcast current weights to workers
            self._broadcast_weights()

            # Collect rollouts in parallel
            player_id = iteration % self.config.num_players
            rollouts = self._collect_rollouts(player_id)

            # Calculate rollout statistics
            total_hands = sum(r.num_hands for r in rollouts)
            total_profit = sum(r.total_profit for r in rollouts)
            total_steps = sum(len(r.states) for r in rollouts)

            self.total_hands += total_hands
            self.total_steps += total_steps

            if total_hands > 0:
                avg_reward = total_profit / total_hands
                self.episode_rewards.append(avg_reward)

            # Train on collected data
            train_stats = self._train_step(rollouts)

            iter_time = time.time() - iter_start

            # Log metrics
            metrics = {
                "train/iteration": iteration + 1,
                "train/total_hands": self.total_hands,
                "train/total_steps": self.total_steps,
                "train/hands_this_iter": total_hands,
                "train/steps_this_iter": total_steps,
                "train/mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
                "train/iter_time": iter_time,
                **{f"train/{k}": v for k, v in train_stats.items()},
            }
            self.logger.log_scalars(metrics, step=iteration)
            self.logger.increment_step()

            # Print progress
            if (iteration + 1) % 10 == 0:
                elapsed = time.time() - start_time
                hands_per_sec = self.total_hands / elapsed if elapsed > 0 else 0
                print(
                    f"Iter {iteration + 1}/{self.config.total_iterations} | "
                    f"Hands: {self.total_hands:,} ({hands_per_sec:.0f}/s) | "
                    f"Reward: {np.mean(self.episode_rewards):.1f} | "
                    f"Policy Loss: {train_stats.get('policy_loss', 0):.4f} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Evaluation
            if (iteration + 1) % self.config.eval_every == 0:
                eval_stats = self._evaluate(100)
                self.logger.log_scalars(eval_stats, step=iteration)
                print(f"  Eval: {eval_stats['eval/bb_per_100']:.2f} BB/100")

            # Save checkpoint
            if (iteration + 1) % self.config.save_every == 0:
                self.save(os.path.join(
                    self.config.save_dir,
                    f"checkpoint_{iteration + 1}.pt"
                ))

        # Final save
        self.save(os.path.join(self.config.save_dir, "final.pt"))

        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {total_time:.0f}s")
        print(f"  Total hands: {self.total_hands:,}")
        print(f"  Hands/sec: {self.total_hands / total_time:.0f}")

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": vars(self.config),
            "total_steps": self.total_steps,
            "total_hands": self.total_hands,
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_hands = checkpoint.get("total_hands", 0)
        print(f"Loaded checkpoint from {path}")

    def shutdown(self):
        """Shutdown workers and cleanup."""
        self.logger.close()
        if self.use_ray and ray.is_initialized():
            ray.shutdown()


def merge_rollouts(rollouts: List[RolloutData]) -> RolloutData:
    """Merge multiple rollouts into one."""
    merged = RolloutData()

    for r in rollouts:
        merged.states.extend(r.states)
        merged.actions.extend(r.actions)
        merged.rewards.extend(r.rewards)
        merged.dones.extend(r.dones)
        merged.action_masks.extend(r.action_masks)
        merged.values.extend(r.values)
        merged.log_probs.extend(r.log_probs)
        merged.player_ids.extend(r.player_ids)
        merged.total_profit += r.total_profit
        merged.num_hands += r.num_hands

    return merged


# Legacy DistributedTrainer for backwards compatibility
class DistributedTrainer:
    """Distributed training orchestrator (legacy interface)."""

    def __init__(
        self,
        num_workers: int = 4,
        num_players: int = 6,
        hands_per_rollout: int = 100,
        opponent_pool_name: str = "mixed_pool",
        use_ray: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize distributed trainer."""
        self.config = TrainingConfig(
            num_workers=num_workers,
            num_players=num_players,
            hands_per_rollout=hands_per_rollout,
            opponent_pool_name=opponent_pool_name,
            seed=seed,
        )
        self.trainer = RayPPOTrainer(self.config)

    def collect_rollouts(
        self,
        model_weights: Optional[Dict] = None,
        player_id: int = 0,
    ) -> List[RolloutData]:
        """Collect rollouts from all workers."""
        return self.trainer._collect_rollouts(player_id)

    def parallel_evaluate(
        self,
        model_weights: Optional[Dict] = None,
        num_hands_per_worker: int = 100,
        player_id: int = 0,
    ) -> Tuple[float, int]:
        """Evaluate in parallel."""
        stats = self.trainer._evaluate(num_hands_per_worker)
        return stats["eval/total_profit"], stats["eval/total_hands"]

    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            "num_workers": self.config.num_workers,
            "ray_available": RAY_AVAILABLE,
            "using_ray": self.trainer.use_ray,
            "hands_per_rollout": self.config.hands_per_rollout,
            "total_hands_per_collect": self.config.num_workers * self.config.hands_per_rollout,
        }

    def shutdown(self):
        """Shutdown workers."""
        self.trainer.shutdown()


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed Poker AI Training")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--players", type=int, default=6, help="Players per table")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--hands", type=int, default=100, help="Hands per rollout per worker")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--log-dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="poker-ai", help="WandB project name")
    parser.add_argument("--test", action="store_true", help="Run quick test instead of full training")

    args = parser.parse_args()

    if args.test:
        print("=== Distributed Training Quick Test ===\n")
        print(f"Ray available: {RAY_AVAILABLE}")
        print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
        print(f"WandB available: {WANDB_AVAILABLE}")
        print(f"Torch available: {TORCH_AVAILABLE}")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print()

        config = TrainingConfig(
            num_workers=2,
            num_players=2,
            hands_per_rollout=50,
            total_iterations=10,
            eval_every=5,
            save_every=10,
            use_tensorboard=False,
            use_wandb=False,
            seed=42,
        )

        trainer = RayPPOTrainer(config)

        print("\nRunning quick training test...")
        trainer.train()

        trainer.shutdown()
        print("\nQuick test passed!")
    else:
        config = TrainingConfig(
            num_workers=args.workers,
            num_players=args.players,
            hands_per_rollout=args.hands,
            total_iterations=args.iterations,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            use_tensorboard=not args.no_tensorboard,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            seed=args.seed,
        )

        trainer = RayPPOTrainer(config)

        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            trainer.shutdown()
