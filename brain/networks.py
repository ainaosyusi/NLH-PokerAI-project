"""Neural network architectures for poker AI.

This module provides:
1. RegretNetwork - For Deep CFR regret prediction
2. StrategyNetwork - For Deep CFR strategy output
3. ActorCriticNetwork - For PPO/A2C reinforcement learning
4. ValueNetwork - For value estimation
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        output_activation: Optional[str] = None,
    ):
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function ("relu", "elu", "tanh", "gelu")
            dropout: Dropout probability
            output_activation: Optional output activation
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Build layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0 else nn.Identity())
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Activation functions
        self.activation = self._get_activation(activation)
        self.output_activation = (
            self._get_activation(output_activation) if output_activation else None
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.Softmax(dim=-1),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer, dropout in zip(self.layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)

        x = self.output_layer(x)

        if self.output_activation:
            x = self.output_activation(x)

        return x


class RegretNetwork(nn.Module):
    """Network for predicting action regrets in Deep CFR.

    In CFR, regret represents how much better an action would have been
    compared to the mixed strategy. Positive regret means the action
    should be taken more often.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
    ):
        """Initialize regret network.

        Args:
            input_dim: Feature vector dimension
            num_actions: Number of possible actions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation="relu",
            dropout=dropout,
            output_activation=None,  # Raw regret values
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - predict regrets for each action.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Regret predictions [batch_size, num_actions]
        """
        return self.network(x)

    def get_strategy(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert regrets to strategy (probability distribution).

        Uses regret matching: strategy proportional to positive regret.

        Args:
            x: Input features
            action_mask: Optional mask for valid actions

        Returns:
            Strategy (probability distribution over actions)
        """
        regrets = self.forward(x)

        # Apply action mask if provided
        if action_mask is not None:
            regrets = regrets.masked_fill(~action_mask.bool(), float('-inf'))

        # Regret matching: take positive part and normalize
        positive_regrets = F.relu(regrets)

        # Add small epsilon to avoid division by zero
        sum_positive = positive_regrets.sum(dim=-1, keepdim=True) + 1e-8

        strategy = positive_regrets / sum_positive

        # If all regrets are negative, use uniform over valid actions
        uniform_mask = (sum_positive.squeeze(-1) < 1e-6)
        if uniform_mask.any():
            if action_mask is not None:
                uniform = action_mask.float() / action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            else:
                uniform = torch.ones_like(strategy) / self.num_actions
            strategy = torch.where(uniform_mask.unsqueeze(-1), uniform, strategy)

        return strategy


class StrategyNetwork(nn.Module):
    """Network for directly outputting action probabilities.

    Used in Deep CFR as the average strategy network.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
    ):
        """Initialize strategy network.

        Args:
            input_dim: Feature vector dimension
            num_actions: Number of possible actions
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation="relu",
            dropout=dropout,
            output_activation=None,  # We'll apply softmax with masking
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - output strategy (probabilities).

        Args:
            x: Input features [batch_size, input_dim]
            action_mask: Optional mask for valid actions [batch_size, num_actions]

        Returns:
            Strategy probabilities [batch_size, num_actions]
        """
        logits = self.network(x)

        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))

        # Softmax to get probabilities
        return F.softmax(logits, dim=-1)

    def get_log_prob(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get log probability of specific actions.

        Args:
            x: Input features
            actions: Action indices [batch_size]
            action_mask: Optional action mask

        Returns:
            Log probabilities [batch_size]
        """
        strategy = self.forward(x, action_mask)
        return torch.log(strategy.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)


class ValueNetwork(nn.Module):
    """Network for estimating state value (expected return)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        dropout: float = 0.1,
    ):
        """Initialize value network.

        Args:
            input_dim: Feature vector dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
            output_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - estimate state value.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Value estimates [batch_size]
        """
        return self.network(x).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network for PPO/A2C.

    The actor outputs action probabilities (policy).
    The critic outputs state value estimate.
    They share some layers for efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        shared_dims: List[int] = [256, 256],
        actor_dims: List[int] = [128],
        critic_dims: List[int] = [128],
        dropout: float = 0.1,
    ):
        """Initialize actor-critic network.

        Args:
            input_dim: Feature vector dimension
            num_actions: Number of possible actions
            shared_dims: Shared layer dimensions
            actor_dims: Actor-specific layer dimensions
            critic_dims: Critic-specific layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions

        # Shared layers
        self.shared = MLP(
            input_dim=input_dim,
            hidden_dims=shared_dims[:-1] if len(shared_dims) > 1 else [],
            output_dim=shared_dims[-1],
            activation="relu",
            dropout=dropout,
        )

        # Actor head
        self.actor = MLP(
            input_dim=shared_dims[-1],
            hidden_dims=actor_dims,
            output_dim=num_actions,
            activation="relu",
            dropout=dropout,
        )

        # Critic head
        self.critic = MLP(
            input_dim=shared_dims[-1],
            hidden_dims=critic_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]
            action_mask: Optional mask for valid actions

        Returns:
            Tuple of (action_probs, value)
        """
        shared_features = self.shared(x)

        # Actor output
        logits = self.actor(shared_features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))
        action_probs = F.softmax(logits, dim=-1)

        # Critic output
        value = self.critic(shared_features).squeeze(-1)

        return action_probs, value

    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Args:
            x: Input features
            action_mask: Optional action mask
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_probs, value = self.forward(x, action_mask)

        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)).squeeze(1) + 1e-8)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO loss calculation.

        Args:
            x: Input features
            actions: Actions to evaluate
            action_mask: Optional action mask

        Returns:
            Tuple of (log_prob, value, entropy)
        """
        action_probs, value = self.forward(x, action_mask)

        dist = Categorical(action_probs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy


class DuelingNetwork(nn.Module):
    """Dueling network architecture for DQN-style learning.

    Separates value and advantage streams for better learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [256, 256],
        value_dims: List[int] = [128],
        advantage_dims: List[int] = [128],
        dropout: float = 0.1,
    ):
        """Initialize dueling network.

        Args:
            input_dim: Feature vector dimension
            num_actions: Number of possible actions
            hidden_dims: Shared hidden layer dimensions
            value_dims: Value stream dimensions
            advantage_dims: Advantage stream dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.num_actions = num_actions

        # Shared feature extractor
        self.shared = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else [],
            output_dim=hidden_dims[-1],
            activation="relu",
            dropout=dropout,
        )

        # Value stream
        self.value_stream = MLP(
            input_dim=hidden_dims[-1],
            hidden_dims=value_dims,
            output_dim=1,
            activation="relu",
            dropout=dropout,
        )

        # Advantage stream
        self.advantage_stream = MLP(
            input_dim=hidden_dims[-1],
            hidden_dims=advantage_dims,
            output_dim=num_actions,
            activation="relu",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute Q-values.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Q-values [batch_size, num_actions]
        """
        shared_features = self.shared(x)

        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values


def create_regret_network(
    input_dim: int,
    num_actions: int,
    size: str = "medium",
) -> RegretNetwork:
    """Factory function to create regret network.

    Args:
        input_dim: Feature dimension
        num_actions: Number of actions
        size: "small", "medium", or "large"

    Returns:
        RegretNetwork instance
    """
    configs = {
        "small": {"hidden_dims": [128, 128], "dropout": 0.1},
        "medium": {"hidden_dims": [256, 256, 128], "dropout": 0.1},
        "large": {"hidden_dims": [512, 512, 256, 128], "dropout": 0.15},
    }
    config = configs.get(size, configs["medium"])
    return RegretNetwork(input_dim, num_actions, **config)


def create_actor_critic(
    input_dim: int,
    num_actions: int,
    size: str = "medium",
) -> ActorCriticNetwork:
    """Factory function to create actor-critic network.

    Args:
        input_dim: Feature dimension
        num_actions: Number of actions
        size: "small", "medium", or "large"

    Returns:
        ActorCriticNetwork instance
    """
    configs = {
        "small": {
            "shared_dims": [128, 128],
            "actor_dims": [64],
            "critic_dims": [64],
            "dropout": 0.1,
        },
        "medium": {
            "shared_dims": [256, 256],
            "actor_dims": [128],
            "critic_dims": [128],
            "dropout": 0.1,
        },
        "large": {
            "shared_dims": [512, 512, 256],
            "actor_dims": [256, 128],
            "critic_dims": [256, 128],
            "dropout": 0.15,
        },
    }
    config = configs.get(size, configs["medium"])
    return ActorCriticNetwork(input_dim, num_actions, **config)
