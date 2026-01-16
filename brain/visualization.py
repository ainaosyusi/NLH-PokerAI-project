"""Learning Visualization Module for Poker AI.

This module provides comprehensive visualization and monitoring tools:
1. Training metrics dashboard (loss, reward, entropy)
2. Action distribution analysis
3. Win rate tracking by position and street
4. Exploitability metrics
5. Real-time plotting with matplotlib and TensorBoard
"""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import os
import time
import json
from datetime import datetime

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


# Action names for 11-action pot-geometric system
ACTION_NAMES = [
    "Fold", "Check/Call", "Min Raise",
    "33% Pot", "50% Pot", "75% Pot", "100% Pot",
    "125% Pot", "150% Pot", "200% Pot", "All-in"
]

STREET_NAMES = ["Preflop", "Flop", "Turn", "River"]
POSITION_NAMES_2P = ["Button/SB", "BB"]
POSITION_NAMES_6P = ["UTG", "MP", "CO", "BTN", "SB", "BB"]


@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""
    iterations: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Loss metrics
    policy_loss: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)

    # Reward metrics
    mean_reward: List[float] = field(default_factory=list)
    reward_std: List[float] = field(default_factory=list)
    bb_per_100: List[float] = field(default_factory=list)

    # Training stats
    hands_played: List[int] = field(default_factory=list)
    steps_collected: List[int] = field(default_factory=list)
    hands_per_second: List[float] = field(default_factory=list)

    # Learning rate (if scheduled)
    learning_rate: List[float] = field(default_factory=list)


@dataclass
class ActionStats:
    """Statistics for action distributions."""
    # Action counts by street [street][action] -> count
    action_counts: Dict[int, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    # Action counts by position [position][action] -> count
    position_action_counts: Dict[int, Dict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    # Action counts overall
    total_action_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def record_action(self, action: int, street: int, position: int):
        """Record an action taken."""
        self.action_counts[street][action] += 1
        self.position_action_counts[position][action] += 1
        self.total_action_counts[action] += 1

    def get_distribution(self, by: str = "total") -> np.ndarray:
        """Get action distribution as probabilities."""
        if by == "total":
            counts = np.array([self.total_action_counts.get(i, 0) for i in range(11)])
        elif by.startswith("street_"):
            street = int(by.split("_")[1])
            counts = np.array([self.action_counts[street].get(i, 0) for i in range(11)])
        elif by.startswith("position_"):
            position = int(by.split("_")[1])
            counts = np.array([self.position_action_counts[position].get(i, 0) for i in range(11)])
        else:
            counts = np.array([self.total_action_counts.get(i, 0) for i in range(11)])

        total = counts.sum()
        if total > 0:
            return counts / total
        return counts


@dataclass
class WinRateStats:
    """Win rate statistics by position and street."""
    # [position] -> (wins, total_hands, total_profit)
    by_position: Dict[int, Tuple[int, int, float]] = field(
        default_factory=lambda: defaultdict(lambda: (0, 0, 0.0))
    )
    # [street_reached] -> (wins, total_hands, total_profit)
    by_street_reached: Dict[int, Tuple[int, int, float]] = field(
        default_factory=lambda: defaultdict(lambda: (0, 0, 0.0))
    )
    # Rolling window for recent performance
    recent_results: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_result(self, position: int, street_reached: int, profit: float, won: bool):
        """Record a hand result."""
        # Update position stats
        wins, hands, total = self.by_position[position]
        self.by_position[position] = (wins + int(won), hands + 1, total + profit)

        # Update street stats
        wins, hands, total = self.by_street_reached[street_reached]
        self.by_street_reached[street_reached] = (wins + int(won), hands + 1, total + profit)

        # Add to rolling window
        self.recent_results.append((position, street_reached, profit, won))

    def get_win_rate_by_position(self, num_players: int = 6) -> Dict[str, float]:
        """Get win rates by position."""
        position_names = POSITION_NAMES_6P if num_players == 6 else POSITION_NAMES_2P
        result = {}
        for pos in range(min(num_players, len(position_names))):
            wins, hands, _ = self.by_position[pos]
            if hands > 0:
                result[position_names[pos]] = wins / hands
            else:
                result[position_names[pos]] = 0.0
        return result

    def get_profit_by_position(self, num_players: int = 6) -> Dict[str, float]:
        """Get average profit by position."""
        position_names = POSITION_NAMES_6P if num_players == 6 else POSITION_NAMES_2P
        result = {}
        for pos in range(min(num_players, len(position_names))):
            _, hands, profit = self.by_position[pos]
            if hands > 0:
                result[position_names[pos]] = profit / hands
            else:
                result[position_names[pos]] = 0.0
        return result


@dataclass
class ExploitabilityMetrics:
    """Metrics for measuring strategy exploitability."""
    # Best response value estimates
    br_values: List[float] = field(default_factory=list)
    # Nash distance estimates
    nash_distance: List[float] = field(default_factory=list)
    # Regret values
    cumulative_regret: List[float] = field(default_factory=list)
    # Iterations when measured
    iterations: List[int] = field(default_factory=list)


class PokerVisualizer:
    """Main visualization class for poker AI training."""

    def __init__(
        self,
        log_dir: str = "logs",
        num_players: int = 6,
        num_actions: int = 11,
        use_tensorboard: bool = True,
        save_plots: bool = True,
        update_freq: int = 10,
    ):
        """Initialize visualizer.

        Args:
            log_dir: Directory for logs and plots
            num_players: Number of players
            num_actions: Number of actions (11 for pot-geometric)
            use_tensorboard: Whether to use TensorBoard
            save_plots: Whether to save matplotlib plots
            update_freq: How often to update plots (iterations)
        """
        self.log_dir = log_dir
        self.num_players = num_players
        self.num_actions = num_actions
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.save_plots = save_plots
        self.update_freq = update_freq

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)

        # Initialize metrics containers
        self.training_metrics = TrainingMetrics()
        self.action_stats = ActionStats()
        self.win_rate_stats = WinRateStats()
        self.exploitability = ExploitabilityMetrics()

        # TensorBoard writer
        self.tb_writer = None
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir)

        # Matplotlib setup
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8-darkgrid')
            self.fig = None
            self.axes = None

        # Start time
        self.start_time = time.time()
        self.current_iteration = 0

        print(f"PokerVisualizer initialized:")
        print(f"  Log directory: {log_dir}")
        print(f"  TensorBoard: {self.use_tensorboard}")
        print(f"  Matplotlib: {MATPLOTLIB_AVAILABLE}")

    def log_training_step(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        mean_reward: float,
        hands_played: int,
        steps_collected: int,
        bb_per_100: Optional[float] = None,
        learning_rate: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log metrics from a training step."""
        self.current_iteration = iteration
        elapsed = time.time() - self.start_time

        # Record to training metrics
        self.training_metrics.iterations.append(iteration)
        self.training_metrics.timestamps.append(elapsed)
        self.training_metrics.policy_loss.append(policy_loss)
        self.training_metrics.value_loss.append(value_loss)
        self.training_metrics.entropy.append(entropy)
        self.training_metrics.total_loss.append(policy_loss + 0.5 * value_loss - 0.01 * entropy)
        self.training_metrics.mean_reward.append(mean_reward)
        self.training_metrics.hands_played.append(hands_played)
        self.training_metrics.steps_collected.append(steps_collected)

        if bb_per_100 is not None:
            self.training_metrics.bb_per_100.append(bb_per_100)

        if learning_rate is not None:
            self.training_metrics.learning_rate.append(learning_rate)

        # Calculate hands per second
        if elapsed > 0:
            total_hands = sum(self.training_metrics.hands_played)
            self.training_metrics.hands_per_second.append(total_hands / elapsed)

        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("loss/policy", policy_loss, iteration)
            self.tb_writer.add_scalar("loss/value", value_loss, iteration)
            self.tb_writer.add_scalar("loss/entropy", entropy, iteration)
            self.tb_writer.add_scalar("reward/mean", mean_reward, iteration)
            self.tb_writer.add_scalar("training/hands_played", hands_played, iteration)
            self.tb_writer.add_scalar("training/steps_collected", steps_collected, iteration)

            if bb_per_100 is not None:
                self.tb_writer.add_scalar("performance/bb_per_100", bb_per_100, iteration)

            if learning_rate is not None:
                self.tb_writer.add_scalar("training/learning_rate", learning_rate, iteration)

            if extra_metrics:
                for name, value in extra_metrics.items():
                    self.tb_writer.add_scalar(f"extra/{name}", value, iteration)

        # Update plots periodically
        if self.save_plots and iteration % self.update_freq == 0:
            self.save_training_plots()

    def log_action(self, action: int, street: int, position: int):
        """Log an action taken during training/evaluation."""
        self.action_stats.record_action(action, street, position)

    def log_hand_result(
        self,
        position: int,
        street_reached: int,
        profit: float,
        won: bool,
    ):
        """Log the result of a completed hand."""
        self.win_rate_stats.record_result(position, street_reached, profit, won)

    def log_action_distribution(self, distribution: np.ndarray, iteration: int, tag: str = "overall"):
        """Log an action distribution to TensorBoard."""
        if self.tb_writer:
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(len(distribution)), distribution)
            ax.set_xticks(range(len(distribution)))
            ax.set_xticklabels(ACTION_NAMES, rotation=45, ha='right')
            ax.set_ylabel("Probability")
            ax.set_title(f"Action Distribution ({tag})")
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, val in zip(bars, distribution):
                if val > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            self.tb_writer.add_figure(f"action_distribution/{tag}", fig, iteration)
            plt.close(fig)

    def log_exploitability(self, iteration: int, br_value: float, nash_distance: Optional[float] = None):
        """Log exploitability metrics."""
        self.exploitability.iterations.append(iteration)
        self.exploitability.br_values.append(br_value)
        if nash_distance is not None:
            self.exploitability.nash_distance.append(nash_distance)

        if self.tb_writer:
            self.tb_writer.add_scalar("exploitability/best_response_value", br_value, iteration)
            if nash_distance is not None:
                self.tb_writer.add_scalar("exploitability/nash_distance", nash_distance, iteration)

    def save_training_plots(self):
        """Save training progress plots."""
        if not MATPLOTLIB_AVAILABLE:
            return

        metrics = self.training_metrics
        if len(metrics.iterations) < 2:
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        iterations = metrics.iterations

        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics.policy_loss:
            ax1.plot(iterations, metrics.policy_loss, label='Policy Loss', alpha=0.8)
        if metrics.value_loss:
            ax1.plot(iterations, metrics.value_loss, label='Value Loss', alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Entropy
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics.entropy:
            ax2.plot(iterations, metrics.entropy, color='green', alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Policy Entropy')
        ax2.grid(True, alpha=0.3)

        # 3. Mean Reward
        ax3 = fig.add_subplot(gs[0, 2])
        if metrics.mean_reward:
            ax3.plot(iterations, metrics.mean_reward, color='orange', alpha=0.8)
            # Add smoothed line
            if len(metrics.mean_reward) > 10:
                window = min(50, len(metrics.mean_reward) // 5)
                smoothed = np.convolve(metrics.mean_reward,
                                       np.ones(window)/window, mode='valid')
                ax3.plot(iterations[window-1:], smoothed,
                        color='red', linewidth=2, label='Smoothed')
                ax3.legend()
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Mean Reward')
        ax3.set_title('Training Reward')
        ax3.grid(True, alpha=0.3)

        # 4. BB/100 if available
        ax4 = fig.add_subplot(gs[1, 0])
        if metrics.bb_per_100:
            ax4.plot(range(len(metrics.bb_per_100)), metrics.bb_per_100,
                    color='purple', alpha=0.8)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('BB/100')
        ax4.set_title('Win Rate (BB/100 hands)')
        ax4.grid(True, alpha=0.3)

        # 5. Hands per second
        ax5 = fig.add_subplot(gs[1, 1])
        if metrics.hands_per_second:
            ax5.plot(iterations, metrics.hands_per_second, color='teal', alpha=0.8)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Hands/sec')
        ax5.set_title('Training Speed')
        ax5.grid(True, alpha=0.3)

        # 6. Action Distribution (overall)
        ax6 = fig.add_subplot(gs[1, 2])
        action_dist = self.action_stats.get_distribution("total")
        if action_dist.sum() > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, len(action_dist)))
            bars = ax6.bar(range(len(action_dist)), action_dist, color=colors)
            ax6.set_xticks(range(len(action_dist)))
            ax6.set_xticklabels([f"A{i}" for i in range(len(action_dist))], fontsize=8)
            ax6.set_ylabel('Frequency')
            ax6.set_title('Action Distribution')
        ax6.grid(True, alpha=0.3)

        # 7. Action Distribution by Street
        ax7 = fig.add_subplot(gs[2, 0])
        street_data = []
        for street in range(4):
            dist = self.action_stats.get_distribution(f"street_{street}")
            if dist.sum() > 0:
                street_data.append(dist)
        if street_data:
            street_data = np.array(street_data)
            im = ax7.imshow(street_data, aspect='auto', cmap='YlOrRd')
            ax7.set_yticks(range(len(street_data)))
            ax7.set_yticklabels(STREET_NAMES[:len(street_data)])
            ax7.set_xticks(range(11))
            ax7.set_xticklabels([f"A{i}" for i in range(11)], fontsize=7)
            ax7.set_title('Actions by Street')
            plt.colorbar(im, ax=ax7, label='Frequency')

        # 8. Win Rate by Position
        ax8 = fig.add_subplot(gs[2, 1])
        profit_by_pos = self.win_rate_stats.get_profit_by_position(self.num_players)
        if profit_by_pos:
            positions = list(profit_by_pos.keys())
            profits = list(profit_by_pos.values())
            colors = ['green' if p > 0 else 'red' for p in profits]
            ax8.bar(positions, profits, color=colors, alpha=0.7)
            ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax8.set_ylabel('Avg Profit/Hand')
            ax8.set_title('Profit by Position')
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax8.grid(True, alpha=0.3)

        # 9. Cumulative Hands
        ax9 = fig.add_subplot(gs[2, 2])
        if metrics.hands_played:
            cumulative = np.cumsum(metrics.hands_played)
            ax9.plot(iterations, cumulative, color='navy', alpha=0.8)
            ax9.fill_between(iterations, cumulative, alpha=0.3)
        ax9.set_xlabel('Iteration')
        ax9.set_ylabel('Total Hands')
        ax9.set_title('Cumulative Hands Played')
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.log_dir, "plots", "training_progress.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Also save to TensorBoard
        if self.tb_writer:
            fig2 = plt.figure(figsize=(16, 12))
            # Recreate the figure for TensorBoard (matplotlib figures can't be reused)
            self._recreate_figure_for_tb(fig2)
            self.tb_writer.add_figure("training/dashboard", fig2, self.current_iteration)
            plt.close(fig2)

    def _recreate_figure_for_tb(self, fig):
        """Recreate the training figure for TensorBoard."""
        metrics = self.training_metrics
        gs = gridspec.GridSpec(2, 2, figure=fig)

        iterations = metrics.iterations

        # Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if metrics.policy_loss:
            ax1.plot(iterations, metrics.policy_loss, label='Policy', alpha=0.8)
        if metrics.value_loss:
            ax1.plot(iterations, metrics.value_loss, label='Value', alpha=0.8)
        ax1.set_title('Losses')
        ax1.legend()

        # Reward
        ax2 = fig.add_subplot(gs[0, 1])
        if metrics.mean_reward:
            ax2.plot(iterations, metrics.mean_reward, alpha=0.8)
        ax2.set_title('Mean Reward')

        # BB/100
        ax3 = fig.add_subplot(gs[1, 0])
        if metrics.bb_per_100:
            ax3.plot(metrics.bb_per_100, alpha=0.8)
            ax3.axhline(y=0, color='gray', linestyle='--')
        ax3.set_title('BB/100')

        # Action Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        action_dist = self.action_stats.get_distribution("total")
        if action_dist.sum() > 0:
            ax4.bar(range(len(action_dist)), action_dist)
        ax4.set_title('Actions')

        plt.tight_layout()

    def create_action_heatmap(self) -> Optional[str]:
        """Create a heatmap of actions by street and save it."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for street, ax in enumerate(axes):
            dist = self.action_stats.get_distribution(f"street_{street}")

            # Reshape to 2D for heatmap visualization
            # Group actions: [Passive (0-1), Small (2-4), Medium (5-7), Large (8-10)]
            groups = [
                dist[0:2].sum(),   # Fold, Check/Call
                dist[2:5].sum(),   # Min, 33%, 50%
                dist[5:8].sum(),   # 75%, 100%, 125%
                dist[8:11].sum(),  # 150%, 200%, All-in
            ]

            colors = plt.cm.RdYlGn_r(dist)
            bars = ax.bar(range(len(dist)), dist, color=colors)
            ax.set_xticks(range(len(dist)))
            ax.set_xticklabels([f"A{i}" for i in range(len(dist))], rotation=45, fontsize=8)
            ax.set_title(STREET_NAMES[street])
            ax.set_ylim(0, 1)

        plt.suptitle("Action Distribution by Street")
        plt.tight_layout()

        path = os.path.join(self.log_dir, "plots", "action_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return path

    def create_position_analysis(self) -> Optional[str]:
        """Create position analysis plots."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Win rate by position
        ax1 = axes[0]
        win_rates = self.win_rate_stats.get_win_rate_by_position(self.num_players)
        if win_rates:
            positions = list(win_rates.keys())
            rates = list(win_rates.values())
            ax1.bar(positions, rates, color='steelblue', alpha=0.7)
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.set_ylabel('Win Rate')
            ax1.set_title('Win Rate by Position')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Profit by position
        ax2 = axes[1]
        profits = self.win_rate_stats.get_profit_by_position(self.num_players)
        if profits:
            positions = list(profits.keys())
            values = list(profits.values())
            colors = ['green' if v > 0 else 'red' for v in values]
            ax2.bar(positions, values, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Avg Profit')
            ax2.set_title('Profit by Position')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Action distribution by position (heatmap)
        ax3 = axes[2]
        position_names = POSITION_NAMES_6P if self.num_players == 6 else POSITION_NAMES_2P
        pos_data = []
        for pos in range(min(self.num_players, len(position_names))):
            dist = self.action_stats.get_distribution(f"position_{pos}")
            pos_data.append(dist)

        if pos_data:
            pos_data = np.array(pos_data)
            im = ax3.imshow(pos_data, aspect='auto', cmap='YlOrRd')
            ax3.set_yticks(range(len(pos_data)))
            ax3.set_yticklabels(position_names[:len(pos_data)])
            ax3.set_xticks(range(11))
            ax3.set_xticklabels([f"A{i}" for i in range(11)], fontsize=7)
            ax3.set_title('Actions by Position')
            plt.colorbar(im, ax=ax3, label='Frequency')

        plt.tight_layout()

        path = os.path.join(self.log_dir, "plots", "position_analysis.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return path

    def save_metrics_json(self):
        """Save all metrics to a JSON file."""
        data = {
            "training_metrics": {
                "iterations": self.training_metrics.iterations,
                "policy_loss": self.training_metrics.policy_loss,
                "value_loss": self.training_metrics.value_loss,
                "entropy": self.training_metrics.entropy,
                "mean_reward": self.training_metrics.mean_reward,
                "bb_per_100": self.training_metrics.bb_per_100,
                "hands_played": self.training_metrics.hands_played,
            },
            "action_stats": {
                "total": self.action_stats.get_distribution("total").tolist(),
                "by_street": {
                    STREET_NAMES[i]: self.action_stats.get_distribution(f"street_{i}").tolist()
                    for i in range(4)
                },
            },
            "win_rate_stats": {
                "by_position": self.win_rate_stats.get_profit_by_position(self.num_players),
            },
            "timestamp": datetime.now().isoformat(),
            "total_iterations": self.current_iteration,
        }

        path = os.path.join(self.log_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return path

    def generate_report(self) -> str:
        """Generate a text report of training progress."""
        metrics = self.training_metrics

        report = []
        report.append("=" * 60)
        report.append("POKER AI TRAINING REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary stats
        report.append("## Training Summary")
        report.append(f"Total Iterations: {self.current_iteration}")
        total_hands = sum(metrics.hands_played) if metrics.hands_played else 0
        report.append(f"Total Hands Played: {total_hands:,}")
        elapsed = time.time() - self.start_time
        report.append(f"Training Time: {elapsed/60:.1f} minutes")
        if elapsed > 0:
            report.append(f"Average Speed: {total_hands/elapsed:.0f} hands/sec")
        report.append("")

        # Loss metrics
        report.append("## Loss Metrics (Latest)")
        if metrics.policy_loss:
            report.append(f"Policy Loss: {metrics.policy_loss[-1]:.6f}")
        if metrics.value_loss:
            report.append(f"Value Loss: {metrics.value_loss[-1]:.6f}")
        if metrics.entropy:
            report.append(f"Entropy: {metrics.entropy[-1]:.6f}")
        report.append("")

        # Performance
        report.append("## Performance")
        if metrics.mean_reward:
            recent_reward = np.mean(metrics.mean_reward[-100:])
            report.append(f"Recent Mean Reward: {recent_reward:.2f}")
        if metrics.bb_per_100:
            recent_bb = np.mean(metrics.bb_per_100[-10:]) if len(metrics.bb_per_100) > 0 else 0
            report.append(f"Recent BB/100: {recent_bb:.2f}")
        report.append("")

        # Action distribution
        report.append("## Action Distribution (Overall)")
        action_dist = self.action_stats.get_distribution("total")
        for i, (name, prob) in enumerate(zip(ACTION_NAMES, action_dist)):
            if prob > 0.001:
                report.append(f"  {name}: {prob*100:.1f}%")
        report.append("")

        # Position stats
        report.append("## Position Statistics")
        profit_by_pos = self.win_rate_stats.get_profit_by_position(self.num_players)
        for pos, profit in profit_by_pos.items():
            report.append(f"  {pos}: {profit:+.1f} chips/hand")
        report.append("")

        report.append("=" * 60)

        report_text = "\n".join(report)

        # Save to file
        path = os.path.join(self.log_dir, "training_report.txt")
        with open(path, 'w') as f:
            f.write(report_text)

        return report_text

    def close(self):
        """Close the visualizer and save final outputs."""
        # Save final plots
        if self.save_plots and MATPLOTLIB_AVAILABLE:
            self.save_training_plots()
            self.create_action_heatmap()
            self.create_position_analysis()

        # Save metrics JSON
        self.save_metrics_json()

        # Generate report
        self.generate_report()

        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()

        print(f"\nVisualization outputs saved to: {self.log_dir}")


class LiveDashboard:
    """Real-time training dashboard using matplotlib."""

    def __init__(self, visualizer: PokerVisualizer, update_interval: int = 1000):
        """Initialize live dashboard.

        Args:
            visualizer: PokerVisualizer instance
            update_interval: Update interval in milliseconds
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for LiveDashboard")

        self.visualizer = visualizer
        self.update_interval = update_interval

        # Setup figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle("Poker AI Training Dashboard", fontsize=14)

        # Initialize plots
        self.lines = {}
        self._init_plots()

        plt.tight_layout()

    def _init_plots(self):
        """Initialize plot elements."""
        # Loss plot
        ax = self.axes[0, 0]
        self.lines['policy_loss'], = ax.plot([], [], label='Policy', alpha=0.8)
        self.lines['value_loss'], = ax.plot([], [], label='Value', alpha=0.8)
        ax.set_title('Losses')
        ax.set_xlabel('Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Reward plot
        ax = self.axes[0, 1]
        self.lines['reward'], = ax.plot([], [], color='orange', alpha=0.8)
        ax.set_title('Mean Reward')
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

        # Entropy plot
        ax = self.axes[0, 2]
        self.lines['entropy'], = ax.plot([], [], color='green', alpha=0.8)
        ax.set_title('Entropy')
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

        # BB/100 plot
        ax = self.axes[1, 0]
        self.lines['bb'], = ax.plot([], [], color='purple', alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('BB/100')
        ax.set_xlabel('Evaluation')
        ax.grid(True, alpha=0.3)

        # Action distribution (will update bars)
        self.axes[1, 1].set_title('Action Distribution')
        self.axes[1, 1].set_xlabel('Action')

        # Speed plot
        ax = self.axes[1, 2]
        self.lines['speed'], = ax.plot([], [], color='teal', alpha=0.8)
        ax.set_title('Hands/sec')
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)

    def update(self, frame):
        """Update function for animation."""
        metrics = self.visualizer.training_metrics

        if len(metrics.iterations) < 2:
            return list(self.lines.values())

        iterations = metrics.iterations

        # Update loss lines
        self.lines['policy_loss'].set_data(iterations, metrics.policy_loss)
        self.lines['value_loss'].set_data(iterations, metrics.value_loss)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()

        # Update reward
        self.lines['reward'].set_data(iterations, metrics.mean_reward)
        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()

        # Update entropy
        self.lines['entropy'].set_data(iterations, metrics.entropy)
        self.axes[0, 2].relim()
        self.axes[0, 2].autoscale_view()

        # Update BB/100
        if metrics.bb_per_100:
            self.lines['bb'].set_data(range(len(metrics.bb_per_100)), metrics.bb_per_100)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()

        # Update action distribution
        ax = self.axes[1, 1]
        ax.clear()
        action_dist = self.visualizer.action_stats.get_distribution("total")
        if action_dist.sum() > 0:
            ax.bar(range(len(action_dist)), action_dist, color='steelblue')
        ax.set_title('Action Distribution')
        ax.set_ylim(0, 1)

        # Update speed
        if metrics.hands_per_second:
            self.lines['speed'].set_data(iterations, metrics.hands_per_second)
            self.axes[1, 2].relim()
            self.axes[1, 2].autoscale_view()

        self.fig.canvas.draw()

        return list(self.lines.values())

    def start(self):
        """Start the live dashboard."""
        anim = FuncAnimation(
            self.fig, self.update, interval=self.update_interval, blit=False
        )
        plt.show()


# Convenience function for integration with distributed training
def create_visualizer_for_training(
    config: Any,
    log_dir: Optional[str] = None,
) -> PokerVisualizer:
    """Create a visualizer for distributed training.

    Args:
        config: TrainingConfig instance
        log_dir: Override log directory

    Returns:
        Configured PokerVisualizer
    """
    return PokerVisualizer(
        log_dir=log_dir or getattr(config, 'log_dir', 'logs'),
        num_players=getattr(config, 'num_players', 6),
        num_actions=11,
        use_tensorboard=getattr(config, 'use_tensorboard', True),
        save_plots=True,
        update_freq=getattr(config, 'eval_every', 50),
    )


# Test/demo
if __name__ == "__main__":
    print("=== Poker Visualization Module Test ===\n")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    print(f"Pandas available: {PANDAS_AVAILABLE}")
    print()

    # Create visualizer
    viz = PokerVisualizer(
        log_dir="logs/viz_test",
        num_players=2,
        use_tensorboard=TENSORBOARD_AVAILABLE,
    )

    # Simulate some training
    np.random.seed(42)

    print("Simulating training...")
    for i in range(100):
        # Fake training metrics
        policy_loss = 0.5 * np.exp(-i/50) + np.random.normal(0, 0.01)
        value_loss = 1.0 * np.exp(-i/30) + np.random.normal(0, 0.02)
        entropy = 2.0 - i * 0.01 + np.random.normal(0, 0.1)
        mean_reward = -1000 + i * 20 + np.random.normal(0, 100)

        viz.log_training_step(
            iteration=i,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=max(0, entropy),
            mean_reward=mean_reward,
            hands_played=100,
            steps_collected=200,
            bb_per_100=mean_reward / 100 if i % 10 == 0 else None,
        )

        # Simulate some actions
        for _ in range(50):
            action = np.random.choice(11, p=[0.1, 0.3, 0.15, 0.1, 0.1, 0.08, 0.07, 0.04, 0.03, 0.02, 0.01])
            street = np.random.randint(0, 4)
            position = np.random.randint(0, 2)
            viz.log_action(action, street, position)

        # Simulate hand results
        for _ in range(20):
            position = np.random.randint(0, 2)
            street = np.random.randint(0, 5)
            profit = np.random.normal(0, 500)
            won = profit > 0
            viz.log_hand_result(position, street, profit, won)

    # Generate outputs
    print("\nGenerating outputs...")
    viz.save_training_plots()
    viz.create_action_heatmap()
    viz.create_position_analysis()

    report = viz.generate_report()
    print(report)

    viz.close()
    print("\nTest complete!")
