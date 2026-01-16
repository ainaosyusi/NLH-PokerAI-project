# Poker AI Project - Development Progress

## Overview

High-performance No-Limit Texas Hold'em poker AI with Rust game engine and Python ML components.

---

## Phase 1: Pot-Geometric Action Abstraction (COMPLETED)

### Summary
Expanded the action space from 4 actions (Fold/Call/Raise/AllIn) to 11 actions with pot-geometric sizing for more realistic poker play.

### Action Mapping

| ID | Action | Description |
|----|--------|-------------|
| 0 | Fold | Give up the hand |
| 1 | Check/Call | Match the current bet |
| 2 | Min Raise | Minimum legal raise |
| 3 | 33% Pot | Bet/raise 33% of pot |
| 4 | 50% Pot | Bet/raise 50% of pot |
| 5 | 75% Pot | Bet/raise 75% of pot |
| 6 | 100% Pot | Pot-size bet/raise |
| 7 | 125% Pot | Bet/raise 125% of pot |
| 8 | 150% Pot | Bet/raise 150% of pot |
| 9 | 200% Pot | Bet/raise 200% of pot |
| 10 | All-in | Bet all remaining chips |

### Files Modified

**`core_engine/src/game.rs`**
- Added `calc_pot_fraction(fraction)` - calculates bet amount as % of pot
- Added `calc_min_raise()` - returns minimum raise amount
- Added `step_with_action_id(action_id)` - handles 0-10 action IDs
- Added `pot_fraction_action()` - helper for pot-fraction actions
- Added `is_valid_action_id()` - validates action IDs
- Added `get_action_mask_11()` - returns 11-boolean action mask

**`core_engine/src/lib.rs`**
- Updated `step()` to accept 0-10 pot-geometric actions
- Updated `action_mask` to return 11 booleans
- Added `legacy_action_mask` for backwards compatibility
- Added `NUM_ACTIONS=11` and `ACTION_NAMES` constants
- Added `num_actions` property

### Performance
- ~1.8M hands/sec (2 players) - unchanged from 4-action version
- Backwards compatible with tuple action format

### Usage Example
```python
import poker_engine

env = poker_engine.PokerEnv(num_players=2)
obs = env.reset()

# Get valid actions (11-element boolean mask)
action_mask = obs['action_mask']  # [True, True, True, ...]

# Take action with pot-geometric sizing
obs, rewards, done, _, info = env.step(4)  # 50% pot bet
```

---

## Phase 2: Ray Distributed Training (COMPLETED)

### Summary
Implemented full Ray-based distributed training system with parallel rollout collection and centralized PPO training.

### Features

1. **RayPPOTrainer** - Main distributed trainer class
   - Parallel rollout collection across Ray workers
   - Centralized PPO training with weight synchronization
   - Support for 11-action pot-geometric system
   - Automatic fallback to local workers if Ray unavailable

2. **RayWorker** - Remote Ray actor for parallel data collection
   - Loads model weights from central learner
   - Collects rollouts with proper feature extraction
   - Supports both training and evaluation modes

3. **TrainingLogger** - Unified logging interface
   - TensorBoard integration for training curves
   - WandB support (optional)
   - Automatic metric logging

4. **TrainingConfig** - Comprehensive configuration
   - Game settings (blinds, stacks)
   - PPO hyperparameters (lr, gamma, clip_epsilon, etc.)
   - Network architecture configuration
   - Training settings (iterations, eval frequency)

### Files Modified

**`brain/distributed_training.py`** - Complete rewrite (~1000 lines)
- `RayPPOTrainer` class
- `RayWorker` Ray actor
- `LocalWorker` for non-Ray fallback
- `TrainingLogger` for TensorBoard/WandB
- `TrainingConfig` dataclass
- `RolloutData` dataclass with serialization
- CLI interface with argparse

### Usage

```bash
# Quick test
python brain/distributed_training.py --test

# Full training with 4 workers
python brain/distributed_training.py --workers 4 --iterations 500

# With TensorBoard logging
python brain/distributed_training.py --workers 4 --log-dir logs/my_run

# View TensorBoard
tensorboard --logdir logs/
```

### CLI Options
```
--workers       Number of parallel workers (default: 4)
--players       Players per table (default: 6)
--iterations    Training iterations (default: 500)
--hands         Hands per rollout per worker (default: 100)
--lr            Learning rate (default: 3e-4)
--batch-size    Batch size (default: 256)
--log-dir       TensorBoard log directory
--save-dir      Model save directory
--seed          Random seed
--no-tensorboard  Disable TensorBoard
--wandb         Enable WandB logging
--wandb-project WandB project name
```

### Performance
- ~2300 hands/sec with 2 Ray workers
- Scalable to many workers
- Efficient weight synchronization via Ray object store

---

## Project Structure

```
poker-ai-project/
├── core_engine/           # Rust poker engine (PyO3 bindings)
│   ├── src/
│   │   ├── lib.rs         # Python bindings, PokerEnv class
│   │   ├── cards.rs       # Bitboard card representation
│   │   ├── evaluator.rs   # Hand evaluation (PHE algorithm)
│   │   ├── game.rs        # Game state machine, betting logic
│   │   └── rng.rs         # XORShift64* RNG
│   └── Cargo.toml
├── brain/                 # Python AI components
│   ├── abstraction.py     # Card/action abstraction, feature extraction
│   ├── networks.py        # Neural network architectures (PyTorch)
│   ├── deep_cfr.py        # Deep CFR algorithm
│   ├── ppo_agent.py       # PPO self-play agent
│   ├── personality_agents.py  # Rule-based agents
│   ├── population_training.py # Population-based training
│   ├── tournament.py      # Tournament/league system with ELO
│   ├── multi_player_features.py # Multi-player feature extraction
│   └── distributed_training.py  # Ray distributed training
├── ui/                    # User interfaces
│   ├── terminal_ui.py     # Terminal-based UI
│   ├── game_controller.py # Human vs AI game loop
│   ├── hand_history.py    # Hand history viewer
│   └── web_server.py      # FastAPI web UI
├── play_poker.py          # Main entry point for playing
├── CLAUDE.md              # Project documentation for AI
├── PROGRESS.md            # This file
└── .gitignore
```

---

## Dependencies

- Python 3.10+
- Rust (via rustup)
- PyTorch, NumPy, Gymnasium
- Ray (for distributed training)
- TensorBoard (for logging)
- FastAPI, Uvicorn (for web UI)
- maturin (for Rust-Python binding)

---

## Phase 3: Learning Visualization & Monitoring (COMPLETED)

### Summary
Comprehensive visualization and monitoring system for training progress, action distributions, and performance metrics.

### Features

1. **Training Metrics Dashboard**
   - Policy/Value loss curves
   - Entropy tracking
   - Mean reward over time
   - BB/100 performance
   - Hands/second throughput
   - Cumulative hands played

2. **Action Distribution Analysis**
   - Overall action frequencies
   - Actions by street (Preflop/Flop/Turn/River)
   - Actions by position
   - Heatmap visualizations

3. **Win Rate Tracking**
   - Win rate by position
   - Profit/loss by position
   - Performance by street reached
   - Rolling window statistics

4. **Exploitability Metrics**
   - Best response value estimates
   - Nash distance tracking
   - Cumulative regret monitoring

5. **Output Formats**
   - TensorBoard integration
   - WandB support (optional)
   - PNG plot exports
   - JSON metrics export
   - Text training reports

### Files Added

**`brain/visualization.py`** (~800 lines)
- `PokerVisualizer` - Main visualization class
- `TrainingMetrics` - Container for training data
- `ActionStats` - Action distribution tracking
- `WinRateStats` - Win rate statistics
- `ExploitabilityMetrics` - Exploitability tracking
- `LiveDashboard` - Real-time matplotlib dashboard
- `create_visualizer_for_training()` - Factory function

### Files Modified

**`brain/distributed_training.py`**
- Integrated `PokerVisualizer` into `RayPPOTrainer`
- Added street tracking to `RolloutData`
- Added hand result recording for visualization
- Auto-generates plots and reports on shutdown

### Generated Outputs

```
logs/
├── events.out.tfevents.*    # TensorBoard logs
├── metrics.json             # JSON metrics export
├── training_report.txt      # Text summary report
└── plots/
    ├── training_progress.png   # Training dashboard
    ├── action_heatmap.png      # Action by street
    └── position_analysis.png   # Position statistics
```

### Usage

```bash
# Training with visualization (default)
python brain/distributed_training.py --workers 4 --log-dir logs/my_run

# View TensorBoard
tensorboard --logdir logs/

# Standalone visualization test
python brain/visualization.py
```

### Sample Output

```
============================================================
POKER AI TRAINING REPORT
============================================================

## Training Summary
Total Iterations: 100
Total Hands Played: 5,000
Training Time: 2.5 minutes
Average Speed: 2,000 hands/sec

## Loss Metrics (Latest)
Policy Loss: 0.045
Value Loss: 0.125
Entropy: 1.85

## Performance
Recent Mean Reward: +125.5
Recent BB/100: +2.5

## Action Distribution (Overall)
  Fold: 15.2%
  Check/Call: 35.1%
  Min Raise: 12.3%
  50% Pot: 15.4%
  100% Pot: 10.2%
  All-in: 5.8%

## Position Statistics
  Button/SB: +45.2 chips/hand
  BB: -12.3 chips/hand
============================================================
```

---

## Next Steps

### Phase 4: Personality Quantification & League System
- [ ] Player personality profiling (VPIP, PFR, AF)
- [ ] ELO rating system improvements
- [ ] Tournament simulation
- [ ] Agent comparison tools
- [ ] Opponent modeling integration

---

## Quick Start

```bash
cd ~/poker-ai-project
source .venv/bin/activate

# Play against AI (terminal)
python play_poker.py

# Play against AI (web browser)
python ui/web_server.py
# Open http://localhost:8000

# Train with distributed training
python brain/distributed_training.py --workers 4 --iterations 500

# Rebuild Rust engine after changes
cd core_engine && maturin develop --release && cd ..
```

---

*Last updated: 2026-01-17*
