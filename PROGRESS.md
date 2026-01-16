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

## Phase 4: Personality Analysis & Opponent Modeling (COMPLETED)

### Summary
Comprehensive player statistics tracking, automatic personality classification, and opponent modeling system for exploitative play.

### Features

1. **HUD-Style Player Statistics**
   - VPIP (Voluntarily Put $ In Pot)
   - PFR (Pre-Flop Raise)
   - 3-bet% / Fold to 3-bet
   - C-bet% / Fold to C-bet
   - Aggression Factor (AF) & Frequency
   - WTSD (Went To ShowDown) / W$SD (Won $ at ShowDown)
   - WWSF (Won When Saw Flop)
   - BB/100 performance tracking
   - Position-based statistics

2. **Automatic Personality Classification**
   - MANIAC: Very loose, very aggressive (VPIP>50%, AF>3)
   - LAG: Loose-aggressive (VPIP 30-50%, AF>2)
   - TAG: Tight-aggressive (VPIP 15-30%, AF>2)
   - ROCK: Very tight, passive (VPIP<15%, AF<1.5)
   - NIT: Ultra-tight (VPIP<12%)
   - FISH: Loose, passive (VPIP>40%, AF<1.5)
   - CALLING_STATION: Calls too much, rarely raises
   - BALANCED: GTO-ish balanced play
   - Confidence scoring based on sample size

3. **Opponent Modeling System**
   - Real-time statistics tracking
   - Per-player exploit recommendations
   - Session import/export for persistence
   - Multi-player comparison

4. **Agent Analyzer**
   - ELO rating system with head-to-head tracking
   - Leaderboard generation
   - Matchup matrix (profit per 100 hands)
   - Detailed comparison reports

5. **Tournament Integration**
   - Automatic personality tracking during matches
   - HUD display for each participant
   - Exploit advice generation
   - Enhanced result exports with personality data

### Files Added

**`brain/personality_analysis.py`** (~800 lines)
- `PlayerStatistics` - HUD-style statistics tracking
- `HandResult` / `HandAction` - Hand recording dataclasses
- `OpponentModel` - Real-time opponent tracking
- `HandTracker` - Utility for game loop integration
- `AgentAnalyzer` - Agent comparison and ELO tracking
- `AgentProfile` - Extended agent profile with personality
- Automatic personality classification based on observed stats
- Exploit recommendation generator

### Files Modified

**`brain/tournament.py`**
- Added personality analysis imports
- `Tournament` class now tracks personality during matches
- Added `get_personality_report()` method
- Added `get_player_hud()` method
- Added `get_exploit_advice()` method
- Added `get_agent_comparison()` method
- Enhanced `save()` to include personality data
- Integrated `OpponentModel` and `AgentAnalyzer`

### Usage

```python
from brain.personality_analysis import OpponentModel, HandTracker, AgentAnalyzer

# Real-time opponent modeling
model = OpponentModel()
# ... track hands during play ...
print(model.get_summary_report())

# Get HUD stats for a player
stats = model.get_player_stats("Player_1")
print(stats.get_hud_display())
# Output: [Player_1] VPIP:25% PFR:18% AF:2.5 3B:8% CB:70% WTSD:35% (500h)

# Get exploit recommendations
recs = model.get_exploit_recommendations("Player_1")
# Returns: {"value_betting": "...", "bluffing": "...", ...}

# Agent comparison
analyzer = AgentAnalyzer()
analyzer.register_agent("Neural_v1", "neural")
analyzer.register_agent("TAG_Bot", "rule_based")
# ... run matches ...
print(analyzer.get_leaderboard())
print(analyzer.generate_comparison_report(["Neural_v1", "TAG_Bot"]))
```

### Tournament with Personality Tracking

```python
from brain.tournament import Tournament

tournament = Tournament(
    name="Championship",
    hands_per_match=500,
    enable_personality_tracking=True,  # Enable Phase 4 features
)

# Add participants
tournament.add_participant("TAG_Bot", tag_agent, agent_type="rule_based")
tournament.add_participant("Neural_v1", neural_agent, agent_type="neural")

# Run tournament
tournament.run(env, get_action_fn)

# Get analysis
print(tournament.get_personality_report())
print(tournament.get_player_hud("TAG_Bot"))
print(tournament.get_exploit_advice("Neural_v1"))
print(tournament.get_agent_comparison())
```

### Sample Output

```
[TAG_Bot] VPIP:22% PFR:18% AF:2.8 3B:9% CB:72% WTSD:32% (500h)
  Personality: tag (100% confidence)
  Profit: +1250 chips (+12.5 BB/100)

Exploit Recommendations for FISH_Player:
  - value_betting: Bet larger for value - this player calls too much
  - bluffing: Reduce bluff frequency - this player rarely folds
  - hand_selection: Play more hands in position - they'll pay off your value
```

---

## Next Steps

### Phase 5: Advanced Features (TODO)
- [ ] Neural network opponent modeling
- [ ] GTO vs Exploitative mode switching
- [ ] Hand range visualization
- [ ] ICM calculations for tournaments
- [ ] Multi-table tournament support

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
