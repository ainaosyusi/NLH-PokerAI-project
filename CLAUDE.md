# Project Nash-Evo - Poker AI

High-performance No-Limit Texas Hold'em poker AI with Rust game engine and Python ML components.

## Quick Start

```bash
cd ~/poker-ai-project
source .venv/bin/activate

# Play against AI (terminal)
python play_poker.py

# Play against AI (web browser)
python ui/web_server.py
# Open http://localhost:8000

# Rebuild Rust engine after changes
cd core_engine && maturin develop --release && cd ..
```

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
│   ├── personality_agents.py  # Rule-based agents (TAG, LAG, Rock, Fish, Maniac)
│   ├── population_training.py # Population-based training
│   ├── tournament.py      # Tournament/league system with ELO
│   ├── multi_player_features.py # Multi-player feature extraction
│   └── distributed_training.py  # Ray distributed training (optional)
├── ui/                    # User interfaces
│   ├── terminal_ui.py     # Terminal-based UI with Unicode cards
│   ├── game_controller.py # Human vs AI game loop
│   ├── hand_history.py    # Hand history viewer/replay
│   └── web_server.py      # FastAPI web UI
├── play_poker.py          # Main entry point for playing
└── .venv/                 # Python virtual environment
```

## Key Components

### Rust Engine (`poker_engine` module)
- **Performance**: ~1.9M hands/sec (2 players), ~480K hands/sec (6 players)
- **Players**: 2-9 supported
- **API**: `poker_engine.PokerEnv(num_players, small_blind, big_blind, starting_stack)`

```python
import poker_engine
env = poker_engine.PokerEnv(num_players=6, small_blind=50, big_blind=100, starting_stack=10000)
obs = env.reset()
obs, rewards, done, truncated, info = env.step(action)  # action: 0=fold, 1=call, 2=raise, 3=allin
```

### AI Agents
- **Personality types**: TAG, LAG, Rock, Fish, Maniac, Balanced
- **Training algorithms**: Deep CFR, PPO with self-play
- **Population-based training** with hyperparameter evolution

### Training
```bash
# Train PPO agent
python brain/train.py --algorithm ppo --iterations 500

# Train Deep CFR
python brain/train.py --algorithm deep_cfr --iterations 500

# Compare algorithms
python brain/train.py --compare
```

## Development Commands

```bash
# Activate environment
source .venv/bin/activate

# Run tests
python brain/sixmax_training.py    # Phase 3 tests
python ui/test_phase4.py           # Phase 4 tests

# Rebuild Rust engine
cd core_engine && maturin develop --release

# Benchmark
python -c "import poker_engine; print(poker_engine.benchmark_random_hands(100000, 6, 12345))"
```

## Architecture Notes

- **Phase 1**: Rust game engine with PyO3 Python bindings
- **Phase 2**: Deep CFR and PPO training algorithms
- **Phase 3**: Multi-player support, personality pools, PBT, tournaments
- **Phase 4**: Terminal and Web UI for human play

## Important Files

| File | Purpose |
|------|---------|
| `core_engine/src/game.rs` | Core game logic, state machine |
| `core_engine/src/evaluator.rs` | Hand evaluation tables |
| `brain/personality_agents.py` | AI opponent personalities |
| `brain/ppo_agent.py` | PPO training implementation |
| `ui/game_controller.py` | Human vs AI game loop |
| `play_poker.py` | Main entry point |

## Dependencies

- Python 3.10+
- Rust (via rustup)
- PyTorch, NumPy, Gymnasium
- FastAPI, Uvicorn (for web UI)
- maturin (for Rust-Python binding)
