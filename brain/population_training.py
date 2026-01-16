"""Population-Based Training (PBT) for poker AI.

This module provides:
1. Population management for multiple learning agents
2. Hyperparameter mutation and crossover
3. Fitness-based selection and evolution
4. Training against diverse opponent pools
"""

from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
import os
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy
import time

try:
    from .personality_agents import (
        AgentPool,
        RuleBasedAgent,
        PersonalityType,
        create_pool,
        create_all_pools,
    )
    from .multi_player_features import MultiPlayerFeatureExtractor
    from .networks import ActorCriticNetwork
except ImportError:
    from personality_agents import (
        AgentPool,
        RuleBasedAgent,
        PersonalityType,
        create_pool,
        create_all_pools,
    )
    from multi_player_features import MultiPlayerFeatureExtractor
    from networks import ActorCriticNetwork


@dataclass
class HyperParameters:
    """Hyperparameters for a learning agent."""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_dims: List[int] = field(default_factory=lambda: [128])
    critic_dims: List[int] = field(default_factory=lambda: [128])

    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3

    # PPO parameters
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    gae_lambda: float = 0.95
    gamma: float = 0.99

    # Training
    batch_size: int = 64
    epochs_per_update: int = 4
    rollout_length: int = 512
    max_grad_norm: float = 0.5

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995

    def mutate(self, mutation_rate: float = 0.1, rng: np.random.RandomState = None) -> 'HyperParameters':
        """Create mutated copy of hyperparameters.

        Args:
            mutation_rate: Probability of mutating each parameter
            rng: Random number generator

        Returns:
            New HyperParameters with mutations
        """
        if rng is None:
            rng = np.random.RandomState()

        new_hp = copy.deepcopy(self)

        # Mutate continuous parameters
        continuous_params = [
            ('lr_actor', 1e-5, 1e-2),
            ('lr_critic', 1e-5, 1e-2),
            ('clip_ratio', 0.1, 0.4),
            ('entropy_coeff', 0.001, 0.1),
            ('value_coeff', 0.1, 1.0),
            ('gae_lambda', 0.9, 0.99),
            ('gamma', 0.95, 0.999),
            ('max_grad_norm', 0.1, 1.0),
        ]

        for param_name, min_val, max_val in continuous_params:
            if rng.random() < mutation_rate:
                current = getattr(new_hp, param_name)
                # Multiplicative mutation
                multiplier = rng.uniform(0.8, 1.2)
                new_val = np.clip(current * multiplier, min_val, max_val)
                setattr(new_hp, param_name, new_val)

        # Mutate discrete parameters
        if rng.random() < mutation_rate:
            new_hp.batch_size = int(2 ** rng.randint(5, 9))  # 32-256

        if rng.random() < mutation_rate:
            new_hp.epochs_per_update = rng.randint(2, 8)

        if rng.random() < mutation_rate:
            new_hp.rollout_length = int(2 ** rng.randint(8, 11))  # 256-1024

        return new_hp

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'HyperParameters':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class AgentRecord:
    """Record of an agent's performance."""
    agent_id: str
    generation: int = 0
    total_hands: int = 0
    total_profit: float = 0.0
    win_rate: float = 0.0
    bb_per_100: float = 0.0
    elo_rating: float = 1200.0
    parent_ids: List[str] = field(default_factory=list)
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    fitness_history: List[float] = field(default_factory=list)

    def update_fitness(self, profit: float, num_hands: int, bb: int = 100):
        """Update fitness metrics after evaluation.

        Args:
            profit: Total profit in chips
            num_hands: Number of hands played
            bb: Big blind size
        """
        self.total_profit += profit
        self.total_hands += num_hands
        self.bb_per_100 = self.total_profit / max(self.total_hands / 100, 1) / bb

        # Track fitness history
        recent_bb100 = profit / (num_hands / 100) / bb if num_hands > 0 else 0
        self.fitness_history.append(recent_bb100)
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]

    @property
    def fitness(self) -> float:
        """Current fitness score."""
        if not self.fitness_history:
            return self.elo_rating / 1000  # Default to ELO-based

        # Weighted average of recent performance
        weights = np.linspace(0.5, 1.0, len(self.fitness_history))
        return float(np.average(self.fitness_history, weights=weights))


class SelectionMethod(Enum):
    """Methods for selecting agents for reproduction."""
    TOURNAMENT = "tournament"    # Tournament selection
    TRUNCATION = "truncation"    # Top k selection
    ROULETTE = "roulette"        # Fitness proportional
    ELITISM = "elitism"          # Keep best, replace rest


class Population:
    """Population of learning agents for PBT."""

    def __init__(
        self,
        population_size: int = 16,
        num_players: int = 6,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.3,
        elite_fraction: float = 0.2,
        seed: Optional[int] = None,
    ):
        """Initialize population.

        Args:
            population_size: Number of agents in population
            num_players: Number of players per game
            selection_method: How to select parents
            mutation_rate: Probability of parameter mutation
            crossover_rate: Probability of crossover
            elite_fraction: Fraction of population to preserve
            seed: Random seed
        """
        self.population_size = population_size
        self.num_players = num_players
        self.selection_method = selection_method
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.rng = np.random.RandomState(seed)

        self.generation = 0
        self.agents: Dict[str, AgentRecord] = {}
        self.agent_networks: Dict[str, Any] = {}  # Store actual networks

        self._initialize_population()

    def _initialize_population(self):
        """Create initial population with diverse hyperparameters."""
        for i in range(self.population_size):
            agent_id = f"agent_{i:03d}_gen{self.generation:03d}"

            # Create hyperparameters with some diversity
            hp = HyperParameters()
            if i > 0:  # First agent uses defaults
                hp = hp.mutate(mutation_rate=0.5, rng=self.rng)

            record = AgentRecord(
                agent_id=agent_id,
                generation=self.generation,
                hyperparameters=hp,
            )
            self.agents[agent_id] = record

    def get_agent_ids(self) -> List[str]:
        """Get all agent IDs in population."""
        return list(self.agents.keys())

    def get_agent(self, agent_id: str) -> AgentRecord:
        """Get agent record by ID."""
        return self.agents[agent_id]

    def evaluate_agents(
        self,
        env,
        opponent_pools: Dict[str, AgentPool],
        num_hands: int = 500,
        get_agent_action_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Evaluate all agents against opponent pools.

        Args:
            env: Poker environment
            opponent_pools: Dictionary of opponent pools
            num_hands: Hands per evaluation
            get_agent_action_fn: Function to get action from learning agent

        Returns:
            Dictionary of agent_id -> fitness
        """
        fitness_scores = {}

        for agent_id, record in self.agents.items():
            total_profit = 0.0
            hands_played = 0

            # Play against each opponent pool
            for pool_name, pool in opponent_pools.items():
                opponents = pool.sample_agents(self.num_players - 1)

                for _ in range(num_hands // len(opponent_pools)):
                    profit, num = self._play_hand(
                        env, agent_id, opponents, get_agent_action_fn
                    )
                    total_profit += profit
                    hands_played += num

            # Update record
            record.update_fitness(total_profit, hands_played)
            fitness_scores[agent_id] = record.fitness

        return fitness_scores

    def _play_hand(
        self,
        env,
        agent_id: str,
        opponents: List[RuleBasedAgent],
        get_agent_action_fn: Optional[Callable],
    ) -> Tuple[float, int]:
        """Play a single hand and return profit.

        Args:
            env: Poker environment
            agent_id: ID of learning agent
            opponents: List of opponent agents
            get_agent_action_fn: Function to get action

        Returns:
            Tuple of (profit, num_hands)
        """
        obs = env.reset()
        learning_player = 0  # Learning agent at seat 0

        while not obs["is_terminal"]:
            current = obs["current_player"]

            if current == learning_player:
                if get_agent_action_fn is not None:
                    action = get_agent_action_fn(agent_id, obs, env)
                else:
                    # Default to random
                    mask = obs.get("action_mask", [True] * 4)
                    valid = [i for i, v in enumerate(mask) if v]
                    action = self.rng.choice(valid)
            else:
                opp_idx = current - 1 if current > learning_player else current
                if opp_idx < len(opponents):
                    action = opponents[opp_idx].get_action(obs, env)
                else:
                    mask = obs.get("action_mask", [True] * 4)
                    valid = [i for i, v in enumerate(mask) if v]
                    action = self.rng.choice(valid)

            obs, rewards, done, _, _ = env.step(action)

        return rewards[learning_player], 1

    def select_parents(self, num_parents: int) -> List[str]:
        """Select parents for reproduction.

        Args:
            num_parents: Number of parents to select

        Returns:
            List of parent agent IDs
        """
        agent_ids = list(self.agents.keys())
        fitness_values = [self.agents[aid].fitness for aid in agent_ids]

        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(agent_ids, fitness_values, num_parents)
        elif self.selection_method == SelectionMethod.TRUNCATION:
            return self._truncation_selection(agent_ids, fitness_values, num_parents)
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(agent_ids, fitness_values, num_parents)
        else:  # ELITISM
            return self._elitism_selection(agent_ids, fitness_values, num_parents)

    def _tournament_selection(
        self,
        agent_ids: List[str],
        fitness_values: List[float],
        num_parents: int,
        tournament_size: int = 3,
    ) -> List[str]:
        """Tournament selection."""
        parents = []
        for _ in range(num_parents):
            # Random tournament
            candidates = self.rng.choice(len(agent_ids), size=tournament_size, replace=False)
            winner = max(candidates, key=lambda i: fitness_values[i])
            parents.append(agent_ids[winner])
        return parents

    def _truncation_selection(
        self,
        agent_ids: List[str],
        fitness_values: List[float],
        num_parents: int,
    ) -> List[str]:
        """Select top performers."""
        sorted_indices = np.argsort(fitness_values)[::-1]
        return [agent_ids[i] for i in sorted_indices[:num_parents]]

    def _roulette_selection(
        self,
        agent_ids: List[str],
        fitness_values: List[float],
        num_parents: int,
    ) -> List[str]:
        """Fitness-proportional selection."""
        # Shift fitness to be positive
        min_fitness = min(fitness_values)
        shifted = [f - min_fitness + 1.0 for f in fitness_values]
        total = sum(shifted)
        probs = [f / total for f in shifted]

        indices = self.rng.choice(len(agent_ids), size=num_parents, p=probs, replace=True)
        return [agent_ids[i] for i in indices]

    def _elitism_selection(
        self,
        agent_ids: List[str],
        fitness_values: List[float],
        num_parents: int,
    ) -> List[str]:
        """Keep elite, sample from remaining."""
        num_elite = max(1, int(len(agent_ids) * self.elite_fraction))
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Always include elite
        elite = [agent_ids[i] for i in sorted_indices[:num_elite]]

        # Sample remaining from all
        remaining = num_parents - len(elite)
        if remaining > 0:
            sampled = self.rng.choice(agent_ids, size=remaining, replace=True)
            return elite + list(sampled)
        return elite[:num_parents]

    def evolve(self) -> 'Population':
        """Create next generation through selection and mutation.

        Returns:
            New population (modifies self in place)
        """
        self.generation += 1

        # Get number of elite to preserve
        num_elite = max(1, int(self.population_size * self.elite_fraction))

        # Sort by fitness
        sorted_agents = sorted(
            self.agents.items(),
            key=lambda x: x[1].fitness,
            reverse=True
        )

        # Keep elite unchanged
        new_agents = {}
        for i, (agent_id, record) in enumerate(sorted_agents[:num_elite]):
            new_id = f"agent_{i:03d}_gen{self.generation:03d}"
            new_record = copy.deepcopy(record)
            new_record.agent_id = new_id
            new_record.generation = self.generation
            new_agents[new_id] = new_record

        # Create children for remaining slots
        num_children = self.population_size - num_elite
        parents = self.select_parents(num_children * 2)

        for i in range(num_children):
            child_idx = num_elite + i
            child_id = f"agent_{child_idx:03d}_gen{self.generation:03d}"

            # Get two parents
            parent1_id = parents[i * 2 % len(parents)]
            parent2_id = parents[(i * 2 + 1) % len(parents)]
            parent1 = self.agents[parent1_id]
            parent2 = self.agents[parent2_id]

            # Crossover hyperparameters
            if self.rng.random() < self.crossover_rate:
                child_hp = self._crossover_hp(
                    parent1.hyperparameters,
                    parent2.hyperparameters
                )
            else:
                child_hp = copy.deepcopy(parent1.hyperparameters)

            # Mutate
            child_hp = child_hp.mutate(self.mutation_rate, self.rng)

            # Create child record
            child_record = AgentRecord(
                agent_id=child_id,
                generation=self.generation,
                hyperparameters=child_hp,
                parent_ids=[parent1_id, parent2_id],
                elo_rating=(parent1.elo_rating + parent2.elo_rating) / 2,
            )
            new_agents[child_id] = child_record

        self.agents = new_agents
        return self

    def _crossover_hp(self, hp1: HyperParameters, hp2: HyperParameters) -> HyperParameters:
        """Crossover two hyperparameter sets.

        Args:
            hp1: First parent hyperparameters
            hp2: Second parent hyperparameters

        Returns:
            Child hyperparameters
        """
        child = HyperParameters()

        # Uniform crossover for each parameter
        for field_name in ['lr_actor', 'lr_critic', 'clip_ratio', 'entropy_coeff',
                          'value_coeff', 'gae_lambda', 'gamma', 'batch_size',
                          'epochs_per_update', 'rollout_length', 'max_grad_norm']:
            if self.rng.random() < 0.5:
                setattr(child, field_name, getattr(hp1, field_name))
            else:
                setattr(child, field_name, getattr(hp2, field_name))

        return child

    def get_best_agent(self) -> Tuple[str, AgentRecord]:
        """Get the best performing agent.

        Returns:
            Tuple of (agent_id, record)
        """
        best_id = max(self.agents.keys(), key=lambda k: self.agents[k].fitness)
        return best_id, self.agents[best_id]

    def get_statistics(self) -> Dict:
        """Get population statistics.

        Returns:
            Dictionary of statistics
        """
        fitness_values = [r.fitness for r in self.agents.values()]
        elo_values = [r.elo_rating for r in self.agents.values()]

        return {
            "generation": self.generation,
            "population_size": len(self.agents),
            "fitness_mean": float(np.mean(fitness_values)),
            "fitness_std": float(np.std(fitness_values)),
            "fitness_max": float(np.max(fitness_values)),
            "fitness_min": float(np.min(fitness_values)),
            "elo_mean": float(np.mean(elo_values)),
            "elo_max": float(np.max(elo_values)),
        }

    def save(self, path: str):
        """Save population to file.

        Args:
            path: Path to save file
        """
        data = {
            "generation": self.generation,
            "population_size": self.population_size,
            "num_players": self.num_players,
            "selection_method": self.selection_method.value,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elite_fraction": self.elite_fraction,
            "agents": {
                aid: {
                    "agent_id": r.agent_id,
                    "generation": r.generation,
                    "total_hands": r.total_hands,
                    "total_profit": r.total_profit,
                    "bb_per_100": r.bb_per_100,
                    "elo_rating": r.elo_rating,
                    "parent_ids": r.parent_ids,
                    "hyperparameters": r.hyperparameters.to_dict(),
                    "fitness_history": r.fitness_history[-20:],  # Keep last 20
                }
                for aid, r in self.agents.items()
            }
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Population':
        """Load population from file.

        Args:
            path: Path to load file

        Returns:
            Loaded Population
        """
        with open(path, 'r') as f:
            data = json.load(f)

        pop = cls(
            population_size=data["population_size"],
            num_players=data["num_players"],
            selection_method=SelectionMethod(data["selection_method"]),
            mutation_rate=data["mutation_rate"],
            crossover_rate=data["crossover_rate"],
            elite_fraction=data["elite_fraction"],
        )

        pop.generation = data["generation"]
        pop.agents = {}

        for aid, agent_data in data["agents"].items():
            hp = HyperParameters.from_dict(agent_data["hyperparameters"])
            record = AgentRecord(
                agent_id=agent_data["agent_id"],
                generation=agent_data["generation"],
                total_hands=agent_data["total_hands"],
                total_profit=agent_data["total_profit"],
                bb_per_100=agent_data["bb_per_100"],
                elo_rating=agent_data["elo_rating"],
                parent_ids=agent_data["parent_ids"],
                hyperparameters=hp,
                fitness_history=agent_data["fitness_history"],
            )
            pop.agents[aid] = record

        return pop


class PBTTrainer:
    """Population-Based Training orchestrator."""

    def __init__(
        self,
        population: Population,
        opponent_pools: Dict[str, AgentPool],
        env,
        eval_interval: int = 100,
        evolve_interval: int = 500,
        save_dir: str = "pbt_checkpoints",
    ):
        """Initialize PBT trainer.

        Args:
            population: Population to train
            opponent_pools: Opponent pools for evaluation
            env: Poker environment
            eval_interval: Steps between evaluations
            evolve_interval: Steps between evolution
            save_dir: Directory for checkpoints
        """
        self.population = population
        self.opponent_pools = opponent_pools
        self.env = env
        self.eval_interval = eval_interval
        self.evolve_interval = evolve_interval
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.global_step = 0
        self.training_history: List[Dict] = []

    def train(
        self,
        num_generations: int = 10,
        eval_hands: int = 500,
        verbose: bool = True,
    ):
        """Run PBT training loop.

        Args:
            num_generations: Number of generations to train
            eval_hands: Hands per evaluation
            verbose: Print progress
        """
        for gen in range(num_generations):
            start_time = time.time()

            # Evaluate all agents
            fitness_scores = self.population.evaluate_agents(
                self.env,
                self.opponent_pools,
                num_hands=eval_hands,
            )

            # Get statistics
            stats = self.population.get_statistics()
            stats["time"] = time.time() - start_time

            self.training_history.append(stats)

            if verbose:
                best_id, best_agent = self.population.get_best_agent()
                print(
                    f"Gen {gen+1}/{num_generations} | "
                    f"Best: {best_agent.fitness:.2f} BB/100 | "
                    f"Mean: {stats['fitness_mean']:.2f} | "
                    f"Time: {stats['time']:.1f}s"
                )

            # Evolve population
            self.population.evolve()

            # Save checkpoint
            if (gen + 1) % 5 == 0:
                self.save_checkpoint(f"gen_{gen+1:03d}")

        # Final save
        self.save_checkpoint("final")

    def save_checkpoint(self, name: str):
        """Save training checkpoint.

        Args:
            name: Checkpoint name
        """
        self.population.save(os.path.join(self.save_dir, f"{name}_population.json"))

        history_path = os.path.join(self.save_dir, f"{name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


# Testing
if __name__ == "__main__":
    import poker_engine

    print("=== Population-Based Training Test ===\n")

    # Create environment
    env = poker_engine.PokerEnv(
        num_players=6,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    # Create opponent pools
    opponent_pools = {
        "mixed": create_pool("mixed_pool", pool_size=10, seed=42),
        "fish": create_pool("fish_pool", pool_size=10, seed=43),
    }

    # Create population
    population = Population(
        population_size=8,
        num_players=6,
        selection_method=SelectionMethod.TOURNAMENT,
        mutation_rate=0.15,
        seed=12345,
    )

    print(f"Created population with {len(population.agents)} agents")
    print(f"Initial statistics: {population.get_statistics()}")

    # Create trainer
    trainer = PBTTrainer(
        population=population,
        opponent_pools=opponent_pools,
        env=env,
        save_dir="/tmp/pbt_test",
    )

    # Run mini training (2 generations for testing)
    print("\nRunning 2 generations of PBT...")
    trainer.train(num_generations=2, eval_hands=100, verbose=True)

    # Check results
    best_id, best_agent = population.get_best_agent()
    print(f"\nBest agent: {best_id}")
    print(f"  Generation: {best_agent.generation}")
    print(f"  Fitness: {best_agent.fitness:.2f}")
    print(f"  Hyperparameters: lr_actor={best_agent.hyperparameters.lr_actor:.2e}")

    # Test save/load
    population.save("/tmp/pbt_test/test_save.json")
    loaded = Population.load("/tmp/pbt_test/test_save.json")
    print(f"\nLoaded population: {len(loaded.agents)} agents, gen {loaded.generation}")

    print("\n✓ All tests passed!")
