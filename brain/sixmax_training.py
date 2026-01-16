"""6-max training with personality mix - Phase 3 integration test.

This script tests all Phase 3 components together:
1. Multi-player feature extraction
2. Personality-based opponent pools
3. Population-based training
4. Tournament evaluation
5. Distributed training (optional)
"""

from typing import List, Dict, Optional
import numpy as np
import time
import os

try:
    from .multi_player_features import (
        MultiPlayerFeatureExtractor,
        OpponentTracker,
        create_multi_player_extractor,
    )
    from .personality_agents import (
        RuleBasedAgent,
        AgentPool,
        PersonalityType,
        create_pool,
        create_all_pools,
    )
    from .population_training import (
        Population,
        PBTTrainer,
        HyperParameters,
        SelectionMethod,
    )
    from .tournament import (
        Tournament,
        League,
        default_get_action,
    )
    from .distributed_training import (
        DistributedTrainer,
        LocalWorker,
        WorkerConfig,
        merge_rollouts,
    )
except ImportError:
    from multi_player_features import (
        MultiPlayerFeatureExtractor,
        OpponentTracker,
        create_multi_player_extractor,
    )
    from personality_agents import (
        RuleBasedAgent,
        AgentPool,
        PersonalityType,
        create_pool,
        create_all_pools,
    )
    from population_training import (
        Population,
        PBTTrainer,
        HyperParameters,
        SelectionMethod,
    )
    from tournament import (
        Tournament,
        League,
        default_get_action,
    )
    from distributed_training import (
        DistributedTrainer,
        LocalWorker,
        WorkerConfig,
        merge_rollouts,
    )


def test_feature_extraction(env, num_hands: int = 10):
    """Test multi-player feature extraction."""
    print("\n=== Feature Extraction Test ===")

    extractor = create_multi_player_extractor(max_players=6, use_opponent_modeling=False)
    print(f"Feature dimension: {extractor.feature_dim}")

    for hand in range(num_hands):
        obs = env.reset()
        step = 0

        while not obs["is_terminal"]:
            current = obs["current_player"]
            features = extractor.extract_features(env, obs, current)

            # Verify feature shape
            assert features.shape[0] == extractor.feature_dim, f"Feature mismatch: {features.shape}"

            # Random action
            mask = obs.get("action_mask", [True] * 4)
            valid = [i for i, v in enumerate(mask) if v]
            action = np.random.choice(valid)

            obs, _, done, _, _ = env.step(action)
            step += 1

    print(f"✓ Extracted features for {num_hands} hands successfully")
    return True


def test_personality_pools(env, num_hands: int = 100):
    """Test personality-based agent pools."""
    print("\n=== Personality Pool Test ===")

    pools = create_all_pools(pool_size=10, seed=42)
    print(f"Created {len(pools)} pools")

    # Test each pool in a tournament
    results = {}

    for pool_name, pool in pools.items():
        agents = pool.sample_agents(6)
        total_profit = [0.0] * 6

        for hand in range(num_hands // len(pools)):
            obs = env.reset()

            while not obs["is_terminal"]:
                current = obs["current_player"]
                action = agents[current].get_action(obs, env)
                obs, rewards, done, _, _ = env.step(action)

            if done:
                for i in range(6):
                    total_profit[i] += rewards[i]

        avg_profit = sum(total_profit) / 6
        results[pool_name] = avg_profit

    print("Pool results (avg profit):")
    for name, profit in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {profit:+.0f}")

    print("✓ All pools tested successfully")
    return True


def test_population_training(env, num_generations: int = 2):
    """Test population-based training."""
    print("\n=== Population-Based Training Test ===")

    # Create opponent pools
    opponent_pools = {
        "mixed": create_pool("mixed_pool", pool_size=10, seed=100),
        "tough": create_pool("tough_pool", pool_size=10, seed=101),
    }

    # Create population
    population = Population(
        population_size=8,
        num_players=6,
        selection_method=SelectionMethod.TOURNAMENT,
        mutation_rate=0.15,
        seed=12345,
    )

    print(f"Population size: {len(population.agents)}")

    # Create trainer
    trainer = PBTTrainer(
        population=population,
        opponent_pools=opponent_pools,
        env=env,
        save_dir="/tmp/sixmax_pbt",
    )

    # Train
    trainer.train(num_generations=num_generations, eval_hands=50, verbose=True)

    # Get best agent
    best_id, best_record = population.get_best_agent()
    print(f"Best agent: {best_id}, fitness: {best_record.fitness:.2f}")

    print("✓ PBT training completed successfully")
    return True


def test_tournament_system(env):
    """Test tournament system."""
    print("\n=== Tournament System Test ===")

    # Create agents from different pools
    agents = {
        "TAG_pro": RuleBasedAgent(PersonalityType.TAG, player_id=0, seed=1),
        "LAG_pro": RuleBasedAgent(PersonalityType.LAG, player_id=1, seed=2),
        "Maniac_1": RuleBasedAgent(PersonalityType.MANIAC, player_id=2, seed=3),
        "Rock_1": RuleBasedAgent(PersonalityType.ROCK, player_id=3, seed=4),
        "Fish_1": RuleBasedAgent(PersonalityType.FISH, player_id=4, seed=5),
        "Balanced": RuleBasedAgent(PersonalityType.BALANCED, player_id=5, seed=6),
    }

    # Create 2-player environment for heads-up matches
    import poker_engine
    hu_env = poker_engine.PokerEnv(
        num_players=2,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    # Create tournament
    tournament = Tournament(
        name="6-max Personality Tournament",
        hands_per_match=100,
        num_rounds=1,
        seed=42,
    )

    for name, agent in agents.items():
        tournament.add_participant(name, agent)

    # Run tournament
    tournament.run(
        hu_env,
        get_action_fn=default_get_action,
        num_players=2,
        verbose=False,
    )

    # Print standings
    print("\nTournament Standings:")
    for i, standing in enumerate(tournament.get_standings()[:6]):
        print(
            f"  {i+1}. {standing['agent_id']:12} | "
            f"ELO: {standing['elo_rating']:.0f} | "
            f"W-L: {standing['wins']}-{standing['losses']} | "
            f"BB/100: {standing['bb_per_100']:+.1f}"
        )

    print("✓ Tournament completed successfully")
    return True


def test_distributed_collection(env):
    """Test distributed data collection."""
    print("\n=== Distributed Collection Test ===")

    trainer = DistributedTrainer(
        num_workers=2,
        num_players=6,
        hands_per_rollout=20,
        opponent_pool_name="mixed_pool",
        use_ray=False,  # Use local workers
        seed=42,
    )

    print(f"Workers: {trainer.num_workers}")

    # Collect rollouts
    start = time.time()
    rollouts = trainer.collect_rollouts(player_id=0)
    elapsed = time.time() - start

    total_hands = sum(r.num_hands for r in rollouts)
    total_steps = sum(len(r.states) for r in rollouts)

    print(f"Collected {total_hands} hands ({total_steps} steps) in {elapsed:.2f}s")
    print(f"Throughput: {total_hands/elapsed:.0f} hands/sec")

    # Merge rollouts
    merged = merge_rollouts(rollouts)
    print(f"Merged rollout: {merged.num_hands} hands, {len(merged.states)} steps")

    print("✓ Distributed collection completed successfully")
    return True


def run_sixmax_training_demo(env, iterations: int = 5):
    """Run a complete 6-max training demonstration."""
    print("\n" + "="*60)
    print("6-MAX TRAINING WITH PERSONALITY MIX - DEMO")
    print("="*60)

    # Create pools with personality mix
    opponent_pools = {
        "balanced_mix": create_pool("mixed_pool", pool_size=20, seed=1),
        "aggressive_mix": create_pool("lag_pool", pool_size=20, seed=2),
        "passive_mix": create_pool("fish_pool", pool_size=20, seed=3),
    }

    # Create feature extractor
    extractor = create_multi_player_extractor(max_players=6)

    # Create population of learning agents
    population = Population(
        population_size=4,
        num_players=6,
        selection_method=SelectionMethod.ELITISM,
        mutation_rate=0.2,
        seed=42,
    )

    print(f"\nPopulation initialized with {len(population.agents)} agents")
    print(f"Opponent pools: {list(opponent_pools.keys())}")
    print(f"Feature dimension: {extractor.feature_dim}")

    # Training loop
    print(f"\nStarting {iterations} training iterations...")

    for iteration in range(iterations):
        start = time.time()

        # Evaluate against each pool
        total_profit = 0.0
        total_hands = 0

        for pool_name, pool in opponent_pools.items():
            # Evaluate population
            fitness = population.evaluate_agents(
                env,
                {pool_name: pool},
                num_hands=50,
            )
            total_profit += sum(fitness.values())
            total_hands += 50 * len(fitness)

        # Evolve population
        population.evolve()

        elapsed = time.time() - start
        stats = population.get_statistics()

        print(
            f"Iteration {iteration+1}/{iterations} | "
            f"Fitness: {stats['fitness_mean']:.2f} (max: {stats['fitness_max']:.2f}) | "
            f"Time: {elapsed:.2f}s"
        )

    # Final results
    best_id, best_agent = population.get_best_agent()
    print(f"\nBest agent: {best_id}")
    print(f"  Fitness: {best_agent.fitness:.2f}")
    print(f"  Hyperparameters:")
    print(f"    lr_actor: {best_agent.hyperparameters.lr_actor:.2e}")
    print(f"    clip_ratio: {best_agent.hyperparameters.clip_ratio:.3f}")
    print(f"    entropy_coeff: {best_agent.hyperparameters.entropy_coeff:.4f}")

    print("\n✓ Training demo completed!")
    return True


def main():
    """Run all Phase 3 integration tests."""
    import poker_engine

    print("="*60)
    print("PHASE 3: 6-MAX MULTI-PLAYER TRAINING TESTS")
    print("="*60)

    # Create 6-max environment
    env = poker_engine.PokerEnv(
        num_players=6,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    tests = [
        ("Feature Extraction", lambda: test_feature_extraction(env)),
        ("Personality Pools", lambda: test_personality_pools(env)),
        ("Population Training", lambda: test_population_training(env)),
        ("Tournament System", lambda: test_tournament_system(env)),
        ("Distributed Collection", lambda: test_distributed_collection(env)),
        ("6-max Training Demo", lambda: run_sixmax_training_demo(env)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✓ All Phase 3 tests passed!")
    else:
        print(f"\n✗ {failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    main()
