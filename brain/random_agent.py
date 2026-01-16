"""Random agent for testing the poker environment."""

import argparse
import time
from typing import List, Optional

import numpy as np

import poker_engine
from env_wrapper import PokerGymEnv, PokerAECEnv


def random_action(action_mask: List[bool]) -> int:
    """Select a random valid action based on the action mask."""
    valid_actions = np.where(action_mask)[0]
    return int(np.random.choice(valid_actions))


def run_random_episode_gym(
    env: PokerGymEnv,
    verbose: bool = False,
) -> List[float]:
    """Run a single episode with random actions using Gym interface.

    Returns:
        List of final rewards for each player
    """
    obs, _ = env.reset()

    if verbose:
        print("\n" + "=" * 50)
        print("Starting new hand")
        print(f"Player 0 cards: {env.get_hole_cards(0)}")
        print(f"Player 1 cards: {env.get_hole_cards(1)}")

    total_rewards = [0.0] * env.num_players
    step = 0

    while True:
        action = random_action(obs["action_mask"])

        if verbose:
            action_names = ["Fold", "Check/Call", "Raise", "All-in"]
            print(f"Step {step}: Player {obs['current_player']} -> {action_names[action]}")

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        if done:
            # Get final rewards from info
            if verbose:
                print(f"\nHand complete after {step} actions")
                print(f"Board: {obs.get('board', [])}")
            break

    return total_rewards


def run_random_episode_aec(
    env: PokerAECEnv,
    verbose: bool = False,
) -> dict:
    """Run a single episode with random actions using AEC interface.

    Returns:
        Dict mapping agent names to final rewards
    """
    env.reset()

    if verbose:
        print("\n" + "=" * 50)
        print("Starting new hand (AEC)")

    step = 0
    max_steps = 100  # Safety limit

    while step < max_steps:
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            break

        action_mask = obs.get("action_mask", [True, True, True, True])
        action = random_action(action_mask)

        if verbose:
            agent = env.agent_selection
            action_names = ["Fold", "Check/Call", "Raise", "All-in"]
            print(f"Step {step}: {agent} -> {action_names[action]}")

        env.step(action)
        step += 1

    return {agent: env._rewards[agent] for agent in env.agents}


def benchmark_random_agents(
    num_players: int = 6,
    num_hands: int = 1000,
    use_native: bool = True,
) -> float:
    """Benchmark random agents playing poker.

    Args:
        num_players: Number of players
        num_hands: Number of hands to play
        use_native: Use native Rust benchmark (faster)

    Returns:
        Hands per second
    """
    if use_native:
        # Use the optimized Rust benchmark
        return poker_engine.benchmark_random_hands(num_hands, num_players, 12345)

    # Python-based benchmark
    env = PokerGymEnv(num_players=num_players, seed=12345)

    start_time = time.perf_counter()

    for _ in range(num_hands):
        obs, _ = env.reset()

        while True:
            action = random_action(obs["action_mask"])
            obs, reward, done, truncated, info = env.step(action)

            if done:
                break

    elapsed = time.perf_counter() - start_time
    return num_hands / elapsed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Random poker agent")
    parser.add_argument(
        "--players", type=int, default=6, help="Number of players (2-9)"
    )
    parser.add_argument(
        "--hands", type=int, default=10, help="Number of hands to play"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true", help="Run benchmark"
    )
    parser.add_argument(
        "--benchmark-hands",
        type=int,
        default=10000,
        help="Hands for benchmark",
    )
    parser.add_argument(
        "--aec", action="store_true", help="Use AEC interface instead of Gym"
    )

    args = parser.parse_args()

    if args.benchmark:
        print(f"Benchmarking with {args.benchmark_hands} hands, {args.players} players...")

        # Native benchmark
        native_hps = benchmark_random_agents(
            num_players=args.players,
            num_hands=args.benchmark_hands,
            use_native=True,
        )
        print(f"Native Rust: {native_hps:,.0f} hands/second")

        # Python benchmark (smaller number)
        python_hps = benchmark_random_agents(
            num_players=args.players,
            num_hands=min(1000, args.benchmark_hands),
            use_native=False,
        )
        print(f"Python wrapper: {python_hps:,.0f} hands/second")

        return

    # Play hands
    print(f"Playing {args.hands} hands with {args.players} players")
    print("-" * 50)

    if args.aec:
        env = PokerAECEnv(num_players=args.players, seed=42)

        for i in range(args.hands):
            rewards = run_random_episode_aec(env, verbose=args.verbose)
            if args.verbose or i == args.hands - 1:
                print(f"Hand {i + 1} rewards: {rewards}")
    else:
        env = PokerGymEnv(num_players=args.players, seed=42)

        for i in range(args.hands):
            rewards = run_random_episode_gym(env, verbose=args.verbose)
            if args.verbose or i == args.hands - 1:
                print(f"Hand {i + 1} complete")


if __name__ == "__main__":
    main()
