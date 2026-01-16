#!/usr/bin/env python3
"""Unified training script for poker AI.

This script provides:
- Deep CFR training
- PPO self-play training
- Model evaluation
- Agent vs agent comparison
"""

import argparse
import os
import time
from typing import Optional, Dict, List, Tuple
import numpy as np

import poker_engine
try:
    from .deep_cfr import DeepCFR, DeepCFRAgent, train_deep_cfr
    from .ppo_agent import PPOAgent, train_ppo
except ImportError:
    from deep_cfr import DeepCFR, DeepCFRAgent, train_deep_cfr
    from ppo_agent import PPOAgent, train_ppo


def evaluate_agent(
    agent1: any,
    agent2: any,
    num_hands: int = 1000,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate two agents against each other.

    Args:
        agent1: First agent
        agent2: Second agent
        num_hands: Number of hands to play
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    env = poker_engine.PokerEnv(
        num_players=2,
        small_blind=50,
        big_blind=100,
        starting_stack=10000,
    )

    total_rewards = [0.0, 0.0]
    wins = [0, 0]

    for hand in range(num_hands):
        obs = env.reset()

        while not obs["is_terminal"]:
            current_player = obs["current_player"]

            # Get action from appropriate agent
            if current_player == 0:
                action = get_agent_action(agent1, env, obs, 0)
            else:
                action = get_agent_action(agent2, env, obs, 1)

            obs, rewards, done, _, info = env.step(action)

        # Track rewards
        if done:
            for i in range(2):
                total_rewards[i] += rewards[i]
                if rewards[i] > 0:
                    wins[i] += 1

        if verbose and (hand + 1) % 100 == 0:
            print(
                f"Hand {hand + 1}/{num_hands} | "
                f"Agent1: {total_rewards[0]:.0f} ({wins[0]} wins) | "
                f"Agent2: {total_rewards[1]:.0f} ({wins[1]} wins)"
            )

    return {
        "agent1_total": total_rewards[0],
        "agent2_total": total_rewards[1],
        "agent1_bb_per_100": total_rewards[0] / (num_hands / 100) / 100,
        "agent2_bb_per_100": total_rewards[1] / (num_hands / 100) / 100,
        "agent1_win_rate": wins[0] / num_hands,
        "agent2_win_rate": wins[1] / num_hands,
    }


def get_agent_action(agent: any, env: any, obs: Dict, player_id: int) -> int:
    """Get action from any agent type."""
    # Handle different agent types
    if isinstance(agent, DeepCFRAgent):
        return agent.get_action(env, obs, deterministic=False)
    elif isinstance(agent, PPOAgent):
        features = agent._extract_features(env, obs, player_id)
        action_mask = np.array(obs["action_mask"], dtype=np.float32)
        action, _, _ = agent.get_action(features, action_mask, deterministic=False)
        return action
    elif agent == "random":
        # Random agent
        valid_actions = np.where(obs["action_mask"])[0]
        return int(np.random.choice(valid_actions))
    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")


class RandomAgent:
    """Simple random agent for baseline comparison."""

    def __init__(self):
        pass


def train_and_evaluate(
    algorithm: str,
    num_iterations: int,
    eval_every: int = 100,
    save_dir: str = "models",
) -> any:
    """Train an agent and periodically evaluate against random.

    Args:
        algorithm: "deep_cfr" or "ppo"
        num_iterations: Number of training iterations
        eval_every: Evaluate every N iterations
        save_dir: Directory to save models

    Returns:
        Trained agent
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} for {num_iterations} iterations")
    print(f"{'='*60}\n")

    if algorithm == "deep_cfr":
        model = DeepCFR(
            num_players=2,
            num_actions=4,
            hidden_dims=[256, 256, 128],
            buffer_size=50000,
            batch_size=512,
        )

        for iteration in range(num_iterations):
            # Train one iteration
            model.iteration = iteration + 1

            for t in range(100):  # 100 traversals per iteration
                traverse_player = t % 2
                model._traverse_game(traverse_player)

                if (t + 1) % 20 == 0:
                    model._train_networks()

            model._train_networks()

            # Evaluate periodically
            if (iteration + 1) % eval_every == 0:
                agent = DeepCFRAgent(model, player_id=0)
                stats = evaluate_agent(agent, "random", num_hands=200)
                print(
                    f"Iteration {iteration + 1} | "
                    f"vs Random: {stats['agent1_bb_per_100']:.2f} BB/100"
                )

        # Save final model
        model.save(os.path.join(save_dir, "deep_cfr_final.pt"))
        return DeepCFRAgent(model, player_id=0)

    elif algorithm == "ppo":
        agent = PPOAgent(
            num_players=2,
            num_actions=4,
            hidden_dims=[256, 256],
            actor_dims=[128],
            critic_dims=[128],
        )

        env = poker_engine.PokerEnv(
            num_players=2,
            small_blind=50,
            big_blind=100,
            starting_stack=10000,
        )

        for iteration in range(num_iterations):
            # Collect and train
            rollout_stats = agent.collect_rollout(
                env=env,
                opponent=agent,
                num_steps=512,
                player_id=iteration % 2,
            )
            train_stats = agent.train_step(batch_size=64, epochs=4)

            # Evaluate periodically
            if (iteration + 1) % eval_every == 0:
                stats = evaluate_agent(agent, "random", num_hands=200)
                print(
                    f"Iteration {iteration + 1} | "
                    f"vs Random: {stats['agent1_bb_per_100']:.2f} BB/100 | "
                    f"Mean Reward: {rollout_stats['mean_reward']:.1f}"
                )

        # Save final model
        agent.save(os.path.join(save_dir, "ppo_final.pt"))
        return agent

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train poker AI")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["deep_cfr", "ppo"],
        help="Training algorithm",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires saved models)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Deep CFR vs PPO",
    )

    args = parser.parse_args()

    if args.eval_only:
        print("Evaluation mode - loading saved models...")
        # Load and evaluate saved models
        # TODO: Implement model loading
        return

    if args.compare:
        print("\n" + "="*60)
        print("Training both algorithms for comparison")
        print("="*60)

        # Train both
        print("\n--- Training PPO ---")
        ppo_agent = train_and_evaluate(
            "ppo",
            num_iterations=args.iterations,
            eval_every=args.eval_every,
            save_dir=args.save_dir,
        )

        print("\n--- Training Deep CFR ---")
        cfr_agent = train_and_evaluate(
            "deep_cfr",
            num_iterations=args.iterations,
            eval_every=args.eval_every,
            save_dir=args.save_dir,
        )

        # Compare them
        print("\n" + "="*60)
        print("Final Comparison: PPO vs Deep CFR")
        print("="*60)

        stats = evaluate_agent(ppo_agent, cfr_agent, num_hands=1000, verbose=True)
        print(f"\nPPO: {stats['agent1_bb_per_100']:.2f} BB/100")
        print(f"Deep CFR: {stats['agent2_bb_per_100']:.2f} BB/100")

        return

    # Train single algorithm
    agent = train_and_evaluate(
        args.algorithm,
        num_iterations=args.iterations,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
    )

    # Final evaluation vs random
    print("\n" + "="*60)
    print("Final Evaluation vs Random Agent (1000 hands)")
    print("="*60)

    stats = evaluate_agent(agent, "random", num_hands=1000, verbose=True)
    print(f"\nFinal BB/100: {stats['agent1_bb_per_100']:.2f}")
    print(f"Win rate: {stats['agent1_win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
