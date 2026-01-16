#!/usr/bin/env python3
"""
Poker AI - Play against AI opponents

Usage:
    python play_poker.py                    # Quick start 6-max game
    python play_poker.py --players 2        # Heads-up game
    python play_poker.py --stack 50000      # Deep stack
    python play_poker.py --help             # Show all options

Controls during game:
    F - Fold
    C - Check/Call
    B/R - Bet/Raise (will prompt for amount)
    A - All-in

    Enter amount like: R 500, B 1000, or use pot multipliers: B 1P (pot), B 0.5P (half pot)
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.game_controller import GameController, quick_play
from ui.terminal_ui import Color


def print_banner():
    """Print welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ♠ ♥ ♦ ♣   POKER AI   ♣ ♦ ♥ ♠                          ║
    ║                                                           ║
    ║              No-Limit Texas Hold'em                       ║
    ║              Human vs AI Challenge                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(f"{Color.BRIGHT_CYAN}{banner}{Color.RESET}")


def main():
    """Main entry point."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Play No-Limit Texas Hold'em against AI opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_poker.py                    Quick start 6-max
  python play_poker.py --players 2        Heads-up game
  python play_poker.py --stack 50000      Deep stack game
  python play_poker.py --bb 200           Higher stakes
  python play_poker.py --save history.json  Save hand history

AI Personalities:
  balanced - Solid, balanced play
  tag      - Tight-aggressive
  lag      - Loose-aggressive
  rock     - Very tight, passive
  fish     - Loose, passive
  maniac   - Very aggressive
        """,
    )

    parser.add_argument(
        "--players", "-p",
        type=int,
        default=6,
        choices=range(2, 10),
        metavar="N",
        help="Number of players (2-9, default: 6)"
    )
    parser.add_argument(
        "--stack", "-s",
        type=int,
        default=10000,
        help="Starting stack (default: 10000)"
    )
    parser.add_argument(
        "--sb",
        type=int,
        default=50,
        help="Small blind (default: 50)"
    )
    parser.add_argument(
        "--bb",
        type=int,
        default=100,
        help="Big blind (default: 100)"
    )
    parser.add_argument(
        "--seat",
        type=int,
        default=0,
        help="Your seat position (0-based, default: 0)"
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=None,
        help="Maximum hands to play (default: unlimited)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save hand history to file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick start with defaults (skip menu)"
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Validate seat
    if args.seat >= args.players:
        print(f"{Color.RED}Error: Seat {args.seat} invalid for {args.players} players.{Color.RESET}")
        return 1

    # Use current time as seed if not specified
    seed = args.seed if args.seed is not None else int(time.time())

    # Create and run game
    game = GameController(
        num_players=args.players,
        starting_stack=args.stack,
        small_blind=args.sb,
        big_blind=args.bb,
        human_seat=args.seat,
        seed=seed,
    )

    try:
        game.run_session(max_hands=args.hands)
    except Exception as e:
        print(f"\n{Color.RED}Error: {e}{Color.RESET}")
        return 1

    # Save history if requested
    if args.save:
        game.save_history(args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
