"""Hand history viewer and replay system.

This module provides:
1. Hand history loading and parsing
2. Hand replay with step-through
3. Hand analysis and statistics
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from .terminal_ui import CardDisplay, Color, TableDisplay
except ImportError:
    from terminal_ui import CardDisplay, Color, TableDisplay


@dataclass
class ReplayHand:
    """A hand loaded for replay."""
    number: int
    timestamp: str
    player_cards: List[tuple]
    board: List[tuple]
    actions: List[Dict]
    pot: int
    winner: int
    profit: float
    street: str


class HandHistoryViewer:
    """View and replay hand history."""

    STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River", 4: "Showdown"}

    def __init__(self, filepath: str):
        """Initialize viewer with history file.

        Args:
            filepath: Path to hand history JSON file
        """
        self.filepath = filepath
        self.hands: List[ReplayHand] = []
        self.session_start = ""
        self.total_profit = 0.0
        self.current_index = 0

        self._load_history()

    def _load_history(self):
        """Load hand history from file."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)

        self.session_start = data.get("session_start", "")
        self.total_profit = data.get("total_profit", 0.0)

        for h in data.get("hands", []):
            hand = ReplayHand(
                number=h["number"],
                timestamp=h["timestamp"],
                player_cards=[tuple(c) for c in h.get("player_cards", [])],
                board=[tuple(c) for c in h.get("board", [])],
                actions=h.get("actions", []),
                pot=h.get("pot", 0),
                winner=h.get("winner", -1),
                profit=h.get("profit", 0.0),
                street=h.get("street", "Unknown"),
            )
            self.hands.append(hand)

    def display_summary(self):
        """Display session summary."""
        print(f"\n{Color.BOLD}{'='*60}")
        print("  HAND HISTORY VIEWER")
        print(f"{'='*60}{Color.RESET}")
        print(f"  File: {self.filepath}")
        print(f"  Session: {self.session_start}")
        print(f"  Hands: {len(self.hands)}")

        if self.total_profit >= 0:
            print(f"  Total profit: {Color.BRIGHT_GREEN}+${self.total_profit:,.0f}{Color.RESET}")
        else:
            print(f"  Total profit: {Color.BRIGHT_RED}-${abs(self.total_profit):,.0f}{Color.RESET}")

        # Win/loss breakdown
        wins = sum(1 for h in self.hands if h.profit > 0)
        losses = sum(1 for h in self.hands if h.profit < 0)
        ties = len(self.hands) - wins - losses

        print(f"  W/L/T: {wins}/{losses}/{ties}")

        # Biggest hands
        if self.hands:
            biggest_win = max(self.hands, key=lambda h: h.profit)
            biggest_loss = min(self.hands, key=lambda h: h.profit)

            if biggest_win.profit > 0:
                print(f"  Biggest win: Hand #{biggest_win.number} (+${biggest_win.profit:,.0f})")
            if biggest_loss.profit < 0:
                print(f"  Biggest loss: Hand #{biggest_loss.number} (-${abs(biggest_loss.profit):,.0f})")

    def display_hand(self, index: int):
        """Display a single hand.

        Args:
            index: Hand index (0-based)
        """
        if index < 0 or index >= len(self.hands):
            print(f"{Color.RED}Invalid hand index.{Color.RESET}")
            return

        hand = self.hands[index]

        print(f"\n{Color.BOLD}{'#'*60}")
        print(f"  HAND #{hand.number}")
        print(f"{'#'*60}{Color.RESET}")

        print(f"\n  Timestamp: {hand.timestamp}")
        print(f"  Final pot: ${hand.pot:,}")
        print(f"  Reached: {hand.street}")

        # Player cards
        print(f"\n  Your cards: {CardDisplay.format_hand(hand.player_cards)}")

        # Board
        if hand.board:
            print(f"  Board: {CardDisplay.format_board(hand.board)}")

        # Actions
        print(f"\n{Color.BOLD}Actions:{Color.RESET}")
        current_street = -1
        for action in hand.actions:
            street = action.get("street", 0)
            if street != current_street:
                current_street = street
                print(f"\n  {Color.CYAN}--- {self.STREET_NAMES.get(street, 'Unknown')} ---{Color.RESET}")

            player = action.get("player", 0)
            action_type = action.get("action", 0)
            amount = action.get("amount", 0)

            action_name = self._get_action_name(action_type, amount)
            player_name = "You" if player == 0 else f"Player {player}"

            print(f"  {player_name}: {action_name}")

        # Result
        print(f"\n{Color.BOLD}Result:{Color.RESET}")
        if hand.profit > 0:
            print(f"  {Color.BRIGHT_GREEN}Won ${hand.profit:,.0f}{Color.RESET}")
        elif hand.profit < 0:
            print(f"  {Color.BRIGHT_RED}Lost ${abs(hand.profit):,.0f}{Color.RESET}")
        else:
            print(f"  Tie (no profit/loss)")

    def _get_action_name(self, action: int, amount: int) -> str:
        """Get action name from action index."""
        if action == 0:
            return "fold"
        elif action == 1:
            return f"call ${amount}" if amount > 0 else "check"
        elif action == 2:
            return f"raise to ${amount}"
        elif action == 3:
            return f"all-in ${amount}"
        return f"action {action}"

    def list_hands(self, start: int = 0, count: int = 10):
        """List hands in brief format.

        Args:
            start: Starting index
            count: Number of hands to show
        """
        print(f"\n{Color.BOLD}Hands {start+1}-{min(start+count, len(self.hands))} of {len(self.hands)}:{Color.RESET}")
        print(f"{'='*60}")

        for i in range(start, min(start + count, len(self.hands))):
            hand = self.hands[i]
            cards_str = CardDisplay.format_hand(hand.player_cards)

            if hand.profit > 0:
                result = f"{Color.BRIGHT_GREEN}+${hand.profit:,.0f}{Color.RESET}"
            elif hand.profit < 0:
                result = f"{Color.BRIGHT_RED}-${abs(hand.profit):,.0f}{Color.RESET}"
            else:
                result = "$0"

            print(f"  [{i}] Hand #{hand.number}: {cards_str} | {hand.street} | {result}")

    def interactive_viewer(self):
        """Run interactive history viewer."""
        self.display_summary()

        print(f"\n{Color.DIM}Commands: [n]ext, [p]rev, [g]oto #, [l]ist, [q]uit{Color.RESET}")

        while True:
            try:
                cmd = input(f"\n{Color.BRIGHT_WHITE}Enter command: {Color.RESET}").strip().lower()

                if not cmd:
                    # Show current hand
                    self.display_hand(self.current_index)

                elif cmd == 'q':
                    break

                elif cmd == 'n':
                    if self.current_index < len(self.hands) - 1:
                        self.current_index += 1
                        self.display_hand(self.current_index)
                    else:
                        print("Already at last hand.")

                elif cmd == 'p':
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.display_hand(self.current_index)
                    else:
                        print("Already at first hand.")

                elif cmd.startswith('g'):
                    try:
                        parts = cmd.split()
                        if len(parts) > 1:
                            idx = int(parts[1])
                        else:
                            idx = int(input("  Go to hand #: "))

                        if 0 <= idx < len(self.hands):
                            self.current_index = idx
                            self.display_hand(self.current_index)
                        else:
                            print(f"Invalid index. Range: 0-{len(self.hands)-1}")
                    except ValueError:
                        print("Invalid number.")

                elif cmd == 'l':
                    self.list_hands(0, 20)

                elif cmd == 's':
                    self.display_summary()

                else:
                    print(f"Unknown command: {cmd}")
                    print(f"{Color.DIM}Commands: [n]ext, [p]rev, [g]oto #, [l]ist, [s]ummary, [q]uit{Color.RESET}")

            except KeyboardInterrupt:
                print("\n")
                break
            except EOFError:
                break


def main():
    """Main entry point for hand history viewer."""
    import argparse

    parser = argparse.ArgumentParser(description="View poker hand history")
    parser.add_argument("file", help="Hand history JSON file")
    parser.add_argument("--list", "-l", action="store_true", help="List all hands")
    parser.add_argument("--hand", "-n", type=int, help="Show specific hand by index")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"{Color.RED}File not found: {args.file}{Color.RESET}")
        return

    viewer = HandHistoryViewer(args.file)

    if args.list:
        viewer.list_hands(0, len(viewer.hands))
    elif args.hand is not None:
        viewer.display_hand(args.hand)
    else:
        viewer.interactive_viewer()


if __name__ == "__main__":
    main()
