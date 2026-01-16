"""Human vs AI game controller.

This module provides:
1. Complete game loop for human vs AI play
2. AI opponent management
3. Session tracking
4. Hand history recording
"""

import sys
import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_engine

try:
    from .terminal_ui import (
        CardDisplay,
        TableDisplay,
        ActionInput,
        GameMessages,
        Color,
    )
except ImportError:
    from terminal_ui import (
        CardDisplay,
        TableDisplay,
        ActionInput,
        GameMessages,
        Color,
    )

try:
    from brain.personality_agents import (
        RuleBasedAgent,
        PersonalityType,
        PERSONALITY_CONFIGS,
    )
except ImportError:
    # Fallback if brain module not available
    RuleBasedAgent = None
    PersonalityType = None


@dataclass
class HandRecord:
    """Record of a single hand."""
    hand_number: int
    timestamp: str
    player_cards: List[Tuple[int, int]]
    board: List[Tuple[int, int]]
    actions: List[Dict]
    pot: int
    winner: int
    profit: float
    street_reached: str


@dataclass
class SessionStats:
    """Statistics for a playing session."""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    hands_played: int = 0
    hands_won: int = 0
    total_profit: float = 0.0
    biggest_pot_won: float = 0.0
    biggest_pot_lost: float = 0.0
    vpip_hands: int = 0  # Voluntarily put $ in pot
    pfr_hands: int = 0   # Preflop raise
    showdowns: int = 0
    showdown_wins: int = 0

    @property
    def win_rate(self) -> float:
        return self.hands_won / max(self.hands_played, 1)

    @property
    def vpip(self) -> float:
        return self.vpip_hands / max(self.hands_played, 1)

    @property
    def pfr(self) -> float:
        return self.pfr_hands / max(self.hands_played, 1)

    @property
    def wtsd(self) -> float:
        return self.showdowns / max(self.hands_played, 1)

    @property
    def won_at_showdown(self) -> float:
        return self.showdown_wins / max(self.showdowns, 1)


class GameController:
    """Main game controller for Human vs AI play."""

    ACTION_NAMES = {0: "fold", 1: "check/call", 2: "bet/raise", 3: "all_in"}
    STREET_NAMES = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River", 4: "Showdown"}

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        human_seat: int = 0,
        ai_personalities: Optional[Dict[int, str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize game controller.

        Args:
            num_players: Total number of players (2-9)
            starting_stack: Starting chip stack
            small_blind: Small blind amount
            big_blind: Big blind amount
            human_seat: Seat number for human player (0-based)
            ai_personalities: Dict of seat -> personality name
            seed: Random seed for AI agents
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.human_seat = human_seat

        # Create environment
        self.env = poker_engine.PokerEnv(
            num_players=num_players,
            small_blind=small_blind,
            big_blind=big_blind,
            starting_stack=starting_stack,
        )

        # Create AI opponents
        self.ai_agents: Dict[int, RuleBasedAgent] = {}
        self.ai_names: Dict[int, str] = {}
        self._setup_ai_opponents(ai_personalities, seed)

        # Session tracking
        self.stats = SessionStats()
        self.hand_history: List[HandRecord] = []
        self.hand_number = 0
        self.current_hand_actions: List[Dict] = []

    def _setup_ai_opponents(
        self,
        personalities: Optional[Dict[int, str]],
        seed: Optional[int],
    ):
        """Setup AI opponent agents."""
        if RuleBasedAgent is None:
            print(f"{Color.YELLOW}Warning: AI agents not available. Using random actions.{Color.RESET}")
            return

        # Default personalities if not specified
        default_personalities = {
            0: PersonalityType.BALANCED,
            1: PersonalityType.TAG,
            2: PersonalityType.LAG,
            3: PersonalityType.ROCK,
            4: PersonalityType.FISH,
            5: PersonalityType.MANIAC,
            6: PersonalityType.TAG,
            7: PersonalityType.LAG,
            8: PersonalityType.BALANCED,
        }

        for seat in range(self.num_players):
            if seat == self.human_seat:
                continue

            # Get personality
            if personalities and seat in personalities:
                personality_name = personalities[seat]
                try:
                    personality = PersonalityType(personality_name.lower())
                except ValueError:
                    personality = PersonalityType.BALANCED
            else:
                personality = default_personalities.get(seat, PersonalityType.BALANCED)

            # Create agent
            agent_seed = seed + seat * 100 if seed else None
            agent = RuleBasedAgent(
                personality=personality,
                player_id=seat,
                seed=agent_seed,
            )
            self.ai_agents[seat] = agent
            self.ai_names[seat] = f"{personality.value.upper()}-{seat}"

    def _get_ai_action(self, seat: int, obs: Dict) -> int:
        """Get action from AI agent.

        Args:
            seat: AI's seat number
            obs: Current observation

        Returns:
            Action index
        """
        if seat in self.ai_agents:
            return self.ai_agents[seat].get_action(obs, self.env)
        else:
            # Random fallback
            action_mask = obs.get("action_mask", [True] * 4)
            valid = [i for i, v in enumerate(action_mask) if v]
            import random
            return random.choice(valid)

    def _parse_cards(self, cards) -> List[Tuple[int, int]]:
        """Parse cards from various formats."""
        result = []
        for card in cards:
            if isinstance(card, (list, tuple)) and len(card) >= 2:
                r, s = card[0], card[1]
                if isinstance(r, str):
                    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                               '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                    r = rank_map.get(r, 0)
                if isinstance(s, str):
                    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
                    s = suit_map.get(s.lower(), 0)
                result.append((int(r), int(s)))
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                result.append((card.rank, card.suit))
        return result

    def play_hand(self) -> float:
        """Play a single hand.

        Returns:
            Profit/loss for human player
        """
        self.hand_number += 1
        self.current_hand_actions = []

        # Reset environment
        obs = self.env.reset()

        # Display new hand
        TableDisplay.clear_screen()
        print(GameMessages.new_hand(self.hand_number))

        # Track human's preflop action for stats
        human_vpip = False
        human_pfr = False
        last_street = 0

        # Main game loop
        while not obs["is_terminal"]:
            current_player = obs["current_player"]
            street = obs.get("street", 0)

            # Street change announcement
            if street != last_street:
                print(GameMessages.street_change(self.STREET_NAMES.get(street, "Unknown")))
                last_street = street

            # Display table
            print(TableDisplay.display_table(
                obs,
                player_id=self.human_seat,
                player_name="You",
                ai_names=self.ai_names,
            ))

            # Get action
            if current_player == self.human_seat:
                # Human's turn
                print(TableDisplay.display_actions(obs, min_raise=self.big_blind))

                stack = obs["stacks"][self.human_seat]
                action, amount = ActionInput.get_action(obs, stack, self.big_blind)

                # Track stats
                if street == 0:  # Preflop
                    if action in (1, 2, 3) and obs.get("to_call", 0) > 0:
                        human_vpip = True
                    if action in (2, 3):
                        human_pfr = True

                action_name = self._get_action_name(action, obs, amount)
                print(f"\n{GameMessages.action_taken('You', action_name, amount)}")

            else:
                # AI's turn
                action = self._get_ai_action(current_player, obs)
                amount = self._get_action_amount(action, obs, current_player)
                action_name = self._get_action_name(action, obs, amount)

                ai_name = self.ai_names.get(current_player, f"AI-{current_player}")
                print(f"{GameMessages.action_taken(ai_name, action_name, amount)}")

                time.sleep(0.3)  # Small delay for readability

            # Record action
            self.current_hand_actions.append({
                "player": current_player,
                "action": action,
                "amount": amount,
                "street": street,
            })

            # Execute action
            obs, rewards, done, _, info = self.env.step(action)

        # Hand complete
        TableDisplay.clear_screen()
        print(TableDisplay.display_table(
            obs,
            player_id=self.human_seat,
            player_name="You",
            ai_names=self.ai_names,
            show_all_cards=True,
        ))

        # Calculate result
        human_profit = rewards[self.human_seat]

        # Determine winner
        max_reward = max(rewards)
        winners = [i for i, r in enumerate(rewards) if r == max_reward and r > 0]

        for winner in winners:
            if winner == self.human_seat:
                print(GameMessages.winner("You", int(max_reward)))
            else:
                ai_name = self.ai_names.get(winner, f"AI-{winner}")
                print(GameMessages.winner(ai_name, int(rewards[winner])))

        # Update stats
        self.stats.hands_played += 1
        self.stats.total_profit += human_profit

        if human_profit > 0:
            self.stats.hands_won += 1
            self.stats.biggest_pot_won = max(self.stats.biggest_pot_won, human_profit)
        elif human_profit < 0:
            self.stats.biggest_pot_lost = min(self.stats.biggest_pot_lost, human_profit)

        if human_vpip:
            self.stats.vpip_hands += 1
        if human_pfr:
            self.stats.pfr_hands += 1

        # Check if went to showdown
        if last_street >= 3:
            self.stats.showdowns += 1
            if human_profit > 0:
                self.stats.showdown_wins += 1

        # Record hand
        hole_cards = self._parse_cards(obs.get("hole_cards", []))
        board = self._parse_cards(obs.get("board", []))

        record = HandRecord(
            hand_number=self.hand_number,
            timestamp=datetime.now().isoformat(),
            player_cards=hole_cards,
            board=board,
            actions=self.current_hand_actions,
            pot=int(obs.get("pot", 0)),
            winner=winners[0] if winners else -1,
            profit=human_profit,
            street_reached=self.STREET_NAMES.get(last_street, "Unknown"),
        )
        self.hand_history.append(record)

        return human_profit

    def _get_action_name(self, action: int, obs: Dict, amount: int) -> str:
        """Get human-readable action name."""
        to_call = obs.get("to_call", 0)

        if action == 0:
            return "fold"
        elif action == 1:
            return "check" if to_call == 0 else "call"
        elif action == 2:
            return "bet" if to_call == 0 else "raise"
        elif action == 3:
            return "all_in"
        return "unknown"

    def _get_action_amount(self, action: int, obs: Dict, player: int) -> int:
        """Get amount for an action."""
        to_call = obs.get("to_call", 0)
        stack = obs["stacks"][player]

        if action == 0:
            return 0
        elif action == 1:
            return min(to_call, stack)
        elif action == 2:
            # AI uses pot-sized bet
            pot = obs.get("pot", 0)
            return min(max(pot, self.big_blind * 2), stack)
        elif action == 3:
            return stack
        return 0

    def run_session(self, max_hands: Optional[int] = None):
        """Run an interactive playing session.

        Args:
            max_hands: Maximum hands to play (None for unlimited)
        """
        print(f"\n{Color.BOLD}{'='*60}")
        print("  POKER AI - Human vs AI")
        print(f"{'='*60}{Color.RESET}")
        print(f"\n  Players: {self.num_players}")
        print(f"  Blinds: ${self.small_blind}/${self.big_blind}")
        print(f"  Starting stack: ${self.starting_stack:,}")
        print(f"  Your seat: {self.human_seat}")
        print(f"\n  AI Opponents:")
        for seat, name in self.ai_names.items():
            print(f"    Seat {seat}: {name}")

        print(f"\n{Color.DIM}Press Enter to start, Ctrl+C to quit anytime...{Color.RESET}")
        input()

        try:
            while max_hands is None or self.hand_number < max_hands:
                profit = self.play_hand()

                # Show running total
                print(f"\n{Color.DIM}Session profit: ", end="")
                if self.stats.total_profit >= 0:
                    print(f"{Color.BRIGHT_GREEN}+${self.stats.total_profit:,.0f}{Color.RESET}")
                else:
                    print(f"{Color.BRIGHT_RED}-${abs(self.stats.total_profit):,.0f}{Color.RESET}")

                # Continue prompt
                print(f"\n{Color.DIM}Press Enter for next hand, 'q' to quit, 's' for stats...{Color.RESET}")
                user_input = input().strip().lower()

                if user_input == 'q':
                    break
                elif user_input == 's':
                    self._show_stats()
                    input(f"\n{Color.DIM}Press Enter to continue...{Color.RESET}")

        except KeyboardInterrupt:
            print(f"\n\n{Color.YELLOW}Session interrupted.{Color.RESET}")
        except EOFError:
            pass

        # Show final summary
        self._show_summary()

    def _show_stats(self):
        """Display current session statistics."""
        s = self.stats
        print(f"\n{Color.BOLD}Session Statistics:{Color.RESET}")
        print(f"  Hands: {s.hands_played}")
        print(f"  Win rate: {s.win_rate*100:.1f}%")
        print(f"  VPIP: {s.vpip*100:.1f}%")
        print(f"  PFR: {s.pfr*100:.1f}%")
        print(f"  WTSD: {s.wtsd*100:.1f}%")
        print(f"  W$SD: {s.won_at_showdown*100:.1f}%")
        print(f"  Biggest win: ${s.biggest_pot_won:,.0f}")
        print(f"  Biggest loss: ${abs(s.biggest_pot_lost):,.0f}")

    def _show_summary(self):
        """Display session summary."""
        print(GameMessages.session_summary(
            self.stats.hands_played,
            self.stats.total_profit,
            self.big_blind,
        ))

        self._show_stats()

    def save_history(self, filepath: str):
        """Save hand history to file.

        Args:
            filepath: Path to save file
        """
        data = {
            "session_start": self.stats.start_time,
            "hands_played": self.stats.hands_played,
            "total_profit": self.stats.total_profit,
            "hands": [
                {
                    "number": h.hand_number,
                    "timestamp": h.timestamp,
                    "player_cards": h.player_cards,
                    "board": h.board,
                    "actions": h.actions,
                    "pot": h.pot,
                    "winner": h.winner,
                    "profit": h.profit,
                    "street": h.street_reached,
                }
                for h in self.hand_history
            ]
        }

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n{Color.GREEN}Hand history saved to {filepath}{Color.RESET}")


def quick_play():
    """Quick start a game with default settings."""
    game = GameController(
        num_players=6,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        human_seat=0,
        seed=int(time.time()),
    )
    game.run_session()


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Play poker against AI")
    parser.add_argument("--players", type=int, default=6, choices=range(2, 10),
                       help="Number of players (2-9)")
    parser.add_argument("--stack", type=int, default=10000, help="Starting stack")
    parser.add_argument("--sb", type=int, default=50, help="Small blind")
    parser.add_argument("--bb", type=int, default=100, help="Big blind")
    parser.add_argument("--seat", type=int, default=0, help="Your seat (0-based)")
    parser.add_argument("--hands", type=int, default=None, help="Max hands to play")
    parser.add_argument("--save", type=str, default=None, help="Save hand history to file")

    args = parser.parse_args()

    game = GameController(
        num_players=args.players,
        starting_stack=args.stack,
        small_blind=args.sb,
        big_blind=args.bb,
        human_seat=args.seat,
        seed=int(time.time()),
    )

    game.run_session(max_hands=args.hands)

    if args.save:
        game.save_history(args.save)


if __name__ == "__main__":
    main()
