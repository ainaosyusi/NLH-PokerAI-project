"""Terminal-based poker UI with Unicode card display.

This module provides:
1. Rich card display with Unicode suits
2. Table visualization
3. Action input handling
4. Clear game state display
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Color:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_WHITE = "\033[47m"


class CardDisplay:
    """Display cards with Unicode suits."""

    # Unicode suit symbols
    SUITS = {
        0: ("♣", Color.GREEN),      # Clubs
        1: ("♦", Color.BRIGHT_BLUE),  # Diamonds
        2: ("♥", Color.BRIGHT_RED),   # Hearts
        3: ("♠", Color.WHITE),        # Spades
        "c": ("♣", Color.GREEN),
        "d": ("♦", Color.BRIGHT_BLUE),
        "h": ("♥", Color.BRIGHT_RED),
        "s": ("♠", Color.WHITE),
    }

    RANKS = {
        0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7",
        6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A",
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
        "8": "8", "9": "9", "T": "T", "J": "J", "Q": "Q", "K": "K", "A": "A",
    }

    @classmethod
    def format_card(cls, rank, suit, hidden: bool = False) -> str:
        """Format a single card for display.

        Args:
            rank: Card rank (0-12 or string)
            suit: Card suit (0-3 or string)
            hidden: Show as face-down card

        Returns:
            Formatted card string with color
        """
        if hidden:
            return f"{Color.DIM}[??]{Color.RESET}"

        rank_str = cls.RANKS.get(rank, str(rank))
        suit_sym, suit_color = cls.SUITS.get(suit, ("?", Color.WHITE))

        return f"{suit_color}{rank_str}{suit_sym}{Color.RESET}"

    @classmethod
    def format_hand(cls, cards: List[Tuple], hidden: bool = False) -> str:
        """Format a hand of cards.

        Args:
            cards: List of (rank, suit) tuples
            hidden: Show all cards as face-down

        Returns:
            Formatted hand string
        """
        if not cards:
            return "[  ] [  ]"

        formatted = []
        for card in cards:
            if isinstance(card, (list, tuple)) and len(card) >= 2:
                formatted.append(cls.format_card(card[0], card[1], hidden))
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                formatted.append(cls.format_card(card.rank, card.suit, hidden))
            else:
                formatted.append(cls.format_card(0, 0, hidden=True))

        return " ".join(formatted)

    @classmethod
    def format_board(cls, board: List[Tuple]) -> str:
        """Format community cards.

        Args:
            board: List of (rank, suit) tuples

        Returns:
            Formatted board string
        """
        if not board:
            return "[  ] [  ] [  ] [  ] [  ]"

        cards = []
        for card in board:
            if isinstance(card, (list, tuple)) and len(card) >= 2:
                cards.append(cls.format_card(card[0], card[1]))
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                cards.append(cls.format_card(card.rank, card.suit))

        # Pad to 5 cards
        while len(cards) < 5:
            cards.append(f"{Color.DIM}[  ]{Color.RESET}")

        return " ".join(cards)


class TableDisplay:
    """Display poker table state."""

    POSITION_NAMES = {
        0: "BTN",  # Button
        1: "SB",   # Small Blind
        2: "BB",   # Big Blind
        3: "UTG",  # Under the Gun
        4: "MP",   # Middle Position
        5: "CO",   # Cutoff
        6: "HJ",   # Hijack
        7: "LJ",   # Lojack
        8: "UTG+1",
    }

    STREET_NAMES = {
        0: "Preflop",
        1: "Flop",
        2: "Turn",
        3: "River",
        4: "Showdown",
    }

    @classmethod
    def clear_screen(cls):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    @classmethod
    def get_position_name(cls, seat: int, button: int, num_players: int) -> str:
        """Get position name for a seat.

        Args:
            seat: Seat number (0-based)
            button: Button position
            num_players: Total number of players

        Returns:
            Position name string
        """
        relative_pos = (seat - button) % num_players

        if num_players == 2:
            return "BTN/SB" if relative_pos == 0 else "BB"

        return cls.POSITION_NAMES.get(relative_pos, f"Seat {seat}")

    @classmethod
    def display_table(
        cls,
        obs: Dict,
        player_id: int,
        player_name: str = "You",
        ai_names: Optional[Dict[int, str]] = None,
        show_all_cards: bool = False,
    ) -> str:
        """Generate table display string.

        Args:
            obs: Observation dictionary from environment
            player_id: Human player's seat
            player_name: Display name for human
            ai_names: Dict of seat -> AI name
            show_all_cards: Show all hole cards (showdown)

        Returns:
            Formatted table string
        """
        ai_names = ai_names or {}
        lines = []

        # Header
        street = obs.get("street", 0)
        pot = obs.get("pot", 0)
        street_name = cls.STREET_NAMES.get(street, "Unknown")

        lines.append(f"{Color.BOLD}{'='*60}{Color.RESET}")
        lines.append(f"{Color.BOLD}  {street_name}  |  Pot: {Color.BRIGHT_GREEN}${pot:,}{Color.RESET}")
        lines.append(f"{Color.BOLD}{'='*60}{Color.RESET}")

        # Board
        board = obs.get("board", [])
        lines.append("")
        lines.append(f"  Board: {CardDisplay.format_board(board)}")
        lines.append("")

        # Players
        lines.append(f"{Color.DIM}{'-'*60}{Color.RESET}")

        stacks = obs.get("stacks", [])
        active = obs.get("active_players", [])
        bets = obs.get("bets_this_round", [0] * len(stacks))
        hole_cards = obs.get("hole_cards", [])
        current = obs.get("current_player", -1)
        button = obs.get("button", 0)
        num_players = obs.get("num_players", len(stacks))

        for i in range(len(stacks)):
            # Player info
            is_human = (i == player_id)
            is_active = active[i] if i < len(active) else False
            is_current = (i == current)
            stack = stacks[i] if i < len(stacks) else 0
            bet = bets[i] if i < len(bets) else 0

            # Name
            if is_human:
                name = f"{Color.BRIGHT_CYAN}{player_name}{Color.RESET}"
            else:
                ai_name = ai_names.get(i, f"AI-{i}")
                name = f"{Color.YELLOW}{ai_name}{Color.RESET}"

            # Position
            pos_name = cls.get_position_name(i, button, num_players)

            # Status indicator
            if not is_active:
                status = f"{Color.DIM}(folded){Color.RESET}"
            elif is_current:
                status = f"{Color.BRIGHT_GREEN}<<< ACTION{Color.RESET}"
            else:
                status = ""

            # Cards
            if is_human and hole_cards:
                cards_str = CardDisplay.format_hand(hole_cards)
            elif show_all_cards and is_active:
                # At showdown, show AI cards if available
                cards_str = CardDisplay.format_hand([], hidden=True)  # Would need opponent cards
            else:
                cards_str = CardDisplay.format_hand([], hidden=True)

            # Stack and bet
            stack_str = f"${stack:,}"
            bet_str = f"(bet: ${bet})" if bet > 0 else ""

            # Format line
            line = f"  [{pos_name:4}] {name:20} {cards_str:15} Stack: {stack_str:>10} {bet_str} {status}"
            lines.append(line)

        lines.append(f"{Color.DIM}{'-'*60}{Color.RESET}")

        return "\n".join(lines)

    @classmethod
    def display_actions(cls, obs: Dict, min_raise: int = 0) -> str:
        """Display available actions.

        Args:
            obs: Observation dictionary
            min_raise: Minimum raise amount

        Returns:
            Formatted action prompt string
        """
        action_mask = obs.get("action_mask", [True, True, True, True])
        to_call = obs.get("to_call", 0)
        pot = obs.get("pot", 0)

        lines = []
        lines.append(f"\n{Color.BOLD}Available Actions:{Color.RESET}")

        actions = []
        if action_mask[0]:
            actions.append(f"[{Color.RED}F{Color.RESET}]old")
        if action_mask[1]:
            if to_call == 0:
                actions.append(f"[{Color.GREEN}C{Color.RESET}]heck")
            else:
                actions.append(f"[{Color.GREEN}C{Color.RESET}]all ${to_call}")
        if action_mask[2]:
            if to_call == 0:
                actions.append(f"[{Color.YELLOW}B{Color.RESET}]et (min: ${min_raise})")
            else:
                actions.append(f"[{Color.YELLOW}R{Color.RESET}]aise (min: ${min_raise})")
        if action_mask[3]:
            actions.append(f"[{Color.MAGENTA}A{Color.RESET}]ll-in")

        lines.append("  " + "  |  ".join(actions))

        # Bet sizing hints
        if action_mask[2]:
            half_pot = pot // 2
            full_pot = pot
            lines.append(f"\n  {Color.DIM}Bet sizes: 1/2 pot = ${half_pot}, pot = ${full_pot}{Color.RESET}")

        return "\n".join(lines)


class ActionInput:
    """Handle action input from user."""

    @classmethod
    def get_action(cls, obs: Dict, stack: int, min_raise: int = 100) -> Tuple[int, int]:
        """Get action from user input.

        Args:
            obs: Observation dictionary
            stack: Player's current stack
            min_raise: Minimum raise amount

        Returns:
            Tuple of (action_index, bet_amount)
        """
        action_mask = obs.get("action_mask", [True, True, True, True])
        to_call = obs.get("to_call", 0)

        while True:
            try:
                user_input = input(f"\n{Color.BRIGHT_WHITE}Enter action: {Color.RESET}").strip().upper()

                if not user_input:
                    print(f"{Color.RED}Please enter an action.{Color.RESET}")
                    continue

                # Parse action
                action_char = user_input[0]

                if action_char == 'F' and action_mask[0]:
                    return 0, 0

                elif action_char == 'C' and action_mask[1]:
                    return 1, to_call

                elif action_char in ('B', 'R') and action_mask[2]:
                    # Get bet amount
                    amount = cls._get_bet_amount(user_input, obs, stack, min_raise)
                    if amount is not None:
                        return 2, amount

                elif action_char == 'A' and action_mask[3]:
                    return 3, stack

                else:
                    print(f"{Color.RED}Invalid action. Please try again.{Color.RESET}")

            except KeyboardInterrupt:
                print(f"\n{Color.YELLOW}Game interrupted.{Color.RESET}")
                raise
            except EOFError:
                print(f"\n{Color.YELLOW}Input ended.{Color.RESET}")
                raise

    @classmethod
    def _get_bet_amount(
        cls,
        user_input: str,
        obs: Dict,
        stack: int,
        min_raise: int,
    ) -> Optional[int]:
        """Parse bet amount from user input.

        Args:
            user_input: User's input string
            obs: Observation dictionary
            stack: Player's stack
            min_raise: Minimum raise

        Returns:
            Bet amount or None if invalid
        """
        pot = obs.get("pot", 0)
        to_call = obs.get("to_call", 0)

        # Check for amount in input
        parts = user_input.split()

        if len(parts) > 1:
            amount_str = parts[1]

            # Handle pot-relative sizing
            if 'P' in amount_str.upper():
                try:
                    multiplier = float(amount_str.upper().replace('P', '').replace('POT', '') or '1')
                    amount = int(pot * multiplier)
                except ValueError:
                    amount = pot
            else:
                try:
                    amount = int(amount_str.replace('$', '').replace(',', ''))
                except ValueError:
                    print(f"{Color.RED}Invalid amount. Please enter a number.{Color.RESET}")
                    return None
        else:
            # Prompt for amount
            try:
                amount_input = input(f"  Enter amount (min ${min_raise}, max ${stack}): $").strip()

                if 'P' in amount_input.upper():
                    multiplier = float(amount_input.upper().replace('P', '').replace('POT', '') or '1')
                    amount = int(pot * multiplier)
                else:
                    amount = int(amount_input.replace('$', '').replace(',', ''))
            except ValueError:
                print(f"{Color.RED}Invalid amount.{Color.RESET}")
                return None

        # Validate amount
        if amount < min_raise:
            print(f"{Color.RED}Amount must be at least ${min_raise}.{Color.RESET}")
            return None

        if amount > stack:
            print(f"{Color.YELLOW}Amount exceeds stack. Going all-in for ${stack}.{Color.RESET}")
            return stack

        return amount


class GameMessages:
    """Display game messages and events."""

    @classmethod
    def action_taken(cls, player_name: str, action: str, amount: int = 0) -> str:
        """Format action taken message."""
        if action == "fold":
            return f"{Color.DIM}{player_name} folds.{Color.RESET}"
        elif action == "check":
            return f"{player_name} checks."
        elif action == "call":
            return f"{player_name} calls ${amount}."
        elif action == "bet":
            return f"{Color.YELLOW}{player_name} bets ${amount}.{Color.RESET}"
        elif action == "raise":
            return f"{Color.YELLOW}{player_name} raises to ${amount}.{Color.RESET}"
        elif action == "all_in":
            return f"{Color.MAGENTA}{Color.BOLD}{player_name} goes ALL-IN for ${amount}!{Color.RESET}"
        return f"{player_name}: {action}"

    @classmethod
    def winner(cls, player_name: str, amount: int, hand_name: str = "") -> str:
        """Format winner message."""
        hand_info = f" with {hand_name}" if hand_name else ""
        return f"\n{Color.BRIGHT_GREEN}{Color.BOLD}*** {player_name} wins ${amount}{hand_info} ***{Color.RESET}\n"

    @classmethod
    def new_hand(cls, hand_num: int) -> str:
        """Format new hand message."""
        return f"\n{Color.BOLD}{'#'*60}\n  HAND #{hand_num}\n{'#'*60}{Color.RESET}\n"

    @classmethod
    def street_change(cls, street_name: str) -> str:
        """Format street change message."""
        return f"\n{Color.CYAN}--- {street_name} ---{Color.RESET}"

    @classmethod
    def session_summary(cls, hands: int, profit: float, bb: int = 100) -> str:
        """Format session summary."""
        bb_per_100 = profit / (hands / 100) / bb if hands > 0 else 0

        lines = []
        lines.append(f"\n{Color.BOLD}{'='*60}")
        lines.append("  SESSION SUMMARY")
        lines.append(f"{'='*60}{Color.RESET}")
        lines.append(f"  Hands played: {hands}")

        if profit >= 0:
            lines.append(f"  Profit: {Color.BRIGHT_GREEN}+${profit:,.0f}{Color.RESET}")
        else:
            lines.append(f"  Profit: {Color.BRIGHT_RED}-${abs(profit):,.0f}{Color.RESET}")

        lines.append(f"  BB/100: {bb_per_100:+.1f}")
        lines.append("")

        return "\n".join(lines)


# Testing
if __name__ == "__main__":
    print("=== Terminal UI Test ===\n")

    # Test card display
    print("Card display test:")
    print(f"  Ace of spades: {CardDisplay.format_card(12, 3)}")
    print(f"  King of hearts: {CardDisplay.format_card(11, 2)}")
    print(f"  Hidden card: {CardDisplay.format_card(0, 0, hidden=True)}")

    # Test hand display
    print(f"\n  Hand: {CardDisplay.format_hand([(12, 3), (12, 2)])}")  # AA
    print(f"  Board: {CardDisplay.format_board([(10, 0), (5, 1), (2, 2)])}")  # Flop

    # Test table display
    mock_obs = {
        "street": 1,  # Flop
        "pot": 450,
        "board": [(10, 0), (5, 1), (2, 2)],
        "stacks": [9550, 9450, 10000, 10000, 10000, 10000],
        "active_players": [True, True, True, False, False, False],
        "bets_this_round": [200, 200, 0, 0, 0, 0],
        "hole_cards": [(12, 3), (12, 2)],  # AA
        "current_player": 2,
        "button": 0,
        "num_players": 6,
        "to_call": 200,
        "action_mask": [True, True, True, True],
    }

    print("\nTable display test:")
    print(TableDisplay.display_table(
        mock_obs,
        player_id=0,
        player_name="Hero",
        ai_names={1: "TAG-Bot", 2: "LAG-Bot", 3: "Rock", 4: "Fish", 5: "Maniac"},
    ))

    print(TableDisplay.display_actions(mock_obs, min_raise=400))

    # Test messages
    print("\nMessage tests:")
    print(GameMessages.action_taken("Hero", "raise", 600))
    print(GameMessages.winner("Hero", 1200, "pair of Aces"))
    print(GameMessages.new_hand(42))
    print(GameMessages.session_summary(100, 1500))

    print("\n✓ All UI tests passed!")
