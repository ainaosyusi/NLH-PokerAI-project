"""
Poker AI - Phase 4: Personality Analysis & Opponent Modeling

This module provides comprehensive player statistics tracking, automatic
personality classification, and opponent modeling for poker AI.

Key Features:
- Real-time HUD-style statistics (VPIP, PFR, AF, 3-bet%, C-bet%, etc.)
- Automatic personality classification (LAG, TAG, ROCK, FISH, etc.)
- Opponent modeling with adaptive tracking
- Agent comparison and analysis tools
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import json
from datetime import datetime


class Street(Enum):
    """Poker streets."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class ActionType(Enum):
    """Categorized action types."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALLIN = "allin"


class PersonalityType(Enum):
    """Player personality archetypes based on observed statistics."""
    MANIAC = "maniac"      # Very loose, very aggressive (VPIP>50%, AF>3)
    LAG = "lag"            # Loose-aggressive (VPIP 30-50%, AF>2)
    TAG = "tag"            # Tight-aggressive (VPIP 15-30%, AF>2)
    ROCK = "rock"          # Very tight, passive (VPIP<15%, AF<1.5)
    FISH = "fish"          # Loose, passive (VPIP>40%, AF<1.5)
    NIT = "nit"            # Ultra-tight (VPIP<12%)
    CALLING_STATION = "calling_station"  # Calls too much, rarely raises
    BALANCED = "balanced"  # GTO-ish balanced play
    UNKNOWN = "unknown"    # Not enough data


@dataclass
class HandAction:
    """Records a single action in a hand."""
    street: Street
    action_type: ActionType
    amount: float = 0.0
    is_facing_bet: bool = False
    is_facing_raise: bool = False
    pot_size: float = 0.0
    position: int = 0  # 0=BTN/SB, 1=BB, 2=UTG, etc.


@dataclass
class HandResult:
    """Records the result of a completed hand."""
    hand_id: int
    player_id: str
    position: int
    hole_cards: Optional[Tuple[int, int]] = None

    # Actions taken
    actions: List[HandAction] = field(default_factory=list)

    # Outcome
    went_to_showdown: bool = False
    won_at_showdown: bool = False
    won_without_showdown: bool = False
    profit: float = 0.0

    # Preflop specifics
    voluntarily_put_money: bool = False  # VPIP
    raised_preflop: bool = False         # PFR
    three_bet: bool = False              # 3-bet
    faced_three_bet: bool = False
    folded_to_three_bet: bool = False
    four_bet: bool = False

    # Postflop specifics
    cbet_opportunity: bool = False       # Had chance to c-bet
    cbet_made: bool = False              # Made c-bet
    faced_cbet: bool = False
    folded_to_cbet: bool = False

    # Street reached
    saw_flop: bool = False
    saw_turn: bool = False
    saw_river: bool = False


@dataclass
class PlayerStatistics:
    """
    Comprehensive HUD-style player statistics.

    All percentages stored as 0-1 floats for calculations,
    displayed as percentages when needed.
    """
    player_id: str

    # Sample size
    hands_played: int = 0
    hands_won: int = 0

    # Preflop stats
    vpip_hands: int = 0          # Voluntarily put $ in pot
    pfr_hands: int = 0           # Pre-flop raise
    three_bet_opportunities: int = 0
    three_bet_made: int = 0
    faced_three_bet: int = 0
    folded_to_three_bet: int = 0
    four_bet_opportunities: int = 0
    four_bet_made: int = 0

    # Postflop stats
    flops_seen: int = 0
    cbet_opportunities: int = 0
    cbet_made: int = 0
    faced_cbet: int = 0
    folded_to_cbet: int = 0

    # Aggression tracking
    total_bets: int = 0          # Bets + Raises
    total_calls: int = 0
    total_checks: int = 0
    total_folds: int = 0

    # Street-specific aggression
    preflop_aggression: int = 0
    flop_aggression: int = 0
    turn_aggression: int = 0
    river_aggression: int = 0

    # Showdown stats
    went_to_showdown: int = 0    # WTSD
    won_at_showdown: int = 0     # W$SD
    won_without_showdown: int = 0

    # Position stats (by position index)
    hands_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    vpip_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    pfr_by_position: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    profit_by_position: Dict[int, float] = field(default_factory=lambda: defaultdict(float))

    # Profit tracking
    total_profit: float = 0.0
    biggest_win: float = 0.0
    biggest_loss: float = 0.0

    # Recent performance (rolling window)
    recent_results: List[float] = field(default_factory=list)
    recent_window: int = 100

    # Classified personality
    personality: PersonalityType = PersonalityType.UNKNOWN
    personality_confidence: float = 0.0

    @property
    def vpip(self) -> float:
        """Voluntarily Put $ In Pot percentage."""
        return self.vpip_hands / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise percentage."""
        return self.pfr_hands / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def three_bet_pct(self) -> float:
        """3-bet percentage when given the opportunity."""
        return self.three_bet_made / self.three_bet_opportunities if self.three_bet_opportunities > 0 else 0.0

    @property
    def fold_to_three_bet(self) -> float:
        """Fold to 3-bet percentage."""
        return self.folded_to_three_bet / self.faced_three_bet if self.faced_three_bet > 0 else 0.0

    @property
    def cbet_pct(self) -> float:
        """Continuation bet percentage."""
        return self.cbet_made / self.cbet_opportunities if self.cbet_opportunities > 0 else 0.0

    @property
    def fold_to_cbet(self) -> float:
        """Fold to c-bet percentage."""
        return self.folded_to_cbet / self.faced_cbet if self.faced_cbet > 0 else 0.0

    @property
    def aggression_factor(self) -> float:
        """
        Aggression Factor (AF) = (Bets + Raises) / Calls
        Higher = more aggressive. Typical: 1-3 for most players.
        """
        return self.total_bets / self.total_calls if self.total_calls > 0 else float('inf')

    @property
    def aggression_frequency(self) -> float:
        """
        Aggression Frequency = (Bets + Raises) / (Bets + Raises + Calls + Checks)
        Percentage of non-fold actions that are aggressive.
        """
        total_actions = self.total_bets + self.total_calls + self.total_checks
        return self.total_bets / total_actions if total_actions > 0 else 0.0

    @property
    def wtsd(self) -> float:
        """Went To ShowDown percentage (of hands that saw flop)."""
        return self.went_to_showdown / self.flops_seen if self.flops_seen > 0 else 0.0

    @property
    def wsd(self) -> float:
        """Won at ShowDown percentage."""
        return self.won_at_showdown / self.went_to_showdown if self.went_to_showdown > 0 else 0.0

    @property
    def wwsf(self) -> float:
        """Won When Saw Flop percentage."""
        total_wins = self.won_at_showdown + self.won_without_showdown
        return total_wins / self.flops_seen if self.flops_seen > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Overall win rate (hands won / hands played)."""
        return self.hands_won / self.hands_played if self.hands_played > 0 else 0.0

    @property
    def bb_per_100(self) -> float:
        """Big blinds won per 100 hands (assuming 1 BB = 2 chips)."""
        if self.hands_played == 0:
            return 0.0
        return (self.total_profit / 2) / self.hands_played * 100

    @property
    def recent_bb_per_100(self) -> float:
        """BB/100 for recent hands only."""
        if not self.recent_results:
            return 0.0
        return (sum(self.recent_results) / 2) / len(self.recent_results) * 100

    def update_from_hand(self, result: HandResult) -> None:
        """Update statistics from a completed hand result."""
        self.hands_played += 1

        # Position tracking
        self.hands_by_position[result.position] += 1

        # VPIP / PFR
        if result.voluntarily_put_money:
            self.vpip_hands += 1
            self.vpip_by_position[result.position] += 1

        if result.raised_preflop:
            self.pfr_hands += 1
            self.pfr_by_position[result.position] += 1

        # 3-bet tracking
        if result.faced_three_bet:
            self.three_bet_opportunities += 1
        if result.three_bet:
            self.three_bet_made += 1
        if result.faced_three_bet:
            self.faced_three_bet += 1
        if result.folded_to_three_bet:
            self.folded_to_three_bet += 1

        # C-bet tracking
        if result.cbet_opportunity:
            self.cbet_opportunities += 1
        if result.cbet_made:
            self.cbet_made += 1
        if result.faced_cbet:
            self.faced_cbet += 1
        if result.folded_to_cbet:
            self.folded_to_cbet += 1

        # Street tracking
        if result.saw_flop:
            self.flops_seen += 1

        # Showdown tracking
        if result.went_to_showdown:
            self.went_to_showdown += 1
        if result.won_at_showdown:
            self.won_at_showdown += 1
            self.hands_won += 1
        if result.won_without_showdown:
            self.won_without_showdown += 1
            self.hands_won += 1

        # Action counting
        for action in result.actions:
            if action.action_type == ActionType.BET or action.action_type == ActionType.RAISE or action.action_type == ActionType.ALLIN:
                self.total_bets += 1
                if action.street == Street.PREFLOP:
                    self.preflop_aggression += 1
                elif action.street == Street.FLOP:
                    self.flop_aggression += 1
                elif action.street == Street.TURN:
                    self.turn_aggression += 1
                elif action.street == Street.RIVER:
                    self.river_aggression += 1
            elif action.action_type == ActionType.CALL:
                self.total_calls += 1
            elif action.action_type == ActionType.CHECK:
                self.total_checks += 1
            elif action.action_type == ActionType.FOLD:
                self.total_folds += 1

        # Profit tracking
        self.total_profit += result.profit
        self.profit_by_position[result.position] += result.profit

        if result.profit > self.biggest_win:
            self.biggest_win = result.profit
        if result.profit < self.biggest_loss:
            self.biggest_loss = result.profit

        # Recent results (rolling window)
        self.recent_results.append(result.profit)
        if len(self.recent_results) > self.recent_window:
            self.recent_results.pop(0)

        # Update personality classification
        self._classify_personality()

    def _classify_personality(self) -> None:
        """Automatically classify player personality based on statistics."""
        # Need minimum sample size for reliable classification
        min_hands = 30

        if self.hands_played < min_hands:
            self.personality = PersonalityType.UNKNOWN
            self.personality_confidence = self.hands_played / min_hands
            return

        vpip = self.vpip
        pfr = self.pfr
        af = self.aggression_factor if self.aggression_factor != float('inf') else 5.0

        # Classification logic based on VPIP, PFR, and AF
        # VPIP thresholds: <15% tight, 15-30% normal, 30-50% loose, >50% very loose
        # AF thresholds: <1.5 passive, 1.5-2.5 normal, >2.5 aggressive, >4 very aggressive

        if vpip < 0.12:
            # Ultra-tight
            self.personality = PersonalityType.NIT
        elif vpip < 0.15:
            # Very tight
            if af >= 2.0:
                self.personality = PersonalityType.TAG
            else:
                self.personality = PersonalityType.ROCK
        elif vpip < 0.30:
            # Normal range
            if af >= 2.0:
                self.personality = PersonalityType.TAG
            elif af >= 1.0:
                self.personality = PersonalityType.BALANCED
            else:
                self.personality = PersonalityType.CALLING_STATION
        elif vpip < 0.50:
            # Loose
            if af >= 2.5:
                self.personality = PersonalityType.LAG
            elif af >= 1.5:
                self.personality = PersonalityType.BALANCED
            else:
                self.personality = PersonalityType.FISH
        else:
            # Very loose (>50%)
            if af >= 3.0:
                self.personality = PersonalityType.MANIAC
            elif af >= 2.0:
                self.personality = PersonalityType.LAG
            else:
                self.personality = PersonalityType.FISH

        # Confidence increases with sample size (max at 200 hands)
        self.personality_confidence = min(1.0, self.hands_played / 200)

    def to_dict(self) -> Dict[str, Any]:
        """Export statistics as dictionary."""
        return {
            "player_id": self.player_id,
            "hands_played": self.hands_played,
            "stats": {
                "vpip": round(self.vpip * 100, 1),
                "pfr": round(self.pfr * 100, 1),
                "three_bet": round(self.three_bet_pct * 100, 1),
                "fold_to_3bet": round(self.fold_to_three_bet * 100, 1),
                "cbet": round(self.cbet_pct * 100, 1),
                "fold_to_cbet": round(self.fold_to_cbet * 100, 1),
                "af": round(self.aggression_factor, 2) if self.aggression_factor != float('inf') else "inf",
                "agg_freq": round(self.aggression_frequency * 100, 1),
                "wtsd": round(self.wtsd * 100, 1),
                "wsd": round(self.wsd * 100, 1),
                "wwsf": round(self.wwsf * 100, 1),
            },
            "profit": {
                "total": round(self.total_profit, 2),
                "bb_per_100": round(self.bb_per_100, 2),
                "recent_bb_per_100": round(self.recent_bb_per_100, 2),
                "biggest_win": round(self.biggest_win, 2),
                "biggest_loss": round(self.biggest_loss, 2),
            },
            "personality": {
                "type": self.personality.value,
                "confidence": round(self.personality_confidence * 100, 1),
            }
        }

    def get_hud_display(self) -> str:
        """Get compact HUD-style display string."""
        af_str = f"{self.aggression_factor:.1f}" if self.aggression_factor != float('inf') else "∞"
        return (
            f"[{self.player_id}] "
            f"VPIP:{self.vpip*100:.0f}% "
            f"PFR:{self.pfr*100:.0f}% "
            f"AF:{af_str} "
            f"3B:{self.three_bet_pct*100:.0f}% "
            f"CB:{self.cbet_pct*100:.0f}% "
            f"WTSD:{self.wtsd*100:.0f}% "
            f"({self.hands_played}h)"
        )

    def __str__(self) -> str:
        return self.get_hud_display()


class OpponentModel:
    """
    Real-time opponent modeling system.

    Tracks statistics for all opponents and provides adaptive
    recommendations for exploitative play.
    """

    def __init__(self):
        self.players: Dict[str, PlayerStatistics] = {}
        self.hand_counter: int = 0
        self.session_start: datetime = datetime.now()

    def get_or_create_player(self, player_id: str) -> PlayerStatistics:
        """Get existing player stats or create new entry."""
        if player_id not in self.players:
            self.players[player_id] = PlayerStatistics(player_id=player_id)
        return self.players[player_id]

    def record_hand(self, results: List[HandResult]) -> None:
        """Record results from a completed hand for all players."""
        self.hand_counter += 1
        for result in results:
            player = self.get_or_create_player(result.player_id)
            player.update_from_hand(result)

    def get_player_stats(self, player_id: str) -> Optional[PlayerStatistics]:
        """Get statistics for a specific player."""
        return self.players.get(player_id)

    def get_all_stats(self) -> Dict[str, PlayerStatistics]:
        """Get statistics for all tracked players."""
        return self.players

    def get_exploit_recommendations(self, player_id: str) -> Dict[str, str]:
        """
        Get exploitative strategy recommendations based on player tendencies.
        """
        stats = self.players.get(player_id)
        if not stats or stats.hands_played < 20:
            return {"status": "insufficient_data", "hands_needed": 20 - (stats.hands_played if stats else 0)}

        recommendations = {}

        # Exploit based on personality
        personality = stats.personality

        if personality == PersonalityType.FISH or personality == PersonalityType.CALLING_STATION:
            recommendations["value_betting"] = "Bet larger for value - this player calls too much"
            recommendations["bluffing"] = "Reduce bluff frequency - this player rarely folds"
            recommendations["hand_selection"] = "Play more hands in position - they'll pay off your value"

        elif personality == PersonalityType.ROCK or personality == PersonalityType.NIT:
            recommendations["stealing"] = "Steal their blinds frequently - they fold too much"
            recommendations["respect_aggression"] = "When they bet/raise, give them credit for strong hands"
            recommendations["bluffing"] = "Bluff more against them preflop"

        elif personality == PersonalityType.MANIAC:
            recommendations["value_betting"] = "Call down lighter - they bluff too much"
            recommendations["trapping"] = "Slow play strong hands more often"
            recommendations["position"] = "Try to play in position against them"

        elif personality == PersonalityType.LAG:
            recommendations["3betting"] = "3-bet wider for value and as a bluff"
            recommendations["calling"] = "Call with medium strength hands more often"

        elif personality == PersonalityType.TAG:
            recommendations["respect"] = "Respect their raises - they usually have it"
            recommendations["position"] = "Avoid marginal spots out of position"

        # Specific stat-based recommendations
        if stats.fold_to_cbet > 0.60:
            recommendations["cbet"] = f"C-bet more - they fold {stats.fold_to_cbet*100:.0f}% to c-bets"
        elif stats.fold_to_cbet < 0.35:
            recommendations["cbet"] = f"C-bet only for value - they only fold {stats.fold_to_cbet*100:.0f}%"

        if stats.fold_to_three_bet > 0.70:
            recommendations["3bet_bluff"] = f"3-bet bluff more - they fold {stats.fold_to_three_bet*100:.0f}% to 3-bets"

        if stats.cbet_pct > 0.75:
            recommendations["vs_cbet"] = f"Float/raise their c-bets more - they c-bet {stats.cbet_pct*100:.0f}%"

        return recommendations

    def compare_players(self, player_ids: List[str]) -> Dict[str, Any]:
        """Compare statistics between multiple players."""
        comparison = {"players": []}

        for pid in player_ids:
            stats = self.players.get(pid)
            if stats:
                comparison["players"].append(stats.to_dict())

        if len(comparison["players"]) >= 2:
            # Calculate rankings
            metrics = ["vpip", "pfr", "af", "bb_per_100"]
            rankings = {}

            for metric in metrics:
                if metric == "af":
                    values = [(p["player_id"], p["stats"]["af"] if p["stats"]["af"] != "inf" else 99)
                              for p in comparison["players"]]
                elif metric == "bb_per_100":
                    values = [(p["player_id"], p["profit"]["bb_per_100"])
                              for p in comparison["players"]]
                else:
                    values = [(p["player_id"], p["stats"][metric])
                              for p in comparison["players"]]

                sorted_values = sorted(values, key=lambda x: x[1], reverse=True)
                rankings[metric] = [v[0] for v in sorted_values]

            comparison["rankings"] = rankings

        return comparison

    def export_session(self, filepath: str) -> None:
        """Export all statistics to JSON file."""
        data = {
            "session_start": self.session_start.isoformat(),
            "total_hands": self.hand_counter,
            "players": {pid: stats.to_dict() for pid, stats in self.players.items()}
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_session(self, filepath: str) -> None:
        """Import statistics from JSON file (for continuing sessions)."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.hand_counter = data.get("total_hands", 0)

        # Note: This is a simplified import - full implementation would
        # reconstruct PlayerStatistics objects from the saved data
        print(f"Imported session with {len(data.get('players', {}))} players")

    def get_summary_report(self) -> str:
        """Generate text summary report of all players."""
        lines = [
            "=" * 60,
            "OPPONENT MODELING SUMMARY REPORT",
            "=" * 60,
            f"Session Duration: {datetime.now() - self.session_start}",
            f"Total Hands Tracked: {self.hand_counter}",
            f"Players Tracked: {len(self.players)}",
            "",
            "PLAYER STATISTICS:",
            "-" * 60,
        ]

        # Sort by hands played
        sorted_players = sorted(
            self.players.values(),
            key=lambda p: p.hands_played,
            reverse=True
        )

        for stats in sorted_players:
            lines.append(stats.get_hud_display())
            lines.append(f"  Personality: {stats.personality.value} ({stats.personality_confidence*100:.0f}% confidence)")
            lines.append(f"  Profit: {stats.total_profit:+.0f} chips ({stats.bb_per_100:+.1f} BB/100)")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class HandTracker:
    """
    Utility class to track actions during a hand and generate HandResult.

    Use this to integrate with the poker engine's game loop.
    """

    def __init__(self, hand_id: int, num_players: int):
        self.hand_id = hand_id
        self.num_players = num_players
        self.player_results: Dict[str, HandResult] = {}
        self.current_street = Street.PREFLOP
        self.preflop_raiser: Optional[str] = None
        self.raise_count: int = 0
        self.last_aggressor: Optional[str] = None

    def start_hand(self, player_ids: List[str], positions: List[int]) -> None:
        """Initialize tracking for a new hand."""
        for pid, pos in zip(player_ids, positions):
            self.player_results[pid] = HandResult(
                hand_id=self.hand_id,
                player_id=pid,
                position=pos,
                actions=[]
            )

    def set_hole_cards(self, player_id: str, cards: Tuple[int, int]) -> None:
        """Set hole cards for a player."""
        if player_id in self.player_results:
            self.player_results[player_id].hole_cards = cards

    def advance_street(self, street: Street) -> None:
        """Move to the next street."""
        self.current_street = street
        self.raise_count = 0

        if street == Street.FLOP:
            for result in self.player_results.values():
                if not result.actions or result.actions[-1].action_type != ActionType.FOLD:
                    result.saw_flop = True
        elif street == Street.TURN:
            for result in self.player_results.values():
                if result.saw_flop and (not result.actions or result.actions[-1].action_type != ActionType.FOLD):
                    result.saw_turn = True
        elif street == Street.RIVER:
            for result in self.player_results.values():
                if result.saw_turn and (not result.actions or result.actions[-1].action_type != ActionType.FOLD):
                    result.saw_river = True

    def record_action(
        self,
        player_id: str,
        action_type: ActionType,
        amount: float = 0.0,
        pot_size: float = 0.0,
        is_facing_bet: bool = False,
        is_facing_raise: bool = False
    ) -> None:
        """Record an action taken by a player."""
        if player_id not in self.player_results:
            return

        result = self.player_results[player_id]
        position = result.position

        action = HandAction(
            street=self.current_street,
            action_type=action_type,
            amount=amount,
            is_facing_bet=is_facing_bet,
            is_facing_raise=is_facing_raise,
            pot_size=pot_size,
            position=position
        )
        result.actions.append(action)

        # Preflop tracking
        if self.current_street == Street.PREFLOP:
            if action_type in [ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALLIN]:
                result.voluntarily_put_money = True

            if action_type in [ActionType.RAISE, ActionType.BET, ActionType.ALLIN]:
                self.raise_count += 1

                if self.raise_count == 1:
                    result.raised_preflop = True
                    self.preflop_raiser = player_id
                elif self.raise_count == 2:
                    result.three_bet = True
                    # Mark others as faced 3-bet
                    for pid, res in self.player_results.items():
                        if pid != player_id and res.voluntarily_put_money:
                            res.faced_three_bet = True
                elif self.raise_count == 3:
                    result.four_bet = True

            if is_facing_raise and self.raise_count >= 2 and action_type == ActionType.FOLD:
                result.folded_to_three_bet = True

        # Postflop c-bet tracking
        if self.current_street == Street.FLOP:
            if self.preflop_raiser == player_id and self.last_aggressor is None:
                result.cbet_opportunity = True
                if action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALLIN]:
                    result.cbet_made = True
                    self.last_aggressor = player_id
            elif self.last_aggressor is not None and self.last_aggressor != player_id:
                result.faced_cbet = True
                if action_type == ActionType.FOLD:
                    result.folded_to_cbet = True

        # Track aggressor
        if action_type in [ActionType.BET, ActionType.RAISE, ActionType.ALLIN]:
            self.last_aggressor = player_id

    def end_hand(
        self,
        winner_ids: List[str],
        profits: Dict[str, float],
        went_to_showdown: bool = False
    ) -> List[HandResult]:
        """Finalize hand and return results for all players."""
        for player_id, result in self.player_results.items():
            result.profit = profits.get(player_id, 0.0)

            if went_to_showdown and result.saw_flop:
                # Check if player was still in the hand
                if not result.actions or result.actions[-1].action_type != ActionType.FOLD:
                    result.went_to_showdown = True
                    if player_id in winner_ids:
                        result.won_at_showdown = True
            elif player_id in winner_ids:
                result.won_without_showdown = True

        return list(self.player_results.values())


# ============================================================
# Agent Comparison & Analysis Tools
# ============================================================

@dataclass
class AgentProfile:
    """Extended agent profile with personality and performance metrics."""
    agent_id: str
    agent_type: str  # "neural", "rule_based", "random", etc.
    personality: PersonalityType = PersonalityType.UNKNOWN
    elo_rating: float = 1200.0

    # Lifetime stats
    total_hands: int = 0
    total_profit: float = 0.0

    # Performance metrics
    avg_bb_per_100: float = 0.0
    win_rate: float = 0.0

    # Style metrics
    avg_vpip: float = 0.0
    avg_pfr: float = 0.0
    avg_af: float = 0.0

    # Matchup history
    head_to_head: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AgentAnalyzer:
    """Tools for comparing and analyzing agents."""

    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
        self.opponent_model = OpponentModel()

    def register_agent(self, agent_id: str, agent_type: str) -> AgentProfile:
        """Register a new agent for tracking."""
        profile = AgentProfile(agent_id=agent_id, agent_type=agent_type)
        self.agents[agent_id] = profile
        return profile

    def update_from_match(
        self,
        agent_ids: List[str],
        profits: Dict[str, float],
        hands_played: int
    ) -> None:
        """Update agent profiles from match results."""
        for aid in agent_ids:
            if aid not in self.agents:
                self.register_agent(aid, "unknown")

            profile = self.agents[aid]
            profile.total_hands += hands_played
            profile.total_profit += profits.get(aid, 0.0)

            # Update head-to-head
            for other_aid in agent_ids:
                if other_aid != aid:
                    if other_aid not in profile.head_to_head:
                        profile.head_to_head[other_aid] = {"hands": 0, "profit": 0.0}
                    profile.head_to_head[other_aid]["hands"] += hands_played
                    profile.head_to_head[other_aid]["profit"] += profits.get(aid, 0.0)

    def update_elo(
        self,
        winner_id: str,
        loser_id: str,
        k_factor: float = 32.0
    ) -> Tuple[float, float]:
        """Update ELO ratings after a match."""
        if winner_id not in self.agents:
            self.register_agent(winner_id, "unknown")
        if loser_id not in self.agents:
            self.register_agent(loser_id, "unknown")

        winner = self.agents[winner_id]
        loser = self.agents[loser_id]

        # Calculate expected scores
        exp_winner = 1 / (1 + 10 ** ((loser.elo_rating - winner.elo_rating) / 400))
        exp_loser = 1 - exp_winner

        # Update ratings
        winner.elo_rating += k_factor * (1 - exp_winner)
        loser.elo_rating += k_factor * (0 - exp_loser)

        return winner.elo_rating, loser.elo_rating

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get sorted leaderboard of agents."""
        sorted_agents = sorted(
            self.agents.values(),
            key=lambda a: a.elo_rating,
            reverse=True
        )

        return [
            {
                "rank": i + 1,
                "agent_id": a.agent_id,
                "agent_type": a.agent_type,
                "elo": round(a.elo_rating, 1),
                "total_hands": a.total_hands,
                "total_profit": round(a.total_profit, 2),
                "personality": a.personality.value
            }
            for i, a in enumerate(sorted_agents)
        ]

    def get_matchup_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get head-to-head profit matrix between all agents."""
        matrix = {}
        for aid, profile in self.agents.items():
            matrix[aid] = {}
            for other_id, h2h in profile.head_to_head.items():
                if h2h["hands"] > 0:
                    matrix[aid][other_id] = h2h["profit"] / h2h["hands"] * 100  # Per 100 hands
        return matrix

    def generate_comparison_report(self, agent_ids: List[str]) -> str:
        """Generate detailed comparison report between agents."""
        lines = [
            "=" * 60,
            "AGENT COMPARISON REPORT",
            "=" * 60,
            "",
        ]

        for aid in agent_ids:
            if aid not in self.agents:
                lines.append(f"Agent {aid}: Not found")
                continue

            profile = self.agents[aid]
            lines.extend([
                f"Agent: {profile.agent_id}",
                f"  Type: {profile.agent_type}",
                f"  ELO Rating: {profile.elo_rating:.0f}",
                f"  Personality: {profile.personality.value}",
                f"  Total Hands: {profile.total_hands}",
                f"  Total Profit: {profile.total_profit:+.0f}",
                f"  Style: VPIP {profile.avg_vpip*100:.1f}% / PFR {profile.avg_pfr*100:.1f}% / AF {profile.avg_af:.1f}",
                ""
            ])

        # Head-to-head section
        if len(agent_ids) >= 2:
            lines.extend([
                "HEAD-TO-HEAD:",
                "-" * 40,
            ])

            for i, aid1 in enumerate(agent_ids):
                for aid2 in agent_ids[i+1:]:
                    if aid1 in self.agents and aid2 in self.agents:
                        p1 = self.agents[aid1]
                        p2 = self.agents[aid2]

                        h2h1 = p1.head_to_head.get(aid2, {"hands": 0, "profit": 0})
                        h2h2 = p2.head_to_head.get(aid1, {"hands": 0, "profit": 0})

                        if h2h1["hands"] > 0:
                            profit_per_100 = h2h1["profit"] / h2h1["hands"] * 100
                            lines.append(f"  {aid1} vs {aid2}: {profit_per_100:+.1f}/100 hands ({h2h1['hands']} hands)")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# Integration Helper Functions
# ============================================================

def action_id_to_type(action_id: int, is_facing_bet: bool = False) -> ActionType:
    """Convert 11-action ID to ActionType."""
    if action_id == 0:
        return ActionType.FOLD
    elif action_id == 1:
        return ActionType.CALL if is_facing_bet else ActionType.CHECK
    elif action_id == 10:
        return ActionType.ALLIN
    else:
        return ActionType.RAISE if is_facing_bet else ActionType.BET


def create_opponent_model() -> OpponentModel:
    """Factory function to create a new opponent model."""
    return OpponentModel()


def create_agent_analyzer() -> AgentAnalyzer:
    """Factory function to create a new agent analyzer."""
    return AgentAnalyzer()


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    print("Poker AI - Personality Analysis Module Demo")
    print("=" * 50)

    # Create opponent model
    model = OpponentModel()

    # Simulate some hands
    import random

    player_ids = ["Player_A", "Player_B", "Player_C"]

    for hand_num in range(100):
        tracker = HandTracker(hand_num, len(player_ids))
        positions = list(range(len(player_ids)))
        random.shuffle(positions)

        tracker.start_hand(player_ids, positions)

        # Simulate preflop actions
        for pid in player_ids:
            action = random.choices(
                [ActionType.FOLD, ActionType.CALL, ActionType.RAISE],
                weights=[0.3, 0.4, 0.3]
            )[0]
            tracker.record_action(pid, action, pot_size=3.0)

        # Determine winner (simplified)
        active_players = [pid for pid in player_ids]
        winner = random.choice(active_players)

        profits = {pid: -10.0 for pid in player_ids}
        profits[winner] = 20.0

        results = tracker.end_hand([winner], profits, went_to_showdown=random.random() > 0.5)
        model.record_hand(results)

    # Print report
    print(model.get_summary_report())

    # Test exploit recommendations
    print("\nExploit Recommendations:")
    for pid in player_ids:
        print(f"\n{pid}:")
        recs = model.get_exploit_recommendations(pid)
        for key, value in recs.items():
            print(f"  {key}: {value}")

    # Test agent analyzer
    print("\n" + "=" * 50)
    print("Agent Analyzer Demo")
    print("=" * 50)

    analyzer = AgentAnalyzer()

    # Register agents
    analyzer.register_agent("NeuralNet_v1", "neural")
    analyzer.register_agent("RuleBased_TAG", "rule_based")
    analyzer.register_agent("Random_Agent", "random")

    # Simulate matches
    for _ in range(50):
        aid1, aid2 = random.sample(list(analyzer.agents.keys()), 2)
        winner = random.choice([aid1, aid2])
        loser = aid1 if winner == aid2 else aid2

        analyzer.update_elo(winner, loser)
        analyzer.update_from_match(
            [aid1, aid2],
            {aid1: random.uniform(-50, 50), aid2: random.uniform(-50, 50)},
            hands_played=100
        )

    # Print leaderboard
    print("\nLeaderboard:")
    for entry in analyzer.get_leaderboard():
        print(f"  #{entry['rank']} {entry['agent_id']}: ELO {entry['elo']}, {entry['total_hands']} hands")

    # Print comparison
    print(analyzer.generate_comparison_report(list(analyzer.agents.keys())))
