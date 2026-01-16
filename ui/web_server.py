"""Web-based poker UI using FastAPI.

This module provides:
1. RESTful API for game state
2. WebSocket for real-time updates
3. Simple HTML/JS frontend
4. HUD-style personality statistics display (Phase 4)
"""

import os
import sys
import json
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_engine

try:
    from brain.personality_agents import RuleBasedAgent, PersonalityType
except ImportError:
    RuleBasedAgent = None
    PersonalityType = None

try:
    from brain.personality_analysis import (
        OpponentModel,
        HandTracker,
        PlayerStatistics,
        ActionType,
        Street,
        action_id_to_type,
    )
    PERSONALITY_TRACKING_AVAILABLE = True
except ImportError:
    PERSONALITY_TRACKING_AVAILABLE = False
    OpponentModel = None
    HandTracker = None


# Pydantic models for API
class GameConfig(BaseModel):
    num_players: int = 6
    starting_stack: int = 10000
    small_blind: int = 50
    big_blind: int = 100
    human_seat: int = 0


class ActionRequest(BaseModel):
    action: int  # 0=fold, 1=call/check, 2=bet/raise, 3=all-in
    amount: int = 0


# Game session management
class GameSession:
    """Manages a single game session with personality tracking."""

    def __init__(self, session_id: str, config: GameConfig):
        self.session_id = session_id
        self.config = config
        self.env = poker_engine.PokerEnv(
            num_players=config.num_players,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            starting_stack=config.starting_stack,
        )
        self.human_seat = config.human_seat
        self.current_obs = None
        self.hand_number = 0
        self.total_profit = 0.0
        self.ai_agents: Dict[int, RuleBasedAgent] = {}
        self.ai_names: Dict[int, str] = {}

        # Phase 4: Personality tracking
        self.opponent_model = OpponentModel() if PERSONALITY_TRACKING_AVAILABLE else None
        self.hand_tracker: Optional[HandTracker] = None
        self.current_street = Street.PREFLOP if PERSONALITY_TRACKING_AVAILABLE else None
        self.player_ids: Dict[int, str] = {}  # seat -> player_id string

        self._setup_ai()

    def _setup_ai(self):
        """Setup AI opponents."""
        # Setup player IDs for personality tracking
        self.player_ids[self.human_seat] = "You"

        if RuleBasedAgent is None:
            for seat in range(self.config.num_players):
                if seat != self.human_seat:
                    self.player_ids[seat] = f"AI_{seat}"
                    self.ai_names[seat] = f"AI {seat}"
            return

        personalities = [
            PersonalityType.TAG,
            PersonalityType.LAG,
            PersonalityType.ROCK,
            PersonalityType.FISH,
            PersonalityType.MANIAC,
            PersonalityType.BALANCED,
            PersonalityType.TAG,
            PersonalityType.LAG,
        ]

        for seat in range(self.config.num_players):
            if seat == self.human_seat:
                continue

            idx = seat if seat < self.human_seat else seat - 1
            personality = personalities[idx % len(personalities)]

            agent = RuleBasedAgent(
                personality=personality,
                player_id=seat,
                seed=int(time.time()) + seat * 100,
            )
            self.ai_agents[seat] = agent
            self.ai_names[seat] = personality.value.upper()
            self.player_ids[seat] = f"{personality.value.upper()}_{seat}"

    def start_hand(self) -> Dict:
        """Start a new hand."""
        self.hand_number += 1
        self.current_obs = self.env.reset()

        # Initialize hand tracker for personality analysis
        if self.opponent_model and PERSONALITY_TRACKING_AVAILABLE:
            self.hand_tracker = HandTracker(self.hand_number, self.config.num_players)
            player_ids_list = [self.player_ids.get(i, f"Player_{i}") for i in range(self.config.num_players)]
            positions = list(range(self.config.num_players))
            self.hand_tracker.start_hand(player_ids_list, positions)
            self.current_street = Street.PREFLOP

        return self._get_state()

    def _parse_cards(self, cards) -> List[Dict]:
        """Parse cards to dict format."""
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
                result.append({"rank": int(r), "suit": int(s)})
            elif hasattr(card, 'rank') and hasattr(card, 'suit'):
                result.append({"rank": card.rank, "suit": card.suit})
        return result

    def _get_state(self) -> Dict:
        """Get current game state as dict."""
        obs = self.current_obs

        state = {
            "session_id": self.session_id,
            "hand_number": self.hand_number,
            "street": obs.get("street", 0),
            "pot": obs.get("pot", 0),
            "board": self._parse_cards(obs.get("board", [])),
            "hole_cards": self._parse_cards(obs.get("hole_cards", [])),
            "stacks": obs.get("stacks", []),
            "active_players": obs.get("active_players", []),
            "bets_this_round": obs.get("bets_this_round", [0] * self.config.num_players),
            "current_player": obs.get("current_player", -1),
            "to_call": obs.get("to_call", 0),
            "action_mask": obs.get("action_mask", [True] * 4),
            "is_terminal": obs.get("is_terminal", False),
            "is_human_turn": obs.get("current_player", -1) == self.human_seat,
            "human_seat": self.human_seat,
            "button": obs.get("button", 0),
            "total_profit": self.total_profit,
            "ai_names": self.ai_names,
        }

        # Add personality stats for each player
        if self.opponent_model:
            player_stats = {}
            for seat in range(self.config.num_players):
                player_id = self.player_ids.get(seat, f"Player_{seat}")
                stats = self.opponent_model.get_player_stats(player_id)
                if stats and stats.hands_played > 0:
                    player_stats[seat] = {
                        "vpip": round(stats.vpip * 100, 1),
                        "pfr": round(stats.pfr * 100, 1),
                        "af": round(stats.aggression_factor, 1) if stats.aggression_factor != float('inf') else 99.9,
                        "hands": stats.hands_played,
                        "personality": stats.personality.value,
                        "confidence": round(stats.personality_confidence * 100, 0),
                        "three_bet": round(stats.three_bet_pct * 100, 1),
                        "cbet": round(stats.cbet_pct * 100, 1),
                        "wtsd": round(stats.wtsd * 100, 1),
                        "wsd": round(stats.wsd * 100, 1),
                        "bb_per_100": round(stats.bb_per_100, 1),
                    }
                else:
                    player_stats[seat] = None
            state["player_stats"] = player_stats

        return state

    def _track_action(self, seat: int, action: int):
        """Track an action for personality analysis."""
        if not self.hand_tracker or not PERSONALITY_TRACKING_AVAILABLE:
            return

        player_id = self.player_ids.get(seat, f"Player_{seat}")
        is_facing_bet = self.current_obs.get("to_call", 0) > 0
        action_type = action_id_to_type(action, is_facing_bet)
        pot_size = self.current_obs.get("pot", 0)

        # Check for street change
        street_idx = self.current_obs.get("street", 0)
        new_street = Street(street_idx) if street_idx < 4 else Street.RIVER
        if new_street != self.current_street:
            self.hand_tracker.advance_street(new_street)
            self.current_street = new_street

        self.hand_tracker.record_action(
            player_id,
            action_type,
            pot_size=pot_size,
            is_facing_bet=is_facing_bet,
        )

    def _finalize_hand(self, rewards: List[float]):
        """Finalize hand tracking and record results."""
        if not self.hand_tracker or not self.opponent_model:
            return

        # Determine winners
        winner_ids = []
        hand_profits = {}
        for seat in range(self.config.num_players):
            player_id = self.player_ids.get(seat, f"Player_{seat}")
            profit = rewards[seat] if seat < len(rewards) else 0
            hand_profits[player_id] = profit
            if profit > 0:
                winner_ids.append(player_id)

        went_to_showdown = self.current_obs.get("street", 0) >= 3

        results = self.hand_tracker.end_hand(winner_ids, hand_profits, went_to_showdown)
        self.opponent_model.record_hand(results)
        self.hand_tracker = None

    def process_action(self, action: int, amount: int = 0) -> Dict:
        """Process a player action.

        Returns:
            Updated game state
        """
        if self.current_obs is None:
            raise ValueError("No hand in progress")

        if self.current_obs.get("is_terminal", False):
            raise ValueError("Hand already complete")

        current_player = self.current_obs.get("current_player", -1)

        # If it's human's turn, use provided action
        if current_player == self.human_seat:
            # Track human action
            self._track_action(self.human_seat, action)
            self.current_obs, rewards, done, _, info = self.env.step(action)
        else:
            # AI turn - this shouldn't happen via API
            raise ValueError("Not human's turn")

        # Continue with AI actions until human's turn or hand ends
        while not self.current_obs.get("is_terminal", False):
            current = self.current_obs.get("current_player", -1)

            if current == self.human_seat:
                break

            # AI action
            if current in self.ai_agents:
                ai_action = self.ai_agents[current].get_action(self.current_obs, self.env)
            else:
                # Random fallback
                mask = self.current_obs.get("action_mask", [True] * 4)
                valid = [i for i, v in enumerate(mask) if v]
                import random
                ai_action = random.choice(valid)

            # Track AI action
            self._track_action(current, ai_action)
            self.current_obs, rewards, done, _, info = self.env.step(ai_action)

        # Check if hand ended
        if self.current_obs.get("is_terminal", False):
            self.total_profit += rewards[self.human_seat]
            self._finalize_hand(rewards)

        return self._get_state()


# Global session storage
sessions: Dict[str, GameSession] = {}


# FastAPI app
app = FastAPI(title="Poker AI", description="Play poker against AI")


# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poker AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #1a5a1a 0%, #0d3d0d 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }

        .game-info {
            display: flex;
            justify-content: space-between;
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .game-info div { text-align: center; }
        .game-info .value { font-size: 1.5em; font-weight: bold; color: #ffd700; }

        .table {
            background: #2d5a2d;
            border-radius: 150px;
            padding: 40px;
            margin-bottom: 20px;
            border: 8px solid #4a3728;
            box-shadow: inset 0 0 50px rgba(0,0,0,0.3);
        }

        .board {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 30px;
        }

        .card {
            width: 60px;
            height: 84px;
            background: white;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            font-size: 1.4em;
            font-weight: bold;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
        }
        .card.hidden {
            background: linear-gradient(135deg, #2c5aa0 0%, #1a3a6e 100%);
            color: white;
        }
        .card.hearts, .card.diamonds { color: #d32f2f; }
        .card.clubs, .card.spades { color: #1a1a1a; }
        .card.empty { background: rgba(255,255,255,0.1); border: 2px dashed rgba(255,255,255,0.3); }

        .players {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        .player {
            background: rgba(0,0,0,0.4);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .player.human { border: 2px solid #ffd700; }
        .player.active { border: 2px solid #4caf50; animation: pulse 1s infinite; }
        .player.folded { opacity: 0.5; }
        .player .name { font-weight: bold; margin-bottom: 5px; }
        .player .stack { color: #4caf50; }
        .player .bet { color: #ffd700; font-size: 0.9em; }
        .player .cards { display: flex; justify-content: center; gap: 5px; margin-top: 10px; }
        .player .cards .card { width: 40px; height: 56px; font-size: 0.9em; }

        /* HUD Statistics Display */
        .player .hud {
            margin-top: 8px;
            padding: 6px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
            font-size: 0.75em;
            font-family: 'Courier New', monospace;
        }
        .player .hud .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
        }
        .player .hud .stat-label {
            color: #aaa;
        }
        .player .hud .stat-value {
            color: #fff;
            font-weight: bold;
        }
        .player .hud .stat-value.high { color: #f44336; }
        .player .hud .stat-value.medium { color: #ffc107; }
        .player .hud .stat-value.low { color: #4caf50; }
        .player .hud .personality {
            text-align: center;
            margin-top: 4px;
            padding-top: 4px;
            border-top: 1px solid rgba(255,255,255,0.2);
            color: #64b5f6;
            font-weight: bold;
            text-transform: uppercase;
        }
        .player .hud .hands-count {
            text-align: center;
            color: #888;
            font-size: 0.9em;
        }

        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 5px #4caf50; }
            50% { box-shadow: 0 0 20px #4caf50; }
        }

        .actions {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .actions button {
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.1s;
        }
        .actions button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .actions button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-fold { background: #d32f2f; color: white; }
        .btn-check { background: #1976d2; color: white; }
        .btn-call { background: #388e3c; color: white; }
        .btn-raise { background: #f57c00; color: white; }
        .btn-allin { background: #7b1fa2; color: white; }

        .bet-input {
            display: none;
            margin-top: 15px;
            justify-content: center;
            gap: 10px;
            align-items: center;
        }
        .bet-input.visible { display: flex; }
        .bet-input input {
            padding: 10px;
            font-size: 1.1em;
            width: 120px;
            border-radius: 5px;
            border: none;
        }
        .bet-input button {
            padding: 10px 20px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        .message {
            text-align: center;
            padding: 20px;
            font-size: 1.3em;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            margin-top: 20px;
        }
        .message.win { color: #4caf50; }
        .message.lose { color: #d32f2f; }

        .start-btn {
            display: block;
            margin: 30px auto;
            padding: 20px 50px;
            font-size: 1.3em;
            background: #ffd700;
            color: #1a1a1a;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>♠ ♥ Poker AI ♦ ♣</h1>

        <div class="game-info">
            <div>
                <div class="label">Hand</div>
                <div class="value" id="handNum">0</div>
            </div>
            <div>
                <div class="label">Pot</div>
                <div class="value" id="pot">$0</div>
            </div>
            <div>
                <div class="label">To Call</div>
                <div class="value" id="toCall">$0</div>
            </div>
            <div>
                <div class="label">Session</div>
                <div class="value" id="session">$0</div>
            </div>
        </div>

        <div class="table">
            <div class="board" id="board"></div>
            <div class="players" id="players"></div>
        </div>

        <div class="actions" id="actions" style="display:none;">
            <button class="btn-fold" onclick="doAction(0)">Fold</button>
            <button class="btn-check" id="btnCheck" onclick="doAction(1)">Check</button>
            <button class="btn-raise" onclick="showBet()">Raise</button>
            <button class="btn-allin" onclick="doAction(3)">All-In</button>
        </div>

        <div class="bet-input" id="betInput">
            <input type="number" id="betAmount" placeholder="Amount">
            <button onclick="doRaise()">Confirm</button>
            <button onclick="hideBet()" style="background:#666">Cancel</button>
        </div>

        <div class="message" id="message"></div>

        <button class="start-btn" id="startBtn" onclick="startGame()">Start Game</button>
    </div>

    <script>
        let sessionId = null;
        let gameState = null;

        const SUITS = ['♣', '♦', '♥', '♠'];
        const SUIT_CLASSES = ['clubs', 'diamonds', 'hearts', 'spades'];
        const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];

        function formatCard(card, hidden = false) {
            if (hidden) {
                return '<div class="card hidden">?</div>';
            }
            if (!card) {
                return '<div class="card empty"></div>';
            }
            const rank = RANKS[card.rank] || '?';
            const suit = SUITS[card.suit] || '?';
            const suitClass = SUIT_CLASSES[card.suit] || '';
            return `<div class="card ${suitClass}">${rank}${suit}</div>`;
        }

        function getStatClass(value, lowThresh, highThresh) {
            if (value >= highThresh) return 'high';
            if (value >= lowThresh) return 'medium';
            return 'low';
        }

        function formatHUD(stats) {
            if (!stats || stats.hands < 5) {
                return '<div class="hud"><div class="hands-count">Collecting data...</div></div>';
            }

            const vpipClass = getStatClass(stats.vpip, 25, 40);
            const pfrClass = getStatClass(stats.pfr, 15, 25);
            const afClass = getStatClass(stats.af, 2, 3.5);

            return `
                <div class="hud">
                    <div class="stat-row">
                        <span class="stat-label">VPIP:</span>
                        <span class="stat-value ${vpipClass}">${stats.vpip}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">PFR:</span>
                        <span class="stat-value ${pfrClass}">${stats.pfr}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">AF:</span>
                        <span class="stat-value ${afClass}">${stats.af}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">3B:</span>
                        <span class="stat-value">${stats.three_bet}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">WTSD:</span>
                        <span class="stat-value">${stats.wtsd}%</span>
                    </div>
                    <div class="personality">${stats.personality}</div>
                    <div class="hands-count">${stats.hands} hands</div>
                </div>
            `;
        }

        async function startGame() {
            const response = await fetch('/api/game/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({num_players: 6, starting_stack: 10000})
            });
            const data = await response.json();
            sessionId = data.session_id;
            await newHand();
        }

        async function newHand() {
            const response = await fetch(`/api/game/${sessionId}/new-hand`, {method: 'POST'});
            gameState = await response.json();
            updateUI();
            document.getElementById('startBtn').style.display = 'none';
        }

        async function doAction(action, amount = 0) {
            if (!sessionId || !gameState.is_human_turn) return;

            const response = await fetch(`/api/game/${sessionId}/action`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action, amount})
            });
            gameState = await response.json();
            hideBet();
            updateUI();

            if (gameState.is_terminal) {
                showResult();
            }
        }

        function showBet() {
            document.getElementById('betInput').classList.add('visible');
            document.getElementById('betAmount').focus();
        }

        function hideBet() {
            document.getElementById('betInput').classList.remove('visible');
        }

        function doRaise() {
            const amount = parseInt(document.getElementById('betAmount').value) || 0;
            doAction(2, amount);
        }

        function showResult() {
            const msg = document.getElementById('message');
            const profit = gameState.stacks[gameState.human_seat] - 10000;

            if (profit > 0) {
                msg.className = 'message win';
                msg.textContent = `You won $${profit}! Click to continue...`;
            } else if (profit < 0) {
                msg.className = 'message lose';
                msg.textContent = `You lost $${Math.abs(profit)}. Click to continue...`;
            } else {
                msg.className = 'message';
                msg.textContent = 'Push. Click to continue...';
            }
            msg.onclick = newHand;
        }

        function updateUI() {
            if (!gameState) return;

            // Update info
            document.getElementById('handNum').textContent = gameState.hand_number;
            document.getElementById('pot').textContent = '$' + gameState.pot;
            document.getElementById('toCall').textContent = '$' + gameState.to_call;
            document.getElementById('session').textContent =
                (gameState.total_profit >= 0 ? '+$' : '-$') + Math.abs(gameState.total_profit);

            // Update board
            const boardEl = document.getElementById('board');
            let boardHTML = '';
            for (let i = 0; i < 5; i++) {
                boardHTML += formatCard(gameState.board[i]);
            }
            boardEl.innerHTML = boardHTML;

            // Update players
            const playersEl = document.getElementById('players');
            let playersHTML = '';
            for (let i = 0; i < gameState.stacks.length; i++) {
                const isHuman = i === gameState.human_seat;
                const isActive = gameState.active_players[i];
                const isCurrent = i === gameState.current_player;

                let classes = 'player';
                if (isHuman) classes += ' human';
                if (isCurrent) classes += ' active';
                if (!isActive) classes += ' folded';

                const name = isHuman ? 'You' : (gameState.ai_names[i] || `Player ${i}`);
                const cards = isHuman ? gameState.hole_cards : [{}, {}];

                // Get player stats for HUD
                const stats = gameState.player_stats ? gameState.player_stats[i] : null;
                const hudHTML = !isHuman ? formatHUD(stats) : '';

                playersHTML += `
                    <div class="${classes}">
                        <div class="name">${name}</div>
                        <div class="stack">$${gameState.stacks[i]}</div>
                        ${gameState.bets_this_round[i] > 0 ? `<div class="bet">Bet: $${gameState.bets_this_round[i]}</div>` : ''}
                        <div class="cards">
                            ${formatCard(cards[0], !isHuman && isActive)}
                            ${formatCard(cards[1], !isHuman && isActive)}
                        </div>
                        ${hudHTML}
                    </div>
                `;
            }
            playersEl.innerHTML = playersHTML;

            // Update actions
            const actionsEl = document.getElementById('actions');
            const msgEl = document.getElementById('message');

            if (gameState.is_terminal) {
                actionsEl.style.display = 'none';
            } else if (gameState.is_human_turn) {
                actionsEl.style.display = 'flex';
                msgEl.textContent = 'Your turn!';
                msgEl.className = 'message';
                msgEl.onclick = null;

                // Update button states
                document.querySelector('.btn-fold').disabled = !gameState.action_mask[0];
                document.querySelector('.btn-check').disabled = !gameState.action_mask[1];
                document.querySelector('.btn-check').textContent = gameState.to_call > 0 ? `Call $${gameState.to_call}` : 'Check';
                document.querySelector('.btn-raise').disabled = !gameState.action_mask[2];
                document.querySelector('.btn-allin').disabled = !gameState.action_mask[3];
            } else {
                actionsEl.style.display = 'none';
                msgEl.textContent = 'AI thinking...';
                msgEl.className = 'message';
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    return HTML_TEMPLATE


@app.post("/api/game/start")
async def start_game(config: GameConfig):
    """Start a new game session."""
    session_id = f"game_{int(time.time())}_{len(sessions)}"
    session = GameSession(session_id, config)
    sessions[session_id] = session

    return {"session_id": session_id, "config": config.dict()}


@app.post("/api/game/{session_id}/new-hand")
async def new_hand(session_id: str):
    """Start a new hand."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    state = session.start_hand()

    # Auto-play AI actions until human's turn
    while not state["is_terminal"] and not state["is_human_turn"]:
        current = state["current_player"]
        if current in session.ai_agents:
            ai_action = session.ai_agents[current].get_action(session.current_obs, session.env)
        else:
            import random
            mask = state["action_mask"]
            valid = [i for i, v in enumerate(mask) if v]
            ai_action = random.choice(valid)

        # Track AI action for personality analysis
        session._track_action(current, ai_action)
        session.current_obs, rewards, done, _, info = session.env.step(ai_action)
        state = session._get_state()

        # If hand ended during AI actions
        if state["is_terminal"]:
            session._finalize_hand(rewards)

    return state


@app.post("/api/game/{session_id}/action")
async def do_action(session_id: str, action: ActionRequest):
    """Process a player action."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    try:
        state = session.process_action(action.action, action.amount)
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/game/{session_id}/state")
async def get_state(session_id: str):
    """Get current game state."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions[session_id]._get_state()


@app.get("/api/game/{session_id}/stats")
async def get_player_stats(session_id: str):
    """Get detailed personality statistics for all players."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if not session.opponent_model:
        return {"error": "Personality tracking not available"}

    stats = {}
    for seat in range(session.config.num_players):
        player_id = session.player_ids.get(seat, f"Player_{seat}")
        player_stats = session.opponent_model.get_player_stats(player_id)
        if player_stats:
            stats[seat] = player_stats.to_dict()

    return {"session_id": session_id, "player_stats": stats}


@app.get("/api/game/{session_id}/exploit/{seat}")
async def get_exploit_advice(session_id: str, seat: int):
    """Get exploitative play recommendations for a specific opponent."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if not session.opponent_model:
        return {"error": "Personality tracking not available"}

    player_id = session.player_ids.get(seat, f"Player_{seat}")
    recommendations = session.opponent_model.get_exploit_recommendations(player_id)

    return {
        "session_id": session_id,
        "seat": seat,
        "player_id": player_id,
        "recommendations": recommendations
    }


def main():
    """Run the web server."""
    import uvicorn

    print(f"\n{'='*60}")
    print("  Poker AI - Web Server")
    print(f"{'='*60}")
    print(f"\n  Open http://localhost:8000 in your browser")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
