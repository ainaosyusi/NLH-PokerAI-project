"""Phase 4 integration tests - UI and Human vs AI."""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_engine


def test_terminal_ui():
    """Test terminal UI components."""
    print("=== Terminal UI Test ===")

    from ui.terminal_ui import (
        CardDisplay,
        TableDisplay,
        GameMessages,
        Color,
    )

    # Test card display
    card_str = CardDisplay.format_card(12, 3)  # Ace of spades
    assert "A" in card_str, "Card should contain rank"
    print(f"  Card display: {card_str}")

    # Test hand display
    hand_str = CardDisplay.format_hand([(12, 3), (12, 2)])  # AA
    print(f"  Hand display: {hand_str}")

    # Test board display
    board_str = CardDisplay.format_board([(10, 0), (5, 1), (2, 2)])
    print(f"  Board display: {board_str}")

    # Test messages
    msg = GameMessages.action_taken("Player 1", "raise", 500)
    print(f"  Action message: {msg}")

    msg = GameMessages.winner("Hero", 1000, "pair of Aces")
    print(f"  Winner message: {msg[:50]}...")

    print("✓ Terminal UI test passed\n")
    return True


def test_game_controller():
    """Test game controller with simulated actions."""
    print("=== Game Controller Test ===")

    from ui.game_controller import GameController

    # Create controller
    game = GameController(
        num_players=6,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        human_seat=0,
        seed=42,
    )

    print(f"  Players: {game.num_players}")
    print(f"  AI opponents: {list(game.ai_names.values())}")

    # Simulate a few hands with auto-fold
    for hand_num in range(3):
        game.hand_number += 1
        obs = game.env.reset()

        actions_taken = 0
        while not obs["is_terminal"]:
            current = obs["current_player"]

            if current == game.human_seat:
                # Auto-fold for testing
                action = 0
            else:
                # AI action
                action = game._get_ai_action(current, obs)

            obs, rewards, done, _, _ = game.env.step(action)
            actions_taken += 1

        print(f"  Hand {hand_num+1}: {actions_taken} actions, profit: {rewards[0]:+.0f}")

    print("✓ Game controller test passed\n")
    return True


def test_hand_history():
    """Test hand history save/load."""
    print("=== Hand History Test ===")

    from ui.game_controller import GameController
    from ui.hand_history import HandHistoryViewer

    # Create game and play some hands
    game = GameController(
        num_players=2,
        starting_stack=10000,
        human_seat=0,
        seed=123,
    )

    for _ in range(5):
        game.hand_number += 1
        obs = game.env.reset()

        game.current_hand_actions = []

        while not obs["is_terminal"]:
            current = obs["current_player"]

            if current == game.human_seat:
                action = 1  # Call/check
            else:
                action = game._get_ai_action(current, obs)

            game.current_hand_actions.append({
                "player": current,
                "action": action,
                "street": obs.get("street", 0),
            })

            obs, rewards, done, _, _ = game.env.step(action)

        # Record hand
        from ui.game_controller import HandRecord
        from datetime import datetime

        record = HandRecord(
            hand_number=game.hand_number,
            timestamp=datetime.now().isoformat(),
            player_cards=[(12, 0), (11, 1)],  # AK
            board=[(10, 0), (5, 1), (2, 2)],
            actions=game.current_hand_actions,
            pot=200,
            winner=0 if rewards[0] > 0 else 1,
            profit=rewards[0],
            street_reached="River",
        )
        game.hand_history.append(record)
        game.stats.hands_played += 1
        game.stats.total_profit += rewards[0]

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        game.save_history(temp_path)

    print(f"  Saved {len(game.hand_history)} hands to {temp_path}")

    # Load and verify
    viewer = HandHistoryViewer(temp_path)
    print(f"  Loaded {len(viewer.hands)} hands")
    print(f"  Total profit: {viewer.total_profit:+.0f}")

    # Cleanup
    os.unlink(temp_path)

    print("✓ Hand history test passed\n")
    return True


def test_session_stats():
    """Test session statistics tracking."""
    print("=== Session Stats Test ===")

    from ui.game_controller import SessionStats

    stats = SessionStats()
    stats.hands_played = 100
    stats.hands_won = 55
    stats.total_profit = 2500
    stats.vpip_hands = 30
    stats.pfr_hands = 25
    stats.showdowns = 40
    stats.showdown_wins = 22

    print(f"  Win rate: {stats.win_rate*100:.1f}%")
    print(f"  VPIP: {stats.vpip*100:.1f}%")
    print(f"  PFR: {stats.pfr*100:.1f}%")
    print(f"  WTSD: {stats.wtsd*100:.1f}%")
    print(f"  W$SD: {stats.won_at_showdown*100:.1f}%")

    assert 0 <= stats.win_rate <= 1
    assert 0 <= stats.vpip <= 1
    assert 0 <= stats.pfr <= 1

    print("✓ Session stats test passed\n")
    return True


def test_web_api():
    """Test web API endpoints (without actually starting server)."""
    print("=== Web API Test ===")

    from ui.web_server import GameSession, GameConfig

    config = GameConfig(
        num_players=6,
        starting_stack=10000,
        small_blind=50,
        big_blind=100,
        human_seat=0,
    )

    session = GameSession("test_session", config)
    print(f"  Created session: {session.session_id}")
    print(f"  AI opponents: {list(session.ai_names.values())}")

    # Start hand
    state = session.start_hand()
    print(f"  Started hand #{state['hand_number']}")
    print(f"  Pot: ${state['pot']}")
    print(f"  Is human turn: {state['is_human_turn']}")

    # Process actions until human's turn
    actions_processed = 0
    while not state["is_terminal"] and not state["is_human_turn"]:
        current = state["current_player"]
        if current in session.ai_agents:
            ai_action = session.ai_agents[current].get_action(session.current_obs, session.env)
        else:
            import random
            mask = state["action_mask"]
            valid = [i for i, v in enumerate(mask) if v]
            ai_action = random.choice(valid)

        session.current_obs, _, _, _, _ = session.env.step(ai_action)
        state = session._get_state()
        actions_processed += 1

    print(f"  AI actions processed: {actions_processed}")
    print(f"  Current player: {state['current_player']}")

    print("✓ Web API test passed\n")
    return True


def test_ai_personalities():
    """Test AI personality behaviors."""
    print("=== AI Personality Test ===")

    from brain.personality_agents import (
        RuleBasedAgent,
        PersonalityType,
        PERSONALITY_CONFIGS,
    )

    # Create agents with different personalities
    agents = {
        "TAG": RuleBasedAgent(PersonalityType.TAG, player_id=0, seed=1),
        "LAG": RuleBasedAgent(PersonalityType.LAG, player_id=1, seed=2),
        "ROCK": RuleBasedAgent(PersonalityType.ROCK, player_id=2, seed=3),
        "MANIAC": RuleBasedAgent(PersonalityType.MANIAC, player_id=3, seed=4),
    }

    print("  Personality VPIP ranges:")
    for name, agent in agents.items():
        config = PERSONALITY_CONFIGS[agent.personality]
        print(f"    {name}: {config.vpip_range[0]*100:.0f}%-{config.vpip_range[1]*100:.0f}%")

    # Test action generation
    env = poker_engine.PokerEnv(num_players=6)
    obs = env.reset()

    print("\n  Action test (preflop):")
    for name, agent in agents.items():
        actions = []
        for _ in range(10):
            obs = env.reset()
            action = agent.get_action(obs, env)
            actions.append(action)
        fold_pct = actions.count(0) / len(actions) * 100
        print(f"    {name}: {fold_pct:.0f}% fold rate (10 hands)")

    print("✓ AI personality test passed\n")
    return True


def main():
    """Run all Phase 4 tests."""
    print("="*60)
    print("PHASE 4 INTEGRATION TESTS - UI & Human vs AI")
    print("="*60 + "\n")

    tests = [
        ("Terminal UI", test_terminal_ui),
        ("Game Controller", test_game_controller),
        ("Hand History", test_hand_history),
        ("Session Stats", test_session_stats),
        ("Web API", test_web_api),
        ("AI Personalities", test_ai_personalities),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} returned False")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n✓ All Phase 4 tests passed!")
        print("\nTo play poker against AI:")
        print("  Terminal: python play_poker.py")
        print("  Web:      python ui/web_server.py (then open http://localhost:8000)")
    else:
        print(f"\n✗ {failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
