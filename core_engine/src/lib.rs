//! Ultra-fast No-Limit Hold'em poker engine with Python bindings.
//!
//! This crate provides a high-performance poker engine suitable for
//! training AI agents through reinforcement learning.

pub mod cards;
pub mod evaluator;
pub mod game;
pub mod rng;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use cards::{Card, Rank, Suit};
use evaluator::{init_evaluator, Evaluator, HandCategory};
use game::{Action, GameConfig, GameState, PlayerStatus, Street, MAX_PLAYERS};
use rng::FastRng;

/// Python-friendly card representation
#[pyclass]
#[derive(Clone)]
pub struct PyCard {
    #[pyo3(get)]
    rank: u8,
    #[pyo3(get)]
    suit: u8,
}

#[pymethods]
impl PyCard {
    #[new]
    fn new(rank: u8, suit: u8) -> PyResult<Self> {
        if rank > 12 || suit > 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid card: rank must be 0-12, suit must be 0-3",
            ));
        }
        Ok(Self { rank, suit })
    }

    fn __repr__(&self) -> String {
        let rank_chars = [
            '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
        ];
        let suit_chars = ['c', 'd', 'h', 's'];
        format!(
            "{}{}",
            rank_chars[self.rank as usize], suit_chars[self.suit as usize]
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl From<Card> for PyCard {
    fn from(card: Card) -> Self {
        PyCard {
            rank: card.rank() as u8,
            suit: card.suit().to_index(),
        }
    }
}

impl From<PyCard> for Card {
    fn from(py_card: PyCard) -> Self {
        Card::new(Rank::from_index(py_card.rank), Suit::from_index(py_card.suit))
    }
}

/// Main poker environment for Python
#[pyclass]
pub struct PokerEnv {
    state: GameState,
    num_players: usize,
}

#[pymethods]
impl PokerEnv {
    /// Create a new poker environment
    #[new]
    #[pyo3(signature = (num_players=6, small_blind=50, big_blind=100, starting_stack=10000, seed=None))]
    fn new(
        num_players: usize,
        small_blind: u32,
        big_blind: u32,
        starting_stack: u32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if num_players < 2 || num_players > 9 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_players must be between 2 and 9",
            ));
        }

        init_evaluator();

        let config = GameConfig {
            num_players,
            small_blind,
            big_blind,
            starting_stack,
        };

        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
        });

        Ok(Self {
            state: GameState::new(config, seed),
            num_players,
        })
    }

    /// Reset the environment and start a new hand
    fn reset(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        self.state.start_hand();
        self.get_observation(py)
    }

    /// Take an action in the environment
    ///
    /// Args:
    ///     action: Either an int (0-10 for pot-geometric sizing) or a tuple (action_type, amount)
    ///
    ///     Pot-Geometric Actions (0-10):
    ///     - 0: Fold
    ///     - 1: Check/Call
    ///     - 2: Min Raise
    ///     - 3: 33% Pot
    ///     - 4: 50% Pot
    ///     - 5: 75% Pot
    ///     - 6: 100% Pot (Pot-size)
    ///     - 7: 125% Pot
    ///     - 8: 150% Pot
    ///     - 9: 200% Pot
    ///     - 10: All-in
    ///
    /// Returns:
    ///     Tuple of (observation, rewards, done, truncated, info)
    fn step(&mut self, py: Python<'_>, action: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Handle integer action (pot-geometric sizing)
        if let Ok(action_id) = action.extract::<i32>() {
            if action_id < 0 || action_id > 10 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Action must be 0-10 for pot-geometric sizing",
                ));
            }

            let result = self.state.step_with_action_id(action_id);

            if !result.valid {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid action: {} (use get_action_mask() to check valid actions)",
                    action_id
                )));
            }

            let obs = self.get_observation(py)?;
            let rewards: Vec<i32> = result.rewards[..self.num_players].to_vec();
            let done = result.done;
            let truncated = false;

            let info = PyDict::new(py);
            if done {
                info.set_item("terminal", true)?;
                info.set_item("hand_number", self.state.hand_number)?;
            }

            return Ok((obs, rewards, done, truncated, info.into_py(py)).into_py(py));
        }

        // Handle tuple action for backwards compatibility
        let action = self.parse_action(action)?;
        let result = self.state.step(action);

        if !result.valid {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid action"));
        }

        let obs = self.get_observation(py)?;
        let rewards: Vec<i32> = result.rewards[..self.num_players].to_vec();
        let done = result.done;
        let truncated = false;

        let info = PyDict::new(py);
        if done {
            info.set_item("terminal", true)?;
            info.set_item("hand_number", self.state.hand_number)?;
        }

        Ok((obs, rewards, done, truncated, info.into_py(py)).into_py(py))
    }

    /// Get the current observation
    fn get_observation(&self, py: Python<'_>) -> PyResult<PyObject> {
        let obs = PyDict::new(py);

        // Current player
        obs.set_item("current_player", self.state.current_player)?;

        // Street (0-4)
        obs.set_item("street", self.state.street.index())?;

        // Board cards
        let board: Vec<String> = (0..self.state.board_count as usize)
            .map(|i| format!("{}", self.state.board[i]))
            .collect();
        obs.set_item("board", board)?;

        // Pot
        obs.set_item("pot", self.state.total_pot())?;

        // Current bet
        obs.set_item("current_bet", self.state.current_bet)?;

        // Min raise
        obs.set_item("min_raise", self.state.min_raise)?;

        // Players info
        let players_info: Vec<PyObject> = (0..self.num_players)
            .map(|i| {
                let p = &self.state.players[i];
                let info = PyDict::new(py);
                info.set_item("stack", p.stack).unwrap();
                info.set_item("bet", p.bet_this_street).unwrap();
                info.set_item(
                    "status",
                    match p.status {
                        PlayerStatus::Active => 0,
                        PlayerStatus::Folded => 1,
                        PlayerStatus::AllIn => 2,
                        PlayerStatus::Out => 3,
                    },
                )
                .unwrap();
                info.into_py(py)
            })
            .collect();
        obs.set_item("players", players_info)?;

        // Action mask for 11 pot-geometric actions
        let action_mask: Vec<bool> = self.state.get_action_mask_11().to_vec();
        obs.set_item("action_mask", action_mask)?;

        // Also provide legacy 4-action mask for backwards compatibility
        let valid = self.state.valid_actions();
        let legacy_action_mask: Vec<bool> = vec![
            valid.iter().any(|a| matches!(a, Action::Fold)),
            valid
                .iter()
                .any(|a| matches!(a, Action::Check | Action::Call)),
            valid
                .iter()
                .any(|a| matches!(a, Action::Bet(_) | Action::Raise(_))),
            valid.iter().any(|a| matches!(a, Action::AllIn)),
        ];
        obs.set_item("legacy_action_mask", legacy_action_mask)?;

        // Button position
        obs.set_item("button", self.state.button)?;

        // Is terminal
        obs.set_item("is_terminal", self.state.is_terminal)?;

        Ok(obs.into_py(py))
    }

    /// Get hole cards for a specific player
    fn get_hole_cards(&self, player: usize) -> PyResult<Vec<PyCard>> {
        if player >= self.num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid player index",
            ));
        }

        let cards = &self.state.players[player].hole_cards;
        Ok(vec![PyCard::from(cards[0]), PyCard::from(cards[1])])
    }

    /// Evaluate a player's hand (requires 5 board cards)
    fn evaluate_hand(&self, player: usize) -> PyResult<(u16, String)> {
        if player >= self.num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid player index",
            ));
        }

        if self.state.board_count < 5 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need 5 board cards for evaluation",
            ));
        }

        let cards = [
            self.state.players[player].hole_cards[0],
            self.state.players[player].hole_cards[1],
            self.state.board[0],
            self.state.board[1],
            self.state.board[2],
            self.state.board[3],
            self.state.board[4],
        ];

        let rank = Evaluator::evaluate_7(&cards);
        let category = rank.category();

        Ok((rank.value(), category.name().to_string()))
    }

    /// Get list of valid action indices (0-10 for pot-geometric sizing)
    fn get_valid_action_indices(&self) -> Vec<usize> {
        let mask = self.state.get_action_mask_11();
        mask.iter()
            .enumerate()
            .filter_map(|(i, &valid)| if valid { Some(i) } else { None })
            .collect()
    }

    /// Get legacy valid action indices (0-3 for fold/call/raise/allin)
    fn get_legacy_valid_action_indices(&self) -> Vec<usize> {
        let valid = self.state.valid_actions();
        let mut indices = Vec::new();

        if valid.iter().any(|a| matches!(a, Action::Fold)) {
            indices.push(0);
        }
        if valid
            .iter()
            .any(|a| matches!(a, Action::Check | Action::Call))
        {
            indices.push(1);
        }
        if valid
            .iter()
            .any(|a| matches!(a, Action::Bet(_) | Action::Raise(_)))
        {
            indices.push(2);
        }
        if valid.iter().any(|a| matches!(a, Action::AllIn)) {
            indices.push(3);
        }

        indices
    }

    /// Get the number of actions (11 for pot-geometric sizing)
    #[getter]
    fn num_actions(&self) -> usize {
        11
    }

    /// Set player stacks (for testing)
    fn set_stacks(&mut self, stacks: Vec<u32>) -> PyResult<()> {
        if stacks.len() != self.num_players {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Stacks length must match num_players",
            ));
        }

        for (i, &stack) in stacks.iter().enumerate() {
            self.state.players[i].stack = stack;
            if stack == 0 {
                self.state.players[i].status = PlayerStatus::Out;
            }
        }

        Ok(())
    }

    // Properties

    #[getter]
    fn num_players(&self) -> usize {
        self.num_players
    }

    #[getter]
    fn current_player(&self) -> usize {
        self.state.current_player
    }

    #[getter]
    fn is_terminal(&self) -> bool {
        self.state.is_terminal
    }

    #[getter]
    fn hand_number(&self) -> u64 {
        self.state.hand_number
    }

    #[getter]
    fn street(&self) -> usize {
        self.state.street.index()
    }

    #[getter]
    fn pot(&self) -> u32 {
        self.state.total_pot()
    }

    #[getter]
    fn button(&self) -> usize {
        self.state.button
    }
}

impl PokerEnv {
    fn parse_action(&self, action: &Bound<'_, PyAny>) -> PyResult<Action> {
        // Integer action: 0=fold, 1=check/call, 2=min raise, 3=all-in
        if let Ok(idx) = action.extract::<i32>() {
            return match idx {
                0 => Ok(Action::Fold),
                1 => {
                    let player = &self.state.players[self.state.current_player];
                    if player.bet_this_street >= self.state.current_bet {
                        Ok(Action::Check)
                    } else {
                        Ok(Action::Call)
                    }
                }
                2 => {
                    if self.state.current_bet == 0 {
                        Ok(Action::Bet(self.state.config.big_blind))
                    } else {
                        Ok(Action::Raise(self.state.current_bet + self.state.min_raise))
                    }
                }
                3 => Ok(Action::AllIn),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid action index (0-3)",
                )),
            };
        }

        // Tuple action: (action_type, amount)
        if let Ok((action_type, amount)) = action.extract::<(i32, u32)>() {
            return match action_type {
                0 => Ok(Action::Fold),
                1 => Ok(Action::Check),
                2 => Ok(Action::Call),
                3 => Ok(Action::Bet(amount)),
                4 => Ok(Action::Raise(amount)),
                5 => Ok(Action::AllIn),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid action type",
                )),
            };
        }

        Err(pyo3::exceptions::PyValueError::new_err(
            "Action must be int (0-3) or tuple (action_type, amount)",
        ))
    }
}

/// Benchmark function: run random hands and return hands/second
#[pyfunction]
fn benchmark_random_hands(num_hands: u64, num_players: usize, seed: u64) -> PyResult<f64> {
    use std::time::Instant;

    if num_players < 2 || num_players > 9 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_players must be between 2 and 9",
        ));
    }

    init_evaluator();

    let config = GameConfig {
        num_players,
        small_blind: 50,
        big_blind: 100,
        starting_stack: 10000,
    };

    let mut state = GameState::new(config, seed);
    let mut rng = FastRng::new(seed);

    let start = Instant::now();

    for _ in 0..num_hands {
        state.start_hand();

        while !state.is_terminal {
            let valid = state.valid_actions();
            let idx = rng.gen_range(0, valid.len() as u32) as usize;
            state.step(valid[idx]);
        }

        // Reset stacks for next hand
        for i in 0..num_players {
            state.players[i].stack = 10000;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let hands_per_sec = num_hands as f64 / elapsed;

    Ok(hands_per_sec)
}

/// Evaluate a hand given hole cards and board cards
#[pyfunction]
fn evaluate_hand(hole_cards: Vec<PyCard>, board: Vec<PyCard>) -> PyResult<(u16, String)> {
    if hole_cards.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must have exactly 2 hole cards",
        ));
    }
    if board.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must have exactly 5 board cards",
        ));
    }

    init_evaluator();

    let cards: [Card; 7] = [
        Card::from(hole_cards[0].clone()),
        Card::from(hole_cards[1].clone()),
        Card::from(board[0].clone()),
        Card::from(board[1].clone()),
        Card::from(board[2].clone()),
        Card::from(board[3].clone()),
        Card::from(board[4].clone()),
    ];

    let rank = Evaluator::evaluate_7(&cards);
    let category = rank.category();

    Ok((rank.value(), category.name().to_string()))
}

/// Create a card from rank (0-12) and suit (0-3)
#[pyfunction]
fn make_card(rank: u8, suit: u8) -> PyResult<PyCard> {
    PyCard::new(rank, suit)
}

/// Parse a card from string (e.g., "As", "Kh", "2c")
#[pyfunction]
fn parse_card(s: &str) -> PyResult<PyCard> {
    let card = Card::from_str(s)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Invalid card string"))?;
    Ok(PyCard::from(card))
}

/// Python module definition
#[pymodule]
fn poker_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize evaluator
    init_evaluator();

    // Add classes
    m.add_class::<PyCard>()?;
    m.add_class::<PokerEnv>()?;

    // Add functions
    m.add_function(wrap_pyfunction!(benchmark_random_hands, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_hand, m)?)?;
    m.add_function(wrap_pyfunction!(make_card, m)?)?;
    m.add_function(wrap_pyfunction!(parse_card, m)?)?;

    // Add constants
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    m.add("NUM_ACTIONS", 11)?;  // Pot-geometric action space size

    // Add action descriptions
    m.add(
        "ACTION_NAMES",
        vec![
            "Fold",
            "Check/Call",
            "Min Raise",
            "33% Pot",
            "50% Pot",
            "75% Pot",
            "100% Pot",
            "125% Pot",
            "150% Pot",
            "200% Pot",
            "All-in",
        ],
    )?;

    Ok(())
}
