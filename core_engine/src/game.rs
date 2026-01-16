//! No-Limit Hold'em game state machine.
//!
//! Handles game flow, betting rounds, pot calculation, and showdown.

use crate::cards::{Card, Deck};
use crate::evaluator::{Evaluator, HandRank};
use crate::rng::FastRng;

/// Maximum number of players supported
pub const MAX_PLAYERS: usize = 9;

/// Player status in a hand
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    /// Active and can still act
    Active,
    /// Folded this hand
    Folded,
    /// All-in (cannot act but still in hand)
    AllIn,
    /// Out of the game (no chips)
    Out,
}

/// Betting round (street)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
    Showdown,
}

impl Street {
    /// Get the next street
    pub fn next(self) -> Option<Self> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => Some(Street::Showdown),
            Street::Showdown => None,
        }
    }

    /// Get street index (0-4)
    pub fn index(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 1,
            Street::Turn => 2,
            Street::River => 3,
            Street::Showdown => 4,
        }
    }
}

/// Player action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(u32),
    Raise(u32),
    AllIn,
}

/// Individual player state
#[derive(Debug, Clone)]
pub struct Player {
    /// Current stack
    pub stack: u32,
    /// Hole cards
    pub hole_cards: [Card; 2],
    /// Current status
    pub status: PlayerStatus,
    /// Bet amount this street
    pub bet_this_street: u32,
    /// Total bet this hand
    pub total_bet: u32,
}

impl Player {
    /// Create a new player with the given stack
    pub fn new(stack: u32) -> Self {
        Self {
            stack,
            hole_cards: [Card::from_index(0); 2],
            status: if stack > 0 {
                PlayerStatus::Active
            } else {
                PlayerStatus::Out
            },
            bet_this_street: 0,
            total_bet: 0,
        }
    }

    /// Check if player is still in the hand
    #[inline]
    pub fn is_active(&self) -> bool {
        matches!(self.status, PlayerStatus::Active | PlayerStatus::AllIn)
    }

    /// Check if player can take actions
    #[inline]
    pub fn can_act(&self) -> bool {
        self.status == PlayerStatus::Active
    }
}

/// Pot (supports side pots)
#[derive(Debug, Clone)]
pub struct Pot {
    /// Pot amount
    pub amount: u32,
    /// Players eligible for this pot (bitmask)
    pub eligible: u16,
}

/// Game configuration
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub num_players: usize,
    pub small_blind: u32,
    pub big_blind: u32,
    pub starting_stack: u32,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            num_players: 6,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 10000,
        }
    }
}

/// Result of a step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Is the hand complete?
    pub done: bool,
    /// Rewards for each player (profit/loss)
    pub rewards: [i32; MAX_PLAYERS],
    /// Next player to act (if any)
    pub next_player: Option<usize>,
    /// Was the action valid?
    pub valid: bool,
}

impl StepResult {
    fn terminal() -> Self {
        Self {
            done: true,
            rewards: [0; MAX_PLAYERS],
            next_player: None,
            valid: true,
        }
    }

    fn invalid() -> Self {
        Self {
            done: false,
            rewards: [0; MAX_PLAYERS],
            next_player: None,
            valid: false,
        }
    }

    fn continue_hand(next: usize) -> Self {
        Self {
            done: false,
            rewards: [0; MAX_PLAYERS],
            next_player: Some(next),
            valid: true,
        }
    }

    fn hand_complete(rewards: [i32; MAX_PLAYERS]) -> Self {
        Self {
            done: true,
            rewards,
            next_player: None,
            valid: true,
        }
    }
}

/// Complete game state
#[derive(Clone)]
pub struct GameState {
    /// Game configuration
    pub config: GameConfig,
    /// Players
    pub players: [Player; MAX_PLAYERS],
    /// Number of active players
    pub num_players: usize,
    /// Community cards
    pub board: [Card; 5],
    /// Number of board cards dealt
    pub board_count: u8,
    /// Current street
    pub street: Street,
    /// Dealer button position
    pub button: usize,
    /// Current player to act
    pub current_player: usize,
    /// Current bet to call
    pub current_bet: u32,
    /// Minimum raise amount
    pub min_raise: u32,
    /// Pots (main + side pots)
    pub pots: Vec<Pot>,
    /// Number of players who have acted this street
    pub actions_this_street: u8,
    /// Last player to raise
    pub last_aggressor: Option<usize>,
    /// Deck
    deck: Deck,
    /// RNG
    rng: FastRng,
    /// Is hand complete?
    pub is_terminal: bool,
    /// Hand number
    pub hand_number: u64,
}

impl GameState {
    /// Create a new game state
    pub fn new(config: GameConfig, seed: u64) -> Self {
        let mut state = Self {
            config: config.clone(),
            players: std::array::from_fn(|_| Player::new(config.starting_stack)),
            num_players: config.num_players,
            board: [Card::from_index(0); 5],
            board_count: 0,
            street: Street::Preflop,
            button: 0,
            current_player: 0,
            current_bet: 0,
            min_raise: config.big_blind,
            pots: vec![Pot {
                amount: 0,
                eligible: 0,
            }],
            actions_this_street: 0,
            last_aggressor: None,
            deck: Deck::new(),
            rng: FastRng::new(seed),
            is_terminal: true,
            hand_number: 0,
        };

        // Mark excess player slots as Out
        for i in config.num_players..MAX_PLAYERS {
            state.players[i].status = PlayerStatus::Out;
            state.players[i].stack = 0;
        }

        state
    }

    /// Start a new hand
    pub fn start_hand(&mut self) {
        self.hand_number += 1;

        // Reset deck
        self.deck = Deck::new();

        // Reset player states
        for i in 0..self.num_players {
            if self.players[i].stack > 0 {
                self.players[i].status = PlayerStatus::Active;
            } else {
                self.players[i].status = PlayerStatus::Out;
            }
            self.players[i].bet_this_street = 0;
            self.players[i].total_bet = 0;
        }

        // Reset board
        self.board_count = 0;
        self.street = Street::Preflop;
        self.pots = vec![Pot {
            amount: 0,
            eligible: self.active_players_mask(),
        }];
        self.is_terminal = false;

        // Deal hole cards
        for i in 0..self.num_players {
            if self.players[i].status == PlayerStatus::Active {
                self.players[i].hole_cards = [
                    self.deck.draw(&mut self.rng).unwrap(),
                    self.deck.draw(&mut self.rng).unwrap(),
                ];
            }
        }

        // Post blinds
        self.post_blinds();
    }

    /// Post blinds
    fn post_blinds(&mut self) {
        let sb_pos = self.next_active_player(self.button);
        let bb_pos = self.next_active_player(sb_pos);

        self.post_bet(sb_pos, self.config.small_blind);
        self.post_bet(bb_pos, self.config.big_blind);

        self.current_bet = self.config.big_blind;
        self.min_raise = self.config.big_blind;
        self.current_player = self.next_active_player(bb_pos);
        self.last_aggressor = Some(bb_pos);
        self.actions_this_street = 0;
    }

    /// Post a bet
    fn post_bet(&mut self, player: usize, amount: u32) {
        let actual = amount.min(self.players[player].stack);
        self.players[player].stack -= actual;
        self.players[player].bet_this_street += actual;
        self.players[player].total_bet += actual;

        // Add to main pot
        if !self.pots.is_empty() {
            self.pots[0].amount += actual;
        }

        if self.players[player].stack == 0 {
            self.players[player].status = PlayerStatus::AllIn;
        }
    }

    /// Execute an action
    pub fn step(&mut self, action: Action) -> StepResult {
        if self.is_terminal {
            return StepResult::terminal();
        }

        // Validate action
        if !self.is_valid_action(action) {
            return StepResult::invalid();
        }

        let player = self.current_player;

        // Apply action
        match action {
            Action::Fold => {
                self.players[player].status = PlayerStatus::Folded;
            }
            Action::Check => {
                // Nothing to do
            }
            Action::Call => {
                let call_amount = self.current_bet - self.players[player].bet_this_street;
                self.post_bet(player, call_amount);
            }
            Action::Bet(amount) => {
                self.post_bet(player, amount);
                self.current_bet = amount;
                self.min_raise = amount;
                self.last_aggressor = Some(player);
                self.actions_this_street = 0;
            }
            Action::Raise(total) => {
                let to_call = self.current_bet - self.players[player].bet_this_street;
                let raise_to = total;
                let raise_by = raise_to - self.current_bet;

                self.post_bet(player, to_call + raise_by);
                self.current_bet = raise_to;
                self.min_raise = self.min_raise.max(raise_by);
                self.last_aggressor = Some(player);
                self.actions_this_street = 0;
            }
            Action::AllIn => {
                let all_in_total =
                    self.players[player].stack + self.players[player].bet_this_street;
                let stack = self.players[player].stack;
                self.post_bet(player, stack);

                if all_in_total > self.current_bet {
                    let raise_by = all_in_total - self.current_bet;
                    if raise_by >= self.min_raise {
                        self.last_aggressor = Some(player);
                        self.actions_this_street = 0;
                    }
                    self.current_bet = all_in_total;
                    self.min_raise = self.min_raise.max(raise_by);
                }
            }
        }

        self.actions_this_street += 1;

        // Advance game
        self.advance_game()
    }

    /// Advance the game after an action
    fn advance_game(&mut self) -> StepResult {
        // Check if only one player remains
        let active_count = self.count_active_players();
        if active_count <= 1 {
            return self.end_hand();
        }

        // Find next player to act
        if let Some(next) = self.find_next_to_act() {
            self.current_player = next;
            StepResult::continue_hand(next)
        } else {
            // Street complete
            self.advance_street()
        }
    }

    /// Advance to the next street
    fn advance_street(&mut self) -> StepResult {
        // Calculate side pots
        self.calculate_side_pots();

        // Reset betting
        for i in 0..self.num_players {
            self.players[i].bet_this_street = 0;
        }
        self.current_bet = 0;
        self.min_raise = self.config.big_blind;
        self.actions_this_street = 0;
        self.last_aggressor = None;

        // Move to next street
        if let Some(next_street) = self.street.next() {
            self.street = next_street;

            // Deal community cards
            match next_street {
                Street::Flop => {
                    for _ in 0..3 {
                        self.board[self.board_count as usize] =
                            self.deck.draw(&mut self.rng).unwrap();
                        self.board_count += 1;
                    }
                }
                Street::Turn | Street::River => {
                    self.board[self.board_count as usize] = self.deck.draw(&mut self.rng).unwrap();
                    self.board_count += 1;
                }
                Street::Showdown => {
                    return self.end_hand();
                }
                _ => {}
            }

            // Check if anyone can act
            if self.count_can_act() == 0 {
                // All-in runout
                return self.advance_street();
            }

            // First to act is left of button
            self.current_player = self.next_can_act(self.button);
            StepResult::continue_hand(self.current_player)
        } else {
            self.end_hand()
        }
    }

    /// End the hand and calculate rewards
    fn end_hand(&mut self) -> StepResult {
        self.is_terminal = true;

        // Calculate side pots
        self.calculate_side_pots();

        // Calculate rewards
        let rewards = self.calculate_rewards();

        // Apply rewards to stacks
        for (i, &reward) in rewards.iter().enumerate() {
            if i < self.num_players {
                self.players[i].stack = (self.players[i].stack as i32 + reward) as u32;
            }
        }

        // Move button
        self.button = self.next_seated_player(self.button);

        StepResult::hand_complete(rewards)
    }

    /// Calculate rewards (profit/loss for each player)
    fn calculate_rewards(&self) -> [i32; MAX_PLAYERS] {
        let mut rewards = [0i32; MAX_PLAYERS];

        // Subtract initial bets
        for i in 0..self.num_players {
            rewards[i] = -(self.players[i].total_bet as i32);
        }

        // Get active players mask
        let active_mask = self.active_players_mask();

        // If only one player remains, they win everything
        if active_mask.count_ones() == 1 {
            let winner = active_mask.trailing_zeros() as usize;
            let total_pot: u32 = self.pots.iter().map(|p| p.amount).sum();
            rewards[winner] += total_pot as i32;
            return rewards;
        }

        // Showdown: evaluate hands
        let mut hand_ranks = [HandRank::new(u16::MAX); MAX_PLAYERS];
        for i in 0..self.num_players {
            if self.players[i].is_active() {
                let cards = self.make_7_cards(i);
                hand_ranks[i] = Evaluator::evaluate_7(&cards);
            }
        }

        // Award each pot to winner(s)
        for pot in &self.pots {
            let eligible_mask = pot.eligible & active_mask;
            if eligible_mask == 0 {
                continue;
            }

            // Find best hand among eligible players
            let mut best_rank = HandRank::new(u16::MAX);
            let mut winners = 0u16;

            for i in 0..self.num_players {
                if eligible_mask & (1 << i) != 0 {
                    if hand_ranks[i] < best_rank {
                        best_rank = hand_ranks[i];
                        winners = 1 << i;
                    } else if hand_ranks[i] == best_rank {
                        winners |= 1 << i;
                    }
                }
            }

            // Split pot among winners
            let winner_count = winners.count_ones() as u32;
            if winner_count > 0 {
                let share = pot.amount / winner_count;
                let remainder = pot.amount % winner_count;

                let mut first = true;
                for i in 0..self.num_players {
                    if winners & (1 << i) != 0 {
                        rewards[i] += share as i32;
                        if first && remainder > 0 {
                            rewards[i] += remainder as i32;
                            first = false;
                        }
                    }
                }
            }
        }

        rewards
    }

    /// Make a 7-card array for evaluation
    fn make_7_cards(&self, player: usize) -> [Card; 7] {
        [
            self.players[player].hole_cards[0],
            self.players[player].hole_cards[1],
            self.board[0],
            self.board[1],
            self.board[2],
            self.board[3],
            self.board[4],
        ]
    }

    /// Calculate side pots
    fn calculate_side_pots(&mut self) {
        // Collect (player, total_bet) for players who bet something
        let mut contributions: Vec<(usize, u32)> = (0..self.num_players)
            .filter(|&i| self.players[i].total_bet > 0)
            .map(|i| (i, self.players[i].total_bet))
            .collect();

        if contributions.is_empty() {
            self.pots = vec![];
            return;
        }

        // Sort by bet amount
        contributions.sort_by_key(|&(_, bet)| bet);

        // Build side pots
        self.pots.clear();
        let mut prev_bet = 0u32;

        for (i, &(_, bet)) in contributions.iter().enumerate() {
            if bet > prev_bet {
                let pot_size = (bet - prev_bet) * (contributions.len() - i) as u32;
                let eligible: u16 = contributions[i..]
                    .iter()
                    .map(|&(p, _)| 1u16 << p)
                    .fold(0, |acc, x| acc | x);

                self.pots.push(Pot {
                    amount: pot_size,
                    eligible,
                });
                prev_bet = bet;
            }
        }
    }

    /// Check if an action is valid
    pub fn is_valid_action(&self, action: Action) -> bool {
        let player = &self.players[self.current_player];

        match action {
            Action::Fold => true,
            Action::Check => player.bet_this_street >= self.current_bet,
            Action::Call => self.current_bet > player.bet_this_street,
            Action::Bet(amount) => {
                self.current_bet == 0
                    && amount >= self.config.big_blind
                    && amount <= player.stack + player.bet_this_street
            }
            Action::Raise(total) => {
                let min_raise_to = self.current_bet + self.min_raise;
                total >= min_raise_to && total <= player.stack + player.bet_this_street
            }
            Action::AllIn => player.stack > 0,
        }
    }

    /// Get list of valid actions
    pub fn valid_actions(&self) -> Vec<Action> {
        let mut actions = vec![Action::Fold];
        let player = &self.players[self.current_player];

        if player.bet_this_street >= self.current_bet {
            actions.push(Action::Check);
        }

        if self.current_bet > player.bet_this_street {
            actions.push(Action::Call);
        }

        if self.current_bet == 0 && player.stack >= self.config.big_blind {
            actions.push(Action::Bet(self.config.big_blind));
        }

        let min_raise_to = self.current_bet + self.min_raise;
        if min_raise_to <= player.stack + player.bet_this_street {
            actions.push(Action::Raise(min_raise_to));
        }

        if player.stack > 0 {
            actions.push(Action::AllIn);
        }

        actions
    }

    // Helper functions

    #[inline]
    fn active_players_mask(&self) -> u16 {
        let mut mask = 0u16;
        for i in 0..self.num_players {
            if self.players[i].is_active() {
                mask |= 1 << i;
            }
        }
        mask
    }

    #[inline]
    fn count_active_players(&self) -> usize {
        (0..self.num_players)
            .filter(|&i| self.players[i].is_active())
            .count()
    }

    #[inline]
    fn count_can_act(&self) -> usize {
        (0..self.num_players)
            .filter(|&i| self.players[i].can_act())
            .count()
    }

    #[inline]
    fn next_active_player(&self, from: usize) -> usize {
        let mut pos = (from + 1) % self.num_players;
        while pos != from {
            if self.players[pos].is_active() {
                return pos;
            }
            pos = (pos + 1) % self.num_players;
        }
        from
    }

    #[inline]
    fn next_can_act(&self, from: usize) -> usize {
        let mut pos = (from + 1) % self.num_players;
        let start = pos;
        loop {
            if self.players[pos].can_act() {
                return pos;
            }
            pos = (pos + 1) % self.num_players;
            if pos == start {
                break;
            }
        }
        from
    }

    #[inline]
    fn next_seated_player(&self, from: usize) -> usize {
        let mut pos = (from + 1) % self.num_players;
        while pos != from {
            if self.players[pos].stack > 0 || self.players[pos].status != PlayerStatus::Out {
                return pos;
            }
            pos = (pos + 1) % self.num_players;
        }
        from
    }

    fn find_next_to_act(&self) -> Option<usize> {
        let mut pos = self.current_player;

        for _ in 0..self.num_players {
            pos = (pos + 1) % self.num_players;

            if !self.players[pos].can_act() {
                continue;
            }

            // Player needs to act if:
            // 1. They haven't matched the current bet
            if self.players[pos].bet_this_street < self.current_bet {
                return Some(pos);
            }

            // 2. They are the last aggressor (betting round complete)
            if Some(pos) == self.last_aggressor {
                return None;
            }

            // 3. Not everyone has acted yet
            if self.actions_this_street < self.count_can_act() as u8 {
                return Some(pos);
            }
        }

        None
    }

    /// Get the total pot size
    pub fn total_pot(&self) -> u32 {
        self.pots.iter().map(|p| p.amount).sum()
    }

    /// Calculate bet amount as a fraction of the pot (for pot-geometric sizing)
    /// Returns the total bet amount (what the bet/raise will be TO, not BY)
    pub fn calc_pot_fraction(&self, fraction: f32) -> u32 {
        let pot = self.total_pot();
        let to_call = self.current_bet.saturating_sub(
            self.players[self.current_player].bet_this_street
        );

        // Pot after we call + fraction of that pot
        // This gives us the "raise to" amount
        let pot_after_call = pot + to_call;
        let raise_amount = ((pot_after_call as f32) * fraction) as u32;

        // The total bet we're making (current bet + raise amount)
        let total_bet = self.current_bet + raise_amount;

        // Ensure at least min raise
        let min_raise_to = self.current_bet + self.min_raise;
        total_bet.max(min_raise_to)
    }

    /// Calculate the minimum raise amount (raise TO, not BY)
    pub fn calc_min_raise(&self) -> u32 {
        self.current_bet + self.min_raise
    }

    /// Execute an action using action ID (0-10 pot-geometric sizing)
    ///
    /// Action mapping:
    /// - 0: Fold
    /// - 1: Check/Call
    /// - 2: Min Raise
    /// - 3: 33% Pot
    /// - 4: 50% Pot
    /// - 5: 75% Pot
    /// - 6: 100% Pot (Pot-size)
    /// - 7: 125% Pot
    /// - 8: 150% Pot
    /// - 9: 200% Pot
    /// - 10: All-in
    pub fn step_with_action_id(&mut self, action_id: i32) -> StepResult {
        if self.is_terminal {
            return StepResult::terminal();
        }

        let player = &self.players[self.current_player];
        let player_stack = player.stack;
        let player_bet_this_street = player.bet_this_street;
        let total_available = player_stack + player_bet_this_street;

        let action = match action_id {
            0 => Action::Fold,
            1 => {
                // Check or Call
                if player_bet_this_street >= self.current_bet {
                    Action::Check
                } else {
                    Action::Call
                }
            }
            2 => {
                // Min Raise
                let raise_to = self.calc_min_raise();
                if raise_to >= total_available {
                    Action::AllIn
                } else if self.current_bet == 0 {
                    Action::Bet(raise_to)
                } else {
                    Action::Raise(raise_to)
                }
            }
            3 => self.pot_fraction_action(0.33, total_available),
            4 => self.pot_fraction_action(0.50, total_available),
            5 => self.pot_fraction_action(0.75, total_available),
            6 => self.pot_fraction_action(1.00, total_available),
            7 => self.pot_fraction_action(1.25, total_available),
            8 => self.pot_fraction_action(1.50, total_available),
            9 => self.pot_fraction_action(2.00, total_available),
            10 => Action::AllIn,
            _ => return StepResult::invalid(),
        };

        self.step(action)
    }

    /// Helper to create a pot-fraction bet/raise action
    fn pot_fraction_action(&self, fraction: f32, total_available: u32) -> Action {
        let raise_to = self.calc_pot_fraction(fraction);

        if raise_to >= total_available {
            Action::AllIn
        } else if self.current_bet == 0 {
            // No current bet - this is a bet
            Action::Bet(raise_to)
        } else {
            // There's a current bet - this is a raise
            Action::Raise(raise_to)
        }
    }

    /// Check if an action ID (0-10) is valid
    pub fn is_valid_action_id(&self, action_id: i32) -> bool {
        if self.is_terminal {
            return false;
        }

        let player = &self.players[self.current_player];
        let player_stack = player.stack;
        let player_bet_this_street = player.bet_this_street;
        let total_available = player_stack + player_bet_this_street;

        match action_id {
            0 => true, // Fold is always valid
            1 => true, // Check/Call is always valid (either one will work)
            2 => {
                // Min Raise - valid if we can afford at least min raise
                let min_raise_to = self.calc_min_raise();
                total_available >= min_raise_to || player_stack > 0
            }
            3..=9 => {
                // Pot fraction bets - valid if the resulting bet is:
                // 1. At least min raise
                // 2. Different from all-in (otherwise use all-in action)
                // 3. We have enough chips
                let fraction = match action_id {
                    3 => 0.33,
                    4 => 0.50,
                    5 => 0.75,
                    6 => 1.00,
                    7 => 1.25,
                    8 => 1.50,
                    9 => 2.00,
                    _ => return false,
                };
                let raise_to = self.calc_pot_fraction(fraction);
                let min_raise_to = self.calc_min_raise();

                // Valid if: we can afford it, it's at least min raise, and not all-in
                raise_to >= min_raise_to && raise_to < total_available && player_stack > 0
            }
            10 => player_stack > 0, // All-in valid if we have chips
            _ => false,
        }
    }

    /// Get action mask for 11 actions (pot-geometric sizing)
    pub fn get_action_mask_11(&self) -> [bool; 11] {
        let mut mask = [false; 11];
        for i in 0..11 {
            mask[i] = self.is_valid_action_id(i as i32);
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heads_up_fold() {
        let config = GameConfig {
            num_players: 2,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 1000,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        // In heads-up, SB is button and acts first preflop
        // Player 0 is button/SB, Player 1 is BB
        // After blinds, it's SB's turn to act
        let result = state.step(Action::Fold);

        assert!(result.done);
        // BB wins the pot (SB's 50)
    }

    #[test]
    fn test_call_and_check() {
        let config = GameConfig {
            num_players: 2,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 1000,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        // SB calls
        let result = state.step(Action::Call);
        assert!(!result.done);

        // BB checks
        let result = state.step(Action::Check);
        assert!(!result.done);

        // Should be on flop now
        assert_eq!(state.street, Street::Flop);
        assert_eq!(state.board_count, 3);
    }

    #[test]
    fn test_raise() {
        let config = GameConfig {
            num_players: 2,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 1000,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        // SB raises to 200
        let result = state.step(Action::Raise(200));
        assert!(!result.done);
        assert_eq!(state.current_bet, 200);

        // BB calls
        let result = state.step(Action::Call);
        assert!(!result.done);

        // Should be on flop
        assert_eq!(state.street, Street::Flop);
    }

    #[test]
    fn test_all_in() {
        let config = GameConfig {
            num_players: 2,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 500,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        // SB goes all-in
        let result = state.step(Action::AllIn);
        assert!(!result.done);

        // BB calls (also all-in)
        let result = state.step(Action::Call);

        // Hand should complete (all-in runout)
        assert!(result.done);
    }

    #[test]
    fn test_valid_actions() {
        let config = GameConfig {
            num_players: 2,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 1000,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        let actions = state.valid_actions();

        // SB should be able to: fold, call, raise, all-in
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.iter().any(|a| matches!(a, Action::Raise(_))));
        assert!(actions.contains(&Action::AllIn));
        // Should not be able to check (facing a bet)
        assert!(!actions.contains(&Action::Check));
    }

    #[test]
    fn test_multiway_pot() {
        let config = GameConfig {
            num_players: 3,
            small_blind: 50,
            big_blind: 100,
            starting_stack: 1000,
        };

        let mut state = GameState::new(config, 12345);
        state.start_hand();

        // Button folds, SB calls, BB checks
        state.step(Action::Fold);
        state.step(Action::Call);
        state.step(Action::Check);

        assert_eq!(state.street, Street::Flop);
        assert_eq!(state.count_active_players(), 2);
    }
}
