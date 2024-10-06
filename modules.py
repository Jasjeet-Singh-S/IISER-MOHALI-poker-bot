# useful libraries
import random
import torch
import torch.nn as nn
from enum import Enum 


# useful classes
class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single value
        )
    
    def forward(self, x):
        return self.fc(x)

    
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, action_space):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)
    

class Player:
    def __init__(self, id, stack):
        self.id = id
        self.stack = stack
        self.hand = []
        self.current_bet = 0
        self.folded = False
        

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3
    BET = 4


class TexasHoldEm:
    def __init__(self, players, small_blind=10, big_blind=20):
        self.players = players  # List of Player objects
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = self.initialize_deck()
        self.board = []  # Community cards
        self.pot = 0
        self.current_bet = 0
        self.game_over = False
        self.dealer_index = 0  # Index of the dealer in self.players
        self.current_player_index = (self.dealer_index + 1) % len(self.players)  # Player to act
        self.betting_round = 'pre-flop'  # Can be 'pre-flop', 'flop', 'turn', 'river'
        self.round_bets = {}  # Tracks bets per player in the current round
        self.last_raiser = None  # Tracks the last player who raised

    def initialize_deck(self):
        suits = ['H', 'D', 'C', 'S']
        ranks = range(2, 15)  # 2-14 where 11-14 are J, Q, K, A
        deck = [(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(deck)
        return deck

    # Modify the deal_hole_cards method
    def deal_hole_cards(self, player_hands=None):
        for player in self.players:
            if player_hands and player.id in player_hands:
                player.hand = player_hands[player.id]
            else:
                player.hand = [self.deck.pop(), self.deck.pop()]
            player.current_bet = 0
            player.folded = False
        self.round_bets = {player.id: 0 for player in self.players}


    def post_blinds(self):
        small_blind_player = self.players[(self.dealer_index + 1) % len(self.players)]
        big_blind_player = self.players[(self.dealer_index + 2) % len(self.players)]
        
        self._post_blind(small_blind_player, self.small_blind)
        self._post_blind(big_blind_player, self.big_blind)
        
        self.current_bet = self.big_blind
        self.last_raiser = big_blind_player.id

        # Set current player to the one after the big blind
        self.current_player_index = (self.dealer_index + 3) % len(self.players)

    def _post_blind(self, player, amount):
        player.stack -= amount
        player.current_bet = amount
        self.pot += amount
        self.round_bets[player.id] = amount

    def deal_flop(self):
        self.deck.pop()  # Burn card
        self.board.extend([self.deck.pop() for _ in range(3)])
        self.betting_round = 'flop'
        self.reset_bets()

    def deal_turn(self):
        self.deck.pop()  # Burn card
        self.board.append(self.deck.pop())
        self.betting_round = 'turn'
        self.reset_bets()

    def deal_river(self):
        self.deck.pop()  # Burn card
        self.board.append(self.deck.pop())
        self.betting_round = 'river'
        self.reset_bets()

    def reset_bets(self):
        self.current_bet = 0
        for player in self.players:
            player.current_bet = 0
        self.round_bets = {player.id: 0 for player in self.players}
        self.current_player_index = self.dealer_index  # Start with the player after the dealer
        self.last_raiser = None

    def get_current_player(self):
        while True:
            player = self.players[self.current_player_index]
            if not player.folded:
                return player
            self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def get_available_actions(self, player):
        if player.current_bet < self.current_bet:
            # Player needs to call or fold
            actions = [Action.FOLD, Action.CALL, Action.RAISE]
        else:
            # Player can check or bet/raise
            if self.current_bet == 0:
                actions = [Action.CHECK, Action.BET]
            else:
                actions = [Action.CHECK, Action.RAISE]
        return actions

    def execute_action(self, player, action, raise_amount=0):
        if action == Action.FOLD:
            self.handle_fold(player)
        elif action == Action.CHECK:
            self.handle_check(player)
        elif action == Action.CALL:
            self.handle_call(player)
        elif action == Action.BET:
            self.handle_bet(player, raise_amount)
        elif action == Action.RAISE:
            self.handle_raise(player, raise_amount)
        else:
            raise ValueError("Invalid action")

        # Move to the next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def handle_fold(self, player):
        player.folded = True
        print(f"Player {player.id} folds.")

    def handle_check(self, player):
        print(f"Player {player.id} checks.")

    def handle_call(self, player):
        call_amount = self.current_bet - player.current_bet
        player.stack -= call_amount
        player.current_bet += call_amount
        self.pot += call_amount
        self.round_bets[player.id] += call_amount
        print(f"Player {player.id} calls {call_amount}.")

    def handle_bet(self, player, amount):
        if amount <= 0 or amount > player.stack:
            raise ValueError("Invalid bet amount")
        player.stack -= amount
        player.current_bet += amount
        self.current_bet = player.current_bet
        self.pot += amount
        self.round_bets[player.id] += amount
        self.last_raiser = player.id
        print(f"Player {player.id} bets {amount}.")

    def handle_raise(self, player, amount):
        if amount <= 0 or amount > player.stack:
            raise ValueError("Invalid raise amount")
        call_amount = self.current_bet - player.current_bet
        total_amount = call_amount + amount
        player.stack -= total_amount
        player.current_bet += total_amount
        self.current_bet = player.current_bet
        self.pot += total_amount
        self.round_bets[player.id] += total_amount
        self.last_raiser = player.id
        print(f"Player {player.id} raises by {amount} to {player.current_bet}.")

    def is_round_over(self):
        # The betting round is over when all players have either called the current bet or folded
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            return True  # Only one player remains
        for player in active_players:
            if player.id == self.last_raiser:
                continue  # Skip the last raiser
            if player.current_bet != self.current_bet:
                return False
        return True

    def progress_round(self):
        if self.betting_round == 'pre-flop':
            self.deal_flop()
        elif self.betting_round == 'flop':
            self.deal_turn()
        elif self.betting_round == 'turn':
            self.deal_river()
        elif self.betting_round == 'river':
            self.game_over = True  # Proceed to showdown
        else:
            raise ValueError("Invalid betting round")

    def is_game_over(self):
        # The game is over if only one player remains or all betting rounds are complete
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            return True
        return self.game_over
    
    def suit_to_char(self, suit):
        suit_mapping = {'H': 'h', 'D': 'd', 'C': 'c', 'S': 's'}
        return suit_mapping[suit]


    def determine_winner(self):
        # If only one player remains
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            print(f"Player {winner.id} wins the pot of {self.pot} by default.")
            self.pot = 0
            return

        # Showdown: compare hands
        from treys import Evaluator, Card
        evaluator = Evaluator()
        best_rank = None
        winners = []
        for player in active_players:
            hand = [Card.new(f"{self.rank_to_str(rank)}{self.suit_to_char(suit)}") for rank, suit in player.hand]
            board = [Card.new(f"{self.rank_to_str(rank)}{self.suit_to_char(suit)}") for rank, suit in self.board]
            rank = evaluator.evaluate(board, hand)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                winners = [player]
            elif rank == best_rank:
                winners.append(player)
        # Split the pot among winners
        split_pot = self.pot / len(winners)
        for winner in winners:
            winner.stack += split_pot
            print(f"Player {winner.id} wins {split_pot} from the pot.")
        self.pot = 0


    def rank_to_str(self, rank):
        if rank == 14:
            return 'A'
        elif rank == 13:
            return 'K'
        elif rank == 12:
            return 'Q'
        elif rank == 11:
            return 'J'
        elif rank == 10:
            return 'T'
        else:
            return str(rank)

    def get_reward(self, player):
        # Define reward as the change in the player's stack
        return player.stack - 1000  # Assuming initial stack is 1000

    # Additional helper methods can be added as needed
    
class BeliefState:
       def __init__(self, observed_actions, public_cards, pot_size=0):
           self.observed_actions = observed_actions
           self.public_cards = public_cards
           self.private_cards = None
           self.pot_size = pot_size

       def update(self, action, new_public_cards=None, pot_size=None):
           self.observed_actions.append(action)
           if new_public_cards is not None:
               self.public_cards = new_public_cards
           if pot_size is not None:
               self.pot_size = pot_size
            

MAX_FEATURE_LENGTH = 25  # Adjust this value based on your game dynamics            
def extract_features(belief_state):
    features = []
    
    # Encode observed actions
    action_encoding = {
        Action.FOLD: 0,
        Action.CHECK: 1,
        Action.CALL: 2,
        Action.RAISE: 3,
        Action.BET: 4
    }
    max_history_length = 12
    action_features = [action_encoding[action] for action in belief_state.observed_actions[-max_history_length:]]
    action_features += [0] * (max_history_length - len(action_features))
    features.extend(action_features)
    
    # Encode public cards
    rank_encoding = {r: i for i, r in enumerate(range(2, 15), start=1)}
    suit_encoding = {'H': 0, 'D': 1, 'C': 2, 'S': 3}
    max_board_cards = 5
    board_features = []
    for rank, suit in belief_state.public_cards:
        rank_feature = rank_encoding.get(rank, 0)
        suit_feature = suit_encoding.get(suit, 0)
        board_features.extend([rank_feature, suit_feature])
    while len(board_features) < max_board_cards * 2:
        board_features.extend([0, 0])
    features.extend(board_features)
    
    # Encode private cards (player's hand)
    hand_features = []
    if belief_state.private_cards is not None:
        for rank, suit in belief_state.private_cards:
            rank_feature = rank_encoding.get(rank, 0)
            suit_feature = suit_encoding.get(suit, 0)
            hand_features.extend([rank_feature, suit_feature])
    else:
        hand_features.extend([0, 0, 0, 0])  # Assuming 2 hole cards
    features.extend(hand_features)
    
    # Encode pot size
    features.append(belief_state.pot_size / 1000)  # Normalize pot size
    
    # Ensure total feature length matches MAX_FEATURE_LENGTH
    features = features[:MAX_FEATURE_LENGTH] + [0] * max(0, MAX_FEATURE_LENGTH - len(features))
    
    # Convert to tensor
    features = torch.tensor(features, dtype=torch.float32)
    return features


# Define sample_action function
def sample_action(action_probs, valid_actions):
    """
    Samples an action from the given action probabilities, considering only valid actions.

    Args:
        action_probs (torch.Tensor): A tensor containing the probabilities for each action.
        valid_actions (list of Action): List of valid actions in the current state.

    Returns:
        Action: The selected action.
    """
    action_list = list(Action)
    action_to_index = {action: idx for idx, action in enumerate(action_list)}
    valid_action_indices = [action_to_index[action] for action in valid_actions]

    # Get probabilities of valid actions
    valid_action_probs = action_probs[valid_action_indices]
    valid_action_probs /= valid_action_probs.sum()  # Normalize

    # Sample from valid actions
    chosen_index = torch.multinomial(valid_action_probs, num_samples=1).item()
    selected_action = valid_actions[chosen_index]

    return selected_action


def determine_raise_amount(player, game):
    """
    Determines the amount to raise or bet.

    Args:
        player (Player): The player who is raising.
        game (TexasHoldEm): The current game state.

    Returns:
        float: The raise amount.
    """
    # Simple strategy: raise by a fixed amount or percentage of the pot
    # For this example, we'll raise by half the pot or the player's remaining stack, whichever is smaller
    raise_amount = min(player.stack, max(game.pot * 0.5, game.big_blind))
    return raise_amount