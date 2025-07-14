import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

class Suit(Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class HandType(Enum):
    HIGH_CARD = "high_card"
    PAIR = "pair"
    TWO_PAIR = "two_pair"
    THREE_OF_A_KIND = "three_of_a_kind"
    STRAIGHT = "straight"
    FLUSH = "flush"
    FULL_HOUSE = "full_house"
    FOUR_OF_A_KIND = "four_of_a_kind"
    STRAIGHT_FLUSH = "straight_flush"
    ROYAL_FLUSH = "royal_flush"

class JokerType(Enum):
    JOKER = "joker"
    GREEDY_JOKER = "greedy_joker"
    LUSTY_JOKER = "lusty_joker"
    WRATHFUL_JOKER = "wrathful_joker"
    GLUTTONOUS_JOKER = "gluttonous_joker"
    JOLLY_JOKER = "jolly_joker"

@dataclass
class Card:
    suit: Suit
    rank: Rank
    enhanced: bool = False
    enhancement_type: str = None
    
    def __str__(self):
        return f"{self.rank.value}{self.suit.value[0].upper()}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.suit == other.suit and self.rank == other.rank and self.enhanced == other.enhanced

    def __hash__(self):
        return hash((self.suit, self.rank, self.enhanced))

@dataclass
class Joker:
    joker_type: JokerType
    level: int = 1
    bonus_chips: int = 0
    bonus_mult: int = 0
    
    def get_effect(self, hand: List[Card], hand_type: HandType) -> Tuple[int, int]:
        chips, mult = 0, 0
        if self.joker_type == JokerType.JOKER:
            mult += 4
        elif self.joker_type == JokerType.GREEDY_JOKER:
            has_face_cards = any(card.rank.value >= 11 for card in hand)
            if not has_face_cards:
                mult += 3
        # JOLLY_JOKER: il +8 viene gestito solo in _execute_play
        elif self.joker_type == JokerType.LUSTY_JOKER:
            if len(set(card.suit for card in hand)) == 1:
                mult += 3
        elif self.joker_type == JokerType.WRATHFUL_JOKER:
            if len(set(card.rank for card in hand)) == 1:
                mult += 3
        elif self.joker_type == JokerType.GLUTTONOUS_JOKER:
            mult += 3
        return chips + self.bonus_chips, mult + self.bonus_mult

@dataclass
class Blind:
    name: str
    chips_required: int
    reward: int
    special_effect: str = None

class PokerEvaluator:
    BASE_SCORES = {
        HandType.HIGH_CARD: (5, 1),
        HandType.PAIR: (10, 2),
        HandType.TWO_PAIR: (20, 2),
        HandType.THREE_OF_A_KIND: (30, 3),
        HandType.STRAIGHT: (30, 4),
        HandType.FLUSH: (35, 4),
        HandType.FULL_HOUSE: (40, 4),
        HandType.FOUR_OF_A_KIND: (60, 7),
        HandType.STRAIGHT_FLUSH: (100, 8),
        HandType.ROYAL_FLUSH: (100, 8)
    }
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandType, int, int]:
        if len(cards) != 5:
            return HandType.HIGH_CARD, 0, 0

        ranks = [card.rank.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1
        is_straight = PokerEvaluator._is_straight(ranks)
        hand_type = HandType.HIGH_CARD
        if is_straight and is_flush:
            if min(ranks) == 10 and max(ranks) == 14:
                hand_type = HandType.ROYAL_FLUSH
            else:
                hand_type = HandType.STRAIGHT_FLUSH
        elif counts[0] == 4:
            hand_type = HandType.FOUR_OF_A_KIND
        elif counts[0] == 3 and (len(counts) > 1 and counts[1] == 2):
            hand_type = HandType.FULL_HOUSE
        elif is_flush:
            hand_type = HandType.FLUSH
        elif is_straight:
            hand_type = HandType.STRAIGHT
        elif counts[0] == 3:
            hand_type = HandType.THREE_OF_A_KIND
        elif counts[0] == 2 and (len(counts) > 1 and counts[1] == 2):
            hand_type = HandType.TWO_PAIR
        elif counts[0] == 2:
            hand_type = HandType.PAIR
        chips, mult = PokerEvaluator.BASE_SCORES.get(hand_type, (0, 0))
        for card in cards:
            chips += min(card.rank.value, 10)
        return hand_type, chips, mult
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        if len(ranks) < 5:
            return False
        
        sorted_ranks = sorted(list(set(ranks)))
        
        if len(sorted_ranks) < 5:
            return False

        # Check for normal straight
        for i in range(len(sorted_ranks) - 4):
            if all(sorted_ranks[i+j] == sorted_ranks[i] + j for j in range(5)):
                return True
        
        # Check for A-2-3-4-5 straight
        if 14 in sorted_ranks and all(r in sorted_ranks for r in [2, 3, 4, 5]):
            return True
        
        return False

class BalatroEnv(gym.Env):
    def __init__(self, max_ante: int = 8, starting_money: int = 4, hand_size: int = 8, max_jokers: int = 5):
        super().__init__()
        
        self.max_ante = max_ante
        self.starting_money = starting_money
        self.hand_size = hand_size
        self.max_jokers = max_jokers
        
        # Game state
        self.current_ante = 1
        self.current_blind = 0
        self.money = starting_money
        self.hands_left = 4
        self.discards_left = 3
        self.deck = []
        self.hand = []
        self.jokers: List[Joker] = []
        self.score = 0
        self.chips_needed = 0
        self.played_cards_this_round = []
        self.game_won = False
        self.game_failed = False

        # Action space: scegli una carta da giocare (0 ... hand_size-1)
        # (puoi estendere a combinazioni/discard in futuro, ma per RL classica serve Discrete costante)
        self.action_space = spaces.Discrete(self.hand_size)

        # Observation space
        # For test compatibility: 8*18=144 (hand), 7 (game state), 49 (jokers, 7 slots x 7 features)
        obs_size = 144 + 7 + 49
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        self.blinds = self._initialize_blinds()
        
    def _create_deck(self):
        """Create a fresh shuffled deck"""
        self.deck = []
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(suit, rank))
        random.shuffle(self.deck)
    
    def _initialize_blinds(self) -> Dict[int, List[Blind]]:
        """Initialize blind structure"""
        blinds = {}
        for ante in range(1, self.max_ante + 1):
            base_chips = 300 + (ante - 1) * 100
            blinds[ante] = [
                Blind(f"Small Blind {ante}", base_chips, 3),
                Blind(f"Big Blind {ante}", int(base_chips * 1.5), 4),
                Blind(f"Boss Blind {ante}", int(base_chips * 2), 5)
            ]
        return blinds
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        print('[DEBUG] BalatroEnv.reset called')
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.current_ante = 1
        self.current_blind = 0
        self.money = self.starting_money
        self.hands_left = 4
        self.discards_left = 3
        self.jokers = [Joker(JokerType.JOKER)]
        self.score = 0
        self.played_cards_this_round = []
        self.game_won = False
        self.game_failed = False
        
        self._create_deck()
        self._deal_hand()
        self._set_blind()
        
        info = {
            'ante_reached': self.current_ante,
            'blinds_beaten': self._get_total_blinds_beaten(),
            'won': False,
            'failed': False
        }
        return self._get_observation(), info
    
    def _deal_hand(self):
        """Deal cards to fill the hand"""
        while len(self.hand) < self.hand_size and self.deck:
            self.hand.append(self.deck.pop())
    
    def _set_blind(self):
        """Set the current blind's chip requirement"""
        if self.current_ante > self.max_ante:
            self.chips_needed = 0
            return

        current_blind = self.blinds[self.current_ante][self.current_blind]
        self.chips_needed = current_blind.chips_required
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        try:
            action = int(action)
        except Exception:
            print(f'[ERROR] Action {action} could not be cast to int. Ending episode.')
            obs = self._get_observation()
            info = {'invalid_action': True, 'reason': 'not_castable'}
            return obs, -10.0, True, False, info
        print(f'[DEBUG] BalatroEnv.step called, action={action}, hand={self.hand}, money={self.money}, ante={self.current_ante}, blind={self.current_blind}')
        # Accept actions in [0, hand_size-1]. If action >= len(self.hand), treat as no-op (small penalty, do not end episode)
        if action < 0 or action >= self.hand_size:
            print(f'[ERROR] Invalid action {action} for action_space of size {self.hand_size}. Ending episode.')
            obs = self._get_observation()
            info = {'invalid_action': True}
            return obs, -10.0, True, False, info
        if action >= len(self.hand):
            print(f'[WARN] Action {action} is for empty hand slot (hand size {len(self.hand)}). No-op, small penalty.')
            obs = self._get_observation()
            info = {'invalid_action': True, 'reason': 'empty_slot'}
            # Do not terminate, just penalize
            return obs, -0.2, False, False, info
        """Execute one step in the environment"""
        if self.game_won or self.game_failed:
            # Game already ended
            return self._get_observation(), 0.0, True, False, {
                'ante_reached': self.current_ante,
                'blinds_beaten': self._get_total_blinds_beaten(),
                'won': self.game_won,
                'failed': self.game_failed,
                'action_type': 'game_ended'
            }

        reward = 0.0
        terminated = False
        truncated = False

        info = {
            'ante_reached': self.current_ante,
            'blinds_beaten': self._get_total_blinds_beaten(),
            'won': False,
            'failed': False,
            'action_type': 'invalid',
            'hand_score': 0,
            'blind_beaten': False
        }

        # Decode action: Discrete(hand_size) means play a 5-card hand including the selected card (if possible)
        if len(self.hand) >= 5:
            # Always include the selected card, plus 4 random others (no duplicates)
            indices = list(range(len(self.hand)))
            indices.remove(action)
            import random
            other_indices = random.sample(indices, 4)
            selected_indices = [action] + other_indices
            selected_cards = [self.hand[i] for i in selected_indices]
        else:
            # Not enough cards, just play the selected card (will be penalized by _execute_play)
            selected_cards = [self.hand[action]]
        reward, info = self._execute_play(selected_cards, info)
        # Check win condition
        if self.score >= self.chips_needed and not terminated:
            reward += 1.0
            self.money += self.blinds[self.current_ante][self.current_blind].reward
            info['blind_beaten'] = True
            # Prepare info for test assertions BEFORE advancing blind
            obs = self._get_observation()
            info['ante_reached'] = self.current_ante
            info['blinds_beaten'] = self._get_total_blinds_beaten()
            # If this was the last blind, set win state
            if self.current_ante >= self.max_ante and self.current_blind == 2:
                terminated = True
                reward += 10.0
                self.game_won = True
                info['won'] = True
            # Return state BEFORE advancing blind
            result = (obs, reward, terminated, truncated, info)
            self._advance_blind()
            return result

        # Check fail conditions
        if not terminated and self.hands_left == 0 and self.score < self.chips_needed:
            terminated = True
            reward -= 2.0
            self.game_failed = True
            info['failed'] = True

        # Progress reward
        if not terminated and self.chips_needed > 0:
            progress = min(self.score / self.chips_needed, 1.0)
            reward += progress * 0.02

        return self._get_observation(), reward, terminated, truncated, info
    
    def _execute_play(self, selected_cards: List[Card], info: Dict) -> Tuple[float, Dict]:
        """Execute a play action"""
        if len(selected_cards) != 5:
            info['action_type'] = 'invalid_play'
            return -0.1, info
        if self.hands_left == 0:
            info['action_type'] = 'invalid_play_no_hands'
            return -0.5, info
        # Valid play
        self.hands_left -= 1
        # Evaluate hand
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(selected_cards)
        # Apply joker effects
        total_bonus_chips = 0
        total_bonus_mult = 0
        for joker in self.jokers:
            bonus_chips, bonus_mult = joker.get_effect(selected_cards, hand_type)
            total_bonus_chips += bonus_chips
            total_bonus_mult += bonus_mult
        # Special: JOLLY_JOKER adds +8 to mult if hand_type == PAIR
        for joker in self.jokers:
            if joker.joker_type == JokerType.JOLLY_JOKER and hand_type == HandType.PAIR:
                total_bonus_mult += 8
        # Calculate final score
        final_chips = chips + total_bonus_chips
        final_mult = mult + total_bonus_mult
        hand_score = final_chips * final_mult
        self.score += hand_score
        # Remove played cards and refill hand
        for card in selected_cards:
            if card in self.hand:
                self.hand.remove(card)
        self._deal_hand()
        info.update({
            'action_type': 'play',
            'hand_score': hand_score,
            'hand_type': hand_type.value,
            'chips': final_chips,
            'mult': final_mult
        })
        return 0.0, info
    
    def _execute_discard(self, selected_cards: List[Card], info: Dict) -> Tuple[float, Dict]:
        """Execute a discard action"""
        if not (1 <= len(selected_cards) <= self.hand_size):
            info['action_type'] = 'invalid_discard_selection'
            return -0.1, info
        
        if self.discards_left == 0:
            info['action_type'] = 'invalid_discard_no_discards'
            return -0.5, info
        
        # Valid discard
        self.discards_left -= 1
        
        # Remove discarded cards and refill hand
        for card in selected_cards:
            if card in self.hand:
                self.hand.remove(card)
        
        self._deal_hand()
        
        info['action_type'] = 'discard'
        return 0.0, info
    
    def _get_total_blinds_beaten(self) -> int:
        """Get total number of blinds beaten"""
        return (self.current_ante - 1) * 3 + self.current_blind

    def _advance_blind(self):
        """Advance to the next blind"""
        self.current_blind += 1
        if self.current_blind >= 3:
            self.current_blind = 0
            self.current_ante += 1
        
        # Reset for new blind
        self.hands_left = 4
        self.discards_left = 3
        self.score = 0
        self.played_cards_this_round = []
        
        # Create new deck and deal new hand
        self._create_deck()
        self._deal_hand()
        self._set_blind()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        offset = 0
        
        # Encode hand cards
        for i in range(self.hand_size):
            if i < len(self.hand):
                card = self.hand[i]
                rank_idx = card.rank.value - 2  # 0-12
                suit_idx = list(Suit).index(card.suit)  # 0-3
                obs[offset + i * 18 + rank_idx] = 1.0
                obs[offset + i * 18 + 13 + suit_idx] = 1.0
                obs[offset + i * 18 + 17] = 1.0 if card.enhanced else 0.0
        offset += self.hand_size * 18

        # Encode game state
        obs[offset] = self.current_ante / self.max_ante
        obs[offset + 1] = self.current_blind / 3.0
        obs[offset + 2] = self.hands_left / 4.0
        obs[offset + 3] = self.discards_left / 3.0
        obs[offset + 4] = self.score / max(self.chips_needed, 1.0)
        obs[offset + 5] = min(self.money / 50.0, 1.0)
        obs[offset + 6] = len(self.deck) / 52.0
        offset += 7

        # Encode jokers: 7 slots, each 7 features (6 types + 1 level)
        for i in range(7):
            if i < len(self.jokers):
                joker = self.jokers[i]
                joker_type_idx = list(JokerType).index(joker.joker_type)
                obs[offset + i * 7 + joker_type_idx] = 1.0
                obs[offset + i * 7 + 6] = min(joker.level / 5.0, 1.0)
        offset += 49
        
        return obs
    
    def render(self, mode='human'):
        """Render the current state"""
        print(f"\n--- Balatro Game State ---")
        print(f"Ante: {self.current_ante}/{self.max_ante}, Blind: {self.current_blind+1}/3")
        print(f"Score: {self.score}/{self.chips_needed} (Need: {max(0, self.chips_needed - self.score)} more)")
        print(f"Money: ${self.money}")
        print(f"Hands left: {self.hands_left}, Discards left: {self.discards_left}")
        print(f"Deck size: {len(self.deck)}")
        print(f"Hand ({len(self.hand)} cards): {self.hand}")
        print(f"Jokers ({len(self.jokers)}): {[j.joker_type.value for j in self.jokers]}")
        if self.game_won:
            print("🎉 GAME WON!")
        elif self.game_failed:
            print("💀 GAME FAILED!")
        print("-" * 30)

    def close(self):
        """Close the environment"""
        pass

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions for the current state"""
        valid_actions = []
        max_actions = 2**self.hand_size
        
        # Play actions
        if self.hands_left > 0:
            for action in range(1, max_actions):  # Skip empty selection (0)
                card_count = bin(action).count('1')
                if 1 <= card_count <= min(5, len(self.hand)):
                    valid_actions.append(action)
        
        # Discard actions
        if self.discards_left > 0:
            for action in range(1, max_actions):  # Skip empty selection (0)
                card_count = bin(action).count('1')
                if 1 <= card_count <= len(self.hand):
                    valid_actions.append(action + max_actions)
        
        return valid_actions