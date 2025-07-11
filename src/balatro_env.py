
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
        elif self.joker_type == JokerType.JOLLY_JOKER:
            if hand_type == HandType.PAIR:
                mult += 8
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
        if len(cards) < 1 or len(cards) > 5:
            return HandType.HIGH_CARD, 0, 0

        ranks = [card.rank.value for card in cards]
        suits = [card.suit for card in cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1 and len(cards) >= 5
        is_straight = PokerEvaluator._is_straight(ranks) and len(cards) >= 5
        
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

        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] != sorted_ranks[i-1] + 1:
                break
        else:
            return True
        
        if 14 in sorted_ranks:
            temp_ranks = [r for r in sorted_ranks if r != 14] + [1]
            temp_ranks.sort()
            if temp_ranks == [1, 2, 3, 4, 5]:
                return True
        
        return False

class BalatroEnv(gym.Env):
    def __init__(self, max_ante: int = 8, starting_money: int = 4, hand_size: int = 8, max_jokers: int = 5):
        super().__init__()
        
        self.max_ante = max_ante
        self.starting_money = starting_money
        self.hand_size = hand_size
        self.max_jokers = max_jokers
        
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

        self.action_space = spaces.Discrete(2**self.hand_size + 2**self.hand_size)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(200,), dtype=np.float32
        )
        
        self.blinds = self._initialize_blinds()
        
    def _create_deck(self):
        self.deck = []
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(suit, rank))
        random.shuffle(self.deck)
    
    def _initialize_blinds(self) -> Dict[int, List[Blind]]:
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
        super().reset(seed=seed)
        self.current_ante = 1
        self.current_blind = 0
        self.money = self.starting_money
        self.hands_left = 4
        self.discards_left = 3
        self.jokers = [Joker(JokerType.JOKER)]
        self.score = 0
        self.played_cards_this_round = []
        
        self._create_deck()
        self._deal_hand()
        self._set_blind()
        
        info = {
            'ante_reached': self.current_ante,
            'blinds_beaten': self._get_total_blinds_beaten()
        }
        return self._get_observation(), info
    
    def _deal_hand(self):
        self.hand = []
        for _ in range(self.hand_size):
            if self.deck:
                self.hand.append(self.deck.pop())
    
    def _set_blind(self):
        if self.current_ante > self.max_ante:
            self.chips_needed = 0
            return

        current_blind = self.blinds[self.current_ante][self.current_blind]
        self.chips_needed = current_blind.chips_required
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]: # Added truncated to return
        reward = 0
        done = False
        truncated = False
        info = {
            'ante_reached': self.current_ante,
            'blinds_beaten': self._get_total_blinds_beaten(),
            'won': False,
            'failed': False,
            'action_type': 'invalid',
            'hand_score': 0
        }
        
        action_type = "play" if action < (2**self.hand_size) else "discard"
        card_selection_mask = action % (2**self.hand_size)
        
        selected_cards_indices = [i for i in range(min(self.hand_size, len(self.hand))) if (card_selection_mask >> i) & 1]
        selected_cards = [self.hand[i] for i in selected_cards_indices]

        if action_type == "play":
            if not (1 <= len(selected_cards) <= 5):
                reward = -0.1
                info['action_type'] = 'invalid_play'
            else:
                if self.hands_left == 0:
                    reward = -0.5
                    info['action_type'] = 'invalid_play_no_hands'
                else:
                    self.hands_left -= 1
                    
                    hand_type, chips, mult = PokerEvaluator.evaluate_hand(selected_cards)
                    
                    total_bonus_chips = 0
                    total_bonus_mult = 0
                    for joker in self.jokers:
                        bonus_chips, bonus_mult = joker.get_effect(selected_cards, hand_type)
                        total_bonus_chips += bonus_chips
                        total_bonus_mult += bonus_mult
                    
                    final_chips = chips + total_bonus_chips
                    final_mult = mult + total_bonus_mult
                    hand_score = final_chips * final_mult
                    
                    self.score += hand_score
                    info['hand_score'] = hand_score
                    info['action_type'] = 'play'
                    info['hand_type'] = hand_type.value

                    for card in selected_cards:
                        if card in self.hand:
                            self.hand.remove(card)
                        
                    while len(self.hand) < self.hand_size and self.deck:
                        self.hand.append(self.deck.pop())

        elif action_type == "discard":
            if not (1 <= len(selected_cards) <= self.hand_size):
                reward = -0.1
                info['action_type'] = 'invalid_discard_selection'
            elif self.discards_left == 0:
                reward = -0.5
                info['action_type'] = 'invalid_discard_no_discards'
            else:
                self.discards_left -= 1
                info['action_type'] = 'discard'
                
                for card in selected_cards:
                    if card in self.hand:
                        self.hand.remove(card)
                
                while len(self.hand) < self.hand_size and self.deck:
                    self.hand.append(self.deck.pop())
        else:
            reward = -1.0
            info['action_type'] = 'undefined'

        if self.score >= self.chips_needed:
            reward += 1.0
            self.money += self.blinds[self.current_ante][self.current_blind].reward
            self._advance_blind()
            info['blind_beaten'] = True
            info['ante_reached'] = self.current_ante
            info['blinds_beaten'] = self._get_total_blinds_beaten()
            
            if self.current_ante > self.max_ante:
                done = True
                reward += 10.0
                info['won'] = True
        elif self.hands_left == 0 and action_type == "play":
            reward -= 1.0
            done = True
            info['failed'] = True
        else:
            if self.chips_needed > 0:
                reward += (self.score / self.chips_needed) * 0.05
            
        if self.hands_left == 0 and self.discards_left == 0 and self.score < self.chips_needed:
            done = True
            reward -= 2.0
            info['failed'] = True

        return self._get_observation(), reward, done, truncated, info
    
    def _get_total_blinds_beaten(self) -> int:
        return (self.current_ante - 1) * 3 + self.current_blind

    def _advance_blind(self):
        self.current_blind += 1
        if self.current_blind >= 3:
            self.current_blind = 0
            self.current_ante += 1
        
        self.hands_left = 4
        self.discards_left = 3
        self.score = 0
        self.played_cards_this_round = []
        self._create_deck()
        self._deal_hand()
        self._set_blind()
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        offset = 0
        
        for i, card in enumerate(self.hand[:self.hand_size]):
            rank_idx = card.rank.value - 2
            suit_idx = list(Suit).index(card.suit)
            obs[offset + i * 18 + rank_idx] = 1
            obs[offset + i * 18 + 13 + suit_idx] = 1
            obs[offset + i * 18 + 17] = 1 if card.enhanced else 0
        offset += self.hand_size * 18

        obs[offset] = self.current_ante / self.max_ante
        obs[offset+1] = self.current_blind / 3.0
        obs[offset+2] = self.hands_left / 4.0
        obs[offset+3] = self.discards_left / 3.0
        obs[offset+4] = self.score / (self.chips_needed if self.chips_needed > 0 else 1.0)
        obs[offset+5] = self.money / 50.0
        obs[offset+6] = len(self.deck) / 52.0
        offset += 7

        for i, joker in enumerate(self.jokers[:self.max_jokers]):
            joker_type_idx = list(JokerType).index(joker.joker_type)
            obs[offset + i * (len(JokerType) + 1) + joker_type_idx] = 1
            obs[offset + i * (len(JokerType) + 1) + len(JokerType)] = joker.level / 5.0
        offset += self.max_jokers * (len(JokerType) + 1)
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        print(f"\n--- Balatro Game State ---")
        print(f"Ante: {self.current_ante}, Blind: {self.current_blind+1}/3")
        print(f"Score: {self.score}/{self.chips_needed} (Needed: {self.chips_needed - self.score} more)")
        print(f"Money: ${self.money}")
        print(f"Hands left: {self.hands_left}, Discards left: {self.discards_left}")
        print(f"Deck size: {len(self.deck)}")
        print(f"Hand ({len(self.hand)} cards): {self.hand}")
        print(f"Jokers ({len(self.jokers)}): {self.jokers}")
        print("--------------------------")

    def close(self):
        pass
