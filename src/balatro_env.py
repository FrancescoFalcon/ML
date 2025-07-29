import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
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
    JOKER = "joker"                    # Base +5 mult sempre
    CHIP_JOKER = "chip_joker"          # +40 chips sempre
    MULT_JOKER = "mult_joker"          # +8 mult sempre
    BONUS_JOKER = "bonus_joker"        # +20 chips e +4 mult sempre
    WILD_JOKER = "wild_joker"          # +0.75 mult per carta giocata
    LUCKY_JOKER = "lucky_joker"        # 40% chance di bonus extra
    PREMIUM_JOKER = "premium_joker"    # +50 chips, +10 mult, +2 mult per carta giocata - IL MIGLIORE!

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
class ShopItem:
    item_type: str  # "joker", "card", "pack"
    content: Any  # Joker object, Card object, or pack type
    price: int
    description: str = ""

@dataclass
class Joker:
    joker_type: JokerType
    level: int = 1
    bonus_chips: int = 0
    bonus_mult: int = 0
    
    def get_effect(self, hand: List[Card], hand_type: HandType) -> Tuple[int, int]:
        chips, mult = 0, 0
        
        if self.joker_type == JokerType.JOKER:
            # Base joker: +5 mult sempre
            mult += 5
        elif self.joker_type == JokerType.CHIP_JOKER:
            # Chip joker: +40 chips sempre
            chips += 40
        elif self.joker_type == JokerType.MULT_JOKER:
            # Mult joker: +8 mult sempre
            mult += 8
        elif self.joker_type == JokerType.BONUS_JOKER:
            # Bonus joker: +20 chips e +4 mult sempre
            chips += 20
            mult += 4
        elif self.joker_type == JokerType.WILD_JOKER:
            # Wild joker: +0.75 mult per carta giocata
            mult += int(len(hand) * 0.75)
        elif self.joker_type == JokerType.LUCKY_JOKER:
            # Lucky joker: 40% chance di +20 chips e +4 mult
            import random
            if random.random() < 0.4:
                chips += 20
                mult += 4
        elif self.joker_type == JokerType.PREMIUM_JOKER:
            # PREMIUM JOKER: IL MIGLIORE! +50 chips, +10 mult, +2 mult per carta giocata
            chips += 50
            mult += 10
            mult += len(hand) * 2  # +2 mult per ogni carta giocata!
                
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
            # For hands with less than 5 cards, find the best possible combination
            ranks = [card.rank.value for card in cards]
            rank_counts = {}
            for rank in ranks:
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            counts = sorted(rank_counts.values(), reverse=True)
            
            # Analyze what we have and determine best hand type
            hand_type = HandType.HIGH_CARD
            used_cards = []
            
            if len(cards) >= 4 and counts[0] == 4:
                hand_type = HandType.FOUR_OF_A_KIND
                quad_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
                used_cards = [card for card in cards if card.rank.value == quad_rank]
            elif len(cards) >= 3 and counts[0] == 3:
                if len(counts) > 1 and counts[1] == 2:
                    hand_type = HandType.FULL_HOUSE
                    triple_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
                    pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
                    used_cards = [card for card in cards if card.rank.value in [triple_rank, pair_rank]]
                else:
                    hand_type = HandType.THREE_OF_A_KIND
                    triple_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
                    used_cards = [card for card in cards if card.rank.value == triple_rank]
            elif len(cards) >= 4 and counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
                hand_type = HandType.TWO_PAIR
                pair_ranks = [rank for rank, count in rank_counts.items() if count == 2]
                used_cards = [card for card in cards if card.rank.value in pair_ranks]
            elif len(cards) >= 2 and counts[0] == 2:
                hand_type = HandType.PAIR
                pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
                used_cards = [card for card in cards if card.rank.value == pair_rank]
            else:
                # Check for flush (if all same suit)
                if len(cards) >= 5:
                    suits = [card.suit for card in cards]
                    if len(set(suits)) == 1:
                        # Check for straight flush
                        if PokerEvaluator._is_straight(ranks):
                            if min(ranks) == 10 and max(ranks) == 14:
                                hand_type = HandType.ROYAL_FLUSH
                            else:
                                hand_type = HandType.STRAIGHT_FLUSH
                        else:
                            hand_type = HandType.FLUSH
                        used_cards = cards
                    elif PokerEvaluator._is_straight(ranks):
                        hand_type = HandType.STRAIGHT
                        used_cards = cards
                    else:
                        # High card - use best card
                        used_cards = [max(cards, key=lambda c: c.rank.value)]
                else:
                    # High card with fewer than 5 cards
                    used_cards = [max(cards, key=lambda c: c.rank.value)] if cards else []
            
            # Calculate chips
            chips, mult = PokerEvaluator.BASE_SCORES.get(hand_type, (0, 0))
            for card in used_cards:
                # Correct ace handling: Ace = 11 points (not 10)
                if card.rank.value == 14:  # Ace
                    chips += 11
                else:
                    chips += min(card.rank.value, 10)
            return hand_type, chips, mult

        ranks = [card.rank.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)
        is_flush = len(set(suits)) == 1
        is_straight = PokerEvaluator._is_straight(ranks)
        hand_type = HandType.HIGH_CARD
        used_cards = []
        if is_straight and is_flush:
            if min(ranks) == 10 and max(ranks) == 14:
                hand_type = HandType.ROYAL_FLUSH
            else:
                hand_type = HandType.STRAIGHT_FLUSH
            used_cards = cards
        elif counts[0] == 4:
            hand_type = HandType.FOUR_OF_A_KIND
            quad_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            used_cards = [card for card in cards if card.rank.value == quad_rank]
        elif counts[0] == 3 and (len(counts) > 1 and counts[1] == 2):
            hand_type = HandType.FULL_HOUSE
            triple_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            used_cards = [card for card in cards if card.rank.value == triple_rank or card.rank.value == pair_rank]
        elif is_flush:
            hand_type = HandType.FLUSH
            used_cards = cards
        elif is_straight:
            hand_type = HandType.STRAIGHT
            used_cards = cards
        elif counts[0] == 3:
            hand_type = HandType.THREE_OF_A_KIND
            triple_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            used_cards = [card for card in cards if card.rank.value == triple_rank]
        elif counts[0] == 2 and (len(counts) > 1 and counts[1] == 2):
            hand_type = HandType.TWO_PAIR
            pair_ranks = [rank for rank, count in rank_counts.items() if count == 2]
            used_cards = [card for card in cards if card.rank.value in pair_ranks]
        elif counts[0] == 2:
            hand_type = HandType.PAIR
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            used_cards = [card for card in cards if card.rank.value == pair_rank]
        else:
            used_cards = [max(cards, key=lambda c: c.rank.value)] if cards else []
        chips, mult = PokerEvaluator.BASE_SCORES.get(hand_type, (0, 0))
        
        # Special handling for straights to handle A-2-3-4-5 correctly
        if hand_type in [HandType.STRAIGHT, HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            # Check if this is a low straight (A-2-3-4-5)
            ranks_in_hand = [card.rank.value for card in used_cards]
            is_low_straight = (14 in ranks_in_hand and all(r in ranks_in_hand for r in [2, 3, 4, 5]))
            
            for card in used_cards:
                if is_low_straight and card.rank.value == 14:  # Ace in low straight
                    chips += 1  # Ace counts as 1 in low straight
                else:
                    chips += min(card.rank.value, 10)
        else:
            for card in used_cards:
                # Correct ace handling: Ace = 11 points (not 10)
                if card.rank.value == 14:  # Ace
                    chips += 11
                else:
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
    def __init__(self, max_ante: int = 8, starting_money: int = 4, hand_size: int = 8, max_jokers: int = 5, debug: bool = False):
        """Initialize BalatroEnv with bitmask action space and robust info dict."""
        super().__init__()
        self.max_ante = max_ante
        self.starting_money = starting_money
        self.hand_size = hand_size
        self.max_jokers = max_jokers
        self.debug = False  # Force debug OFF for clean training
        # Game state
        self.current_ante = 1
        self.current_blind = 0
        self.money = starting_money
        self.hands_left = 4
        self.discards_left = 3
        self.deck = []
        self.hand = []
        self.jokers: List[Joker] = []
        self.shop_items: List[ShopItem] = []
        self.shop_refreshes_left = 2  # Can refresh shop 2 times per blind
        self.in_shop_phase = False  # Track if we're in shop between blinds
        self.shop_steps = 0  # Track how long we've been in shop
        self.shop_refreshes = 0  # Track refresh count per shop session
        self.max_shop_steps = 5  # RIPRISTINATO: Massimo 5 step nel negozio prima dell'uscita forzata
        self.score = 0
        self.chips_needed = 0
        self.played_cards_this_round = []
        self.game_won = False
        self.game_failed = False
        
        # High score tracking for reward scaling
        self.best_score_this_run = 0
        self.run_high_score = 0  # Track across entire session
        
        # CONTINUOUS TRAINING SYSTEM: Evita terminazione frequente degli episodi
        self.continuous_training = False  # DISABILITA per permettere terminazione episodi nel training
        self.total_runs_completed = 0  # Conta run completate (per statistiche)
        self.continuous_hands_played = 0  # Conta totale mani giocate nella sessione continua
        self.auto_reset_on_fail = True  # Auto-reset quando si perde invece di terminare
        
        # HIGH ANTE TRACKING: Track new ante records with detailed info
        self.highest_ante_reached = 0  # Highest ante ever reached in this session
        self.episode_max_ante = 1  # Highest ante reached in current episode
        self.ante_achievements = {}  # Dict[ante -> achievement_info]
        self.current_run_hands = []  # Track hands played in current run
        self.current_run_jokers = []  # Track joker acquisitions
        
        # ANTI-DISCARD SPAM SYSTEM (NUOVO, PIÙ LEGGERO)
        self.recent_actions = []
        self.max_recent_actions = 10
        self.action_repeat_threshold = 3 # RIDOTTA tolleranza per forzare varietà
        self.consecutive_discards = 0
        self.max_consecutive_discards = 2 # RIDOTTA tolleranza per forzare meno discard
        self.discard_spam_penalty = -10.0 # Penalità AUMENTATA per scoraggiare discard
        
        # Hand type tracking for statistics - CON TRACKING RECENTE
        self.hand_type_stats = {hand_type.value: 0 for hand_type in HandType}
        self.total_hands_played = 0
        
        # RECENT HAND TRACKING: Track last 10k hands for convergence monitoring
        self.recent_hand_types = []  # Store recent hand types
        self.recent_window_size = 10000  # Track last 10k hands
        
        # GRADUATED PRESSURE: Adaptive curriculum system
        self.learning_progress = {
            'antes_completed': defaultdict(int),
            'best_hands_found': defaultdict(int),
            'smart_selections': 0,
            'total_selections': 0,
            'success_rate': 0.0
        }
        
        # SMART DISCARD TRACKING: Track hand state before/after discards for strategic rewards
        self.pre_discard_hand_state = None
        self.discard_attempts = 0
        self.successful_improvements = 0
        
        # 🚀 FORCED PLAY SYSTEM: Force agent to play hands more frequently
        self.steps_since_last_hand = 0
        self.max_steps_without_hand = 20  # RIDOTTO: Force a hand play every 20 steps (era 50)
        self.forced_play_count = 0
        
        # Action space: più efficiente - solo azioni valide
        # Play actions: da 1 a 5 carte (sum of C(8,k) for k=1..5) = 8+28+56+70+56 = 218 azioni
        # Discard actions: da 1 a 8 carte (sum of C(8,k) for k=1..8) = 255 azioni  
        # Totale: 218 + 255 = 473 azioni (invece di 512)
        # Per semplicità manteniamo bitmask ma filtriamo le azioni invalide
        # Extended action space: play + discard + shop actions
        # Shop actions: buy_item_0, buy_item_1, buy_item_2, buy_item_3, refresh_shop, leave_shop
        base_actions = 2**self.hand_size + 2**self.hand_size  # Original play + discard
        shop_actions = 6  # 4 buy slots + refresh + leave
        self.action_space = spaces.Discrete(base_actions + shop_actions)

        # Observation space
        # For test compatibility: 8*18=144 (hand), 7 (game state), 49 (jokers), 20 (shop)
        # For test compatibility, force shape to (220,)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(220,), dtype=np.float32
        )
        
        self.blinds = self._initialize_blinds()
        
        # Anti-loop protection (SOLO sistema recent_actions)
        # self.action_counter rimosso - sostituito da recent_actions sopra
        
        # Stagnation detection
        self.last_score = 0
        self.stagnation_steps = 0
        self.max_stagnation_steps = 200  # MOLTO PIU' AGGRESSIVO per forzare azione
        
        # Lightweight progress tracking
        self.step_count = 0
        self.last_progress_log = 0
        
    def _create_deck(self):
        """Create a fresh shuffled deck"""
        self.deck = []
        for suit in Suit:
            for rank in Rank:
                self.deck.append(Card(suit, rank))
        random.shuffle(self.deck)
    
    def _initialize_blinds(self) -> Dict[int, List[Blind]]:
        """Initialize blind structure with MODERATE but challenging pressure"""
        blinds = {}
        for ante in range(1, self.max_ante + 1):
            if ante == 1:
                small = 300
                big = 450
                boss = 600
            else:
                prev_boss = blinds[ante-1][2].chips_required
                # Il primo blind del nuovo ante deve essere PIÙ difficile del boss precedente
                small = prev_boss + 60   # Incremento ancora più basso
                big = small + 120        # Progressione più lenta
                boss = big + 150         # Boss leggermente più facile

            blinds[ante] = [
                Blind(f"Small Blind {ante}", small, 3),
                Blind(f"Big Blind {ante}", big, 4),
                Blind(f"Boss Blind {ante}", boss, 5)
            ]
        return blinds
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # print('[DEBUG] BalatroEnv.reset called')
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.current_ante = 1
        self.current_blind = 0
        self.money = self.starting_money  # ALWAYS reset to starting money (4)
        self.hands_left = 4
        self.discards_left = 3
        self.jokers = []  # FIXED: Start with 0 jokers like real Balatro
        self.shop_items = []
        self.shop_refreshes_left = 2
        self.in_shop_phase = False
        self.shop_steps = 0
        self.shop_refreshes = 0
        self.score = 0
        
        # Reset high score tracking for this run
        self.best_score_this_run = 0
        
        # Reset anti-loop protection (SOLO recent_actions)
        # self.action_counter rimosso
        self.step_count = 0
        self.last_score = 0
        self.stagnation_steps = 0
        self.played_cards_this_round = []
        self.game_won = False
        self.game_failed = False
        
        # 🚀 RESET FORCED PLAY SYSTEM
        self.steps_since_last_hand = 0
        self.forced_play_count = 0
        
        # Reset hand type tracking
        self.hand_type_stats = {hand_type.value: 0 for hand_type in HandType}
        self.total_hands_played = 0
        self.steps_taken = 0  # Reset step counter per tracking ratio
        
        # Reset contatore azioni nulle consecutive
        self.consecutive_null_actions = 0
        
        # RESET HIGH ANTE TRACKING per nuovo episodio
        self.episode_max_ante = 1  # Reset per nuovo episodio
        self.current_run_hands = []
        self.current_run_jokers = []
        
        # GRADUATED PRESSURE: Adaptive curriculum - don't reset progress to allow learning
        # Only reset episode-specific counters, keep overall learning progress
        
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
        if self.debug:
            print(f"[DEBUG _set_blind] current_ante={self.current_ante}, current_blind={self.current_blind}, max_ante={self.max_ante}")
        
        if self.current_ante > self.max_ante:
            if self.debug:
                print(f"[DEBUG _set_blind] ante > max_ante, setting chips_needed=0")
            self.chips_needed = 0
            return

        # Check if we have blinds for current ante before accessing
        if (self.current_ante in self.blinds and 
            self.current_blind < len(self.blinds[self.current_ante]) and
            self.current_blind >= 0):
            current_blind = self.blinds[self.current_ante][self.current_blind]
            self.chips_needed = current_blind.chips_required
            if self.debug:
                print(f"[DEBUG _set_blind] Valid blind found: chips_needed={self.chips_needed}")
        else:
            # Invalid blind state - recover silently
            if self.debug:
                print(f"[DEBUG _set_blind] Invalid blind state, attempting recovery")
                print(f"[DEBUG _set_blind] ante in blinds: {self.current_ante in self.blinds}")
                if self.current_ante in self.blinds:
                    print(f"[DEBUG _set_blind] blinds[{self.current_ante}] length: {len(self.blinds[self.current_ante])}")
                print(f"[DEBUG _set_blind] current_blind: {self.current_blind}")
            
            # Force reset to valid state
            if self.current_ante <= self.max_ante and self.current_ante >= 1:
                self.current_blind = 0
                if self.current_ante in self.blinds and len(self.blinds[self.current_ante]) > 0:
                    current_blind = self.blinds[self.current_ante][self.current_blind]
                    self.chips_needed = current_blind.chips_required
                    if self.debug:
                        print(f"[DEBUG _set_blind] Recovery successful: chips_needed={self.chips_needed}")
                else:
                    self.chips_needed = 1000  # Default fallback
                    if self.debug:
                        print(f"[DEBUG _set_blind] No blinds for ante, fallback: chips_needed=1000")
            else:
                self.chips_needed = 1000  # Default fallback
                if self.debug:
                    print(f"[DEBUG _set_blind] Invalid ante, fallback: chips_needed=1000")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        try:
            action = int(action)
        except Exception:
            obs = self._get_observation()
            info = {'invalid_action': True, 'reason': 'not_castable'}
            return obs, -10.0, True, False, info

        
        # Increment step counter for ratio tracking
        self.steps_taken += 1
        
        # 🚀 FORCED PLAY SYSTEM: Track steps since last hand played
        self.steps_since_last_hand += 1
        
        # ANTI-SPAM: Track recent actions
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
        
        # Check for action spam (same action repeated too many times)
        if len(self.recent_actions) >= self.action_repeat_threshold:
            recent_count = self.recent_actions[-self.action_repeat_threshold:].count(action)
            if recent_count >= self.action_repeat_threshold:
                if self.continuous_training:
                    # CONTINUOUS TRAINING: Auto-reset invece di terminare
                    self._internal_reset()
                    obs = self._get_observation()
                    info = {'invalid_action': True, 'reason': 'action_spam_reset', 'action_repeated': action}
                    return obs, self.discard_spam_penalty / 2, False, False, info  # Penalità ridotta
                else:
                    # Training normale: termina episodio
                    obs = self._get_observation()
                    info = {'invalid_action': True, 'reason': 'action_spam', 'action_repeated': action}
                    return obs, self.discard_spam_penalty, True, False, info
        
        # 🚀 FORCED PLAY CHECK: If too many steps without playing a hand, force one
        if (not self.in_shop_phase and 
            self.hands_left > 0 and 
            self.steps_since_last_hand >= self.max_steps_without_hand):
            
            # Force play the best possible hand
            if len(self.hand) >= 1:
                # Find best possible 5-card hand or use all available cards
                if len(self.hand) >= 5:
                    best_cards, hand_type, chips, mult = self._find_best_5_card_hand(self.hand)
                else:
                    hand_type, chips, mult = PokerEvaluator.evaluate_hand(self.hand)
                    best_cards = self.hand.copy()
                
                self.forced_play_count += 1
                
                # Execute the forced play
                reward, info = self._execute_play(best_cards, {'forced_play': True})
                
                # Reset step counter and apply small bonus for forced progression
                self.steps_since_last_hand = 0
                reward += 5.0  # Small bonus for progression
                info['forced_play'] = True
                info['forced_play_count'] = self.forced_play_count
                
                # Check win/lose conditions and return
                terminated = self.game_won or self.game_failed
                truncated = False
                
                if self.score >= self.chips_needed and not terminated:
                    reward += 50.0  # Blind beaten bonus
                    if self.current_blind == 2:  # Boss blind
                        ante_completion_bonus = 100.0 + (self.current_ante * 50.0)
                        if self.current_ante >= 2:
                            progression_bonus = 100.0 * (self.current_ante ** 2)
                            ante_completion_bonus += progression_bonus
                        reward += ante_completion_bonus
                    
                    # Enter shop or advance blind
                    if self.current_ante == self.max_ante and self.current_blind == 2:
                        self.game_won = True
                        terminated = True
                        reward += 1000.0
                        info['win'] = True
                    else:
                        self._enter_shop_phase()
                    
                    info['blind_beaten'] = True
                    info['ante_reached'] = self.current_ante
                    info['blinds_beaten'] = self._get_total_blinds_beaten()
                
                return self._get_observation(), reward, terminated, truncated, info
        
        # Pre-filter obviously invalid actions FIRST to avoid wasting computation
        max_play_actions = 2**self.hand_size
        max_discard_actions = 2**self.hand_size
        base_actions = max_play_actions + max_discard_actions
        shop_actions = 6
        
        if action < 0 or action >= base_actions + shop_actions:
            obs = self._get_observation()
            info = {'invalid_action': True, 'reason': 'out_of_bounds', 'lose': True}
            return obs, -10.0, True, False, info
            
        # PRIORITIZE: If in shop, only allow shop actions or force valid ones
        if self.in_shop_phase:
            if action < base_actions:
                # Invalid: trying to play/discard in shop
                obs = self._get_observation()
                info = {'invalid_action': True, 'reason': 'play_discard_in_shop', 'lose': False}
                return obs, -2.0, False, False, info
            elif action >= base_actions + shop_actions:
                # Invalid: shop action out of bounds
                obs = self._get_observation()
                info = {'invalid_action': True, 'reason': 'invalid_shop_action', 'lose': False}
                return obs, -2.0, False, False, info
            # PENALITÀ MASSIVA e forzatura uscita se troppo tempo in shop
            self.shop_steps += 1
            if self.shop_steps > self.max_shop_steps:
                info = {'forced_shop_exit': True, 'reason': 'shop_timeout', 'lose': False}
                self._leave_shop(info)
                reward = -50.0  # PENALITÀ RIDOTTA per forzare apprendimento uscita (era -200)
                obs = self._get_observation()
                return obs, reward, False, False, info
        else:
            # Not in shop: don't allow shop actions
            if action >= base_actions:
                obs = self._get_observation()
                info = {'invalid_action': True, 'reason': 'shop_action_outside_shop', 'lose': False}
                return obs, -2.0, False, False, info
        
        # Pre-filter obviously invalid actions to save computation
        if action < max_play_actions:
            # Play action: quick check for obviously invalid actions
            card_count = bin(action).count('1')
            if card_count == 0 or card_count > 5:
                obs = self._get_observation()
                info = {'invalid_action': True, 'reason': f'impossible_play_action_{card_count}_cards', 'lose': False}
                return obs, -5.0, False, False, info
        elif action < base_actions:
            # Discard action: quick check for obviously invalid actions  
            discard_action = action - max_play_actions
            card_count = bin(discard_action).count('1')
            if card_count == 0 or card_count > len(self.hand):
                obs = self._get_observation()
                info = {'invalid_action': True, 'reason': f'impossible_discard_action_{card_count}_cards', 'lose': False}
                return obs, -5.0, False, False, info
                
            # ANTI-SPAM: Track consecutive discards
            self.consecutive_discards += 1
            if self.consecutive_discards > self.max_consecutive_discards:
                if self.continuous_training:
                    # CONTINUOUS TRAINING: Auto-reset invece di terminare
                    self._internal_reset()
                    obs = self._get_observation()
                    info = {'invalid_action': True, 'reason': 'too_many_consecutive_discards_reset', 'count': self.consecutive_discards}
                    return obs, self.discard_spam_penalty / 2, False, False, info  # Penalità ridotta
                else:
                    # Training normale: termina episodio
                    obs = self._get_observation()
                    info = {'invalid_action': True, 'reason': 'too_many_consecutive_discards', 'count': self.consecutive_discards}
                    return obs, self.discard_spam_penalty, True, False, info
        else:
            # Reset consecutive discard count for non-discard actions
            self.consecutive_discards = 0
        # Shop actions are always valid in terms of format, will be checked in execution
        
        # SOLO SISTEMA ANTI-SPAM NUOVO (recent_actions già implementato sopra)
        # Il sistema action_counter è stato rimosso per evitare conflitti
            obs = self._get_observation()
            
        # Lightweight progress logging (every 10000 steps - molto meno frequente)
        self.step_count += 1
        if self.step_count - self.last_progress_log >= 50000:  # Only log every 50k steps for clean output
            # Decode current action for better debugging
            if action < max_play_actions:
                card_count = bin(action).count('1')
                action_type = f"PLAY {card_count} cards"
            elif action < base_actions:
                discard_bitmask = action - max_play_actions
                card_count = bin(discard_bitmask).count('1')
                action_type = f"DISCARD {card_count} cards"
            else:
                shop_action = action - base_actions
                shop_names = ["BUY_0", "BUY_1", "BUY_2", "BUY_3", "REFRESH", "LEAVE"]
                action_type = f"SHOP_{shop_names[shop_action] if shop_action < 6 else 'INVALID'}"
            
            # Removed debug printing for clean training output
            self.last_progress_log = self.step_count
        
        # print(f'[DEBUG] BalatroEnv.step called, action={action}, hand={self.hand}, money={self.money}, ante={self.current_ante}, blind={self.current_blind}')
        info = {
            'invalid_action': False,
            'reason': None,
            'chips_needed': getattr(self, 'chips_needed', None),
            'score': getattr(self, 'score', None),
            'blind': getattr(self, 'current_blind', None),
            'ante': getattr(self, 'current_ante', None),
            'win': False,
            'lose': False,
            'joker_effect': None,
            'money': getattr(self, 'money', None),
            'in_shop': getattr(self, 'in_shop_phase', False),
            'shop_items_count': len(getattr(self, 'shop_items', []))
        }
            
        if action < max_play_actions:
            # Play action: decode bitmask
            bitmask = action
            indices = [i for i in range(self.hand_size) if (bitmask & (1 << i))]
            selected_cards = [self.hand[i] for i in indices if i < len(self.hand)]
            
            reward, info = self._execute_play(selected_cards, info)
            
            # Inizializza terminated e truncated per questo blocco
            terminated = False
            truncated = False
            
            # Reset action counter ONLY on successful play that advances the game
            if 1 <= len(selected_cards) <= 5:  # Valid play action (1-5 cards)
                # Action counter removal - now handled by recent_actions system
                pass  # Sistema action_counter rimosso
                
            # CRITICAL FIX: Check win condition ONLY after playing a hand
            # This should NOT be checked for shop actions!
            if self.score >= self.chips_needed:
                # Salva lo score vincente prima che venga resettato
                winning_score = self.score
                info['winning_score'] = winning_score
                info['blind_beaten'] = True
                # MASSIVE REWARD for beating blinds, especially at high antes
                reward += 200.0 * self.current_ante  # Scales with ante
                if self.current_blind == 2:  # Boss blind
                    reward += 1000.0 * self.current_ante  # Huge bonus for completing an ante
                if self.current_ante >= 3:
                    reward += 500.0 * (self.current_ante - 2)  # Extra bonus for ante 3+
                obs = self._get_observation()
                return obs, reward, terminated, truncated, info
                # ...existing code for updating achievements, etc...
        
        elif action < base_actions:
            # Discard action: decode bitmask
            bitmask = action - max_play_actions
            indices = [i for i in range(self.hand_size) if (bitmask & (1 << i))]
            selected_cards = [self.hand[i] for i in indices if i < len(self.hand)]
            
            reward, info = self._execute_discard(selected_cards, info)
        else:
            # Shop action
            shop_action_id = action - base_actions
            
            # CRITICAL FIX: Reset info for shop actions to prevent inheriting combat info
            info = {
                'invalid_action': False,
                'reason': None,
                'chips_needed': self.chips_needed,
                'score': self.score,
                'blind': self.current_blind,
                'ante': self.current_ante,
                'win': False,
                'lose': False,
                'joker_effect': None,
                'money': self.money,
                'in_shop': True,
                'shop_items_count': len(self.shop_items),
                'blind_beaten': False  # Shop actions never beat blinds!
            }
            
            reward, info = self._execute_shop_action(shop_action_id, info)
            
            # Track shop time and add MODERATE temporal pressure
            if self.in_shop_phase:
                self.shop_steps += 1
                # MODERATE increasing penalty for staying too long in shop
                if self.shop_steps <= 2:
                    time_penalty = -1.0  # Penalità leggera per i primi 2 step
                elif self.shop_steps <= 4:
                    time_penalty = -2.0  # Penalità moderata step 3-4
                else:
                    time_penalty = -5.0 * (self.shop_steps - 4)  # Penalità crescente dopo
                reward += time_penalty
                
                # MODERATE penalty if overstaying
                if self.shop_steps >= self.max_shop_steps:
                    reward -= 20.0  # RIDOTTA da 50.0 a 20.0
                    info['shop_timeout'] = True
                    # Forza uscita automatica dallo shop
                    self._leave_shop(info)
        # Inizializza sempre terminated e truncated all'inizio
        terminated = False
        truncated = False
        
        # Aggiorna info con stato attuale
        info['chips_needed'] = self.chips_needed
        info['score'] = self.score
        info['blind'] = self.current_blind
        info['ante'] = self.current_ante
        # Assicura che info abbia sempre tutte le chiavi richieste
        for k in ['invalid_action','reason','chips_needed','score','blind','ante','win','lose','joker_effect']:
            if k not in info:
                info[k] = None
        # Check end of game
        if self.game_won or self.game_failed:
            info['win'] = self.game_won
            info['lose'] = self.game_failed
            info['reason'] = 'game_ended'
            # CORREZIONE: Usa episode_max_ante per riportare il massimo ante raggiunto nell'episodio
            info['ante_reached'] = self.episode_max_ante
            info['blinds_beaten'] = self._get_total_blinds_beaten()
            info['score'] = self.score
            return self._get_observation(), 0.0, True, False, info

        # Check lose condition - CONTINUOUS TRAINING
        if self.hands_left == 0 and self.score < self.chips_needed:
            # CONTINUOUS TRAINING: Auto-reset invece di terminare
            if self.continuous_training:
                self.game_failed = True  # Segna fallimento per statistiche
                info['lose'] = True
                info['failed'] = True
                info['reason'] = 'no_hands_left'
                info['ante_reached'] = self.episode_max_ante
                info['blinds_beaten'] = self._get_total_blinds_beaten()
                info['score'] = self.score
                
                # Auto-reset per continuare training
                self._internal_reset()
                terminated = False  # NON terminare l'episodio
                reward -= 1.0  # Penalità ridotta per sconfitta
                return self._get_observation(), reward, terminated, truncated, info
            else:
                # Training normale: termina episodio
                terminated = True
                
                # PENALITÀ PROPORZIONALE ai soldi rimasti (più soldi più penalità, ma nulla di eccessivo)
                money_penalty = min(self.money * 0.8, 15.0)  # Max 15 di penalità anche con tanti soldi
                reward -= (2.0 + money_penalty)  # Penalità base + proporzionale
                
                self.game_failed = True
                info['lose'] = True
                info['failed'] = True
                info['reason'] = 'no_hands_left'
                info['ante_reached'] = self.episode_max_ante
                info['blinds_beaten'] = self._get_total_blinds_beaten()
                info['score'] = self.score
                info['money_penalty'] = money_penalty  # Per debug/logging
                return self._get_observation(), reward, terminated, truncated, info

        # Progress reward - MOLTO ridotto per forzare focus sui blind
        if self.chips_needed > 0:
            progress = min(self.score / self.chips_needed, 1.0)
            # Progress reward molto piccolo
            reward += progress * 0.1  # Drasticamente ridotto da 0.5
            
            # Solo piccoli bonus per essere vicini
            if progress > 0.8:  # 80% of the way there
                reward += 0.2  # Era 0.5
            elif progress > 0.5:  # 50% of the way there
                reward += 0.1  # Era 0.2

        # Stagnation detection - penalize if no progress
        if self.score == self.last_score:
            self.stagnation_steps += 1
            if self.stagnation_steps > self.max_stagnation_steps:
                # PENALITA' PIU' LEGGERA per permettere learning
                reward -= 0.1  # Era 1.0 - molto ridotta
                if self.stagnation_steps > self.max_stagnation_steps * 2:
                    # Penalità graduale per stagnazione prolungata
                    reward -= 0.5  # Era 5.0 - molto ridotta
                    # Solo se stagnazione ESTREMA terminiamo l'episodio o auto-reset
                    if self.stagnation_steps > self.max_stagnation_steps * 8:  # Era *4, ora *8
                        if self.continuous_training:
                            # CONTINUOUS TRAINING: Auto-reset invece di terminare
                            self._internal_reset()
                            terminated = False
                            info['reason'] = 'extreme_stagnation_reset'
                            return self._get_observation(), -1.0, terminated, truncated, info  # Penalità ridotta
                        else:
                            # Training normale: termina episodio
                            terminated = True
                            info['reason'] = 'extreme_stagnation'
                            return self._get_observation(), -2.0, terminated, truncated, info  # Era -10.0
        else:
            # Score improved, reset stagnation counter
            self.stagnation_steps = 0
            self.last_score = self.score

        # === ANTI-EXPLOIT SYSTEM: DRASTICHE PENALITÀ per prevenire reward hacking ===
        
        # 1. AZIONI PRODUTTIVE: Categorizza e premia tutte le azioni che fanno progredire il gioco
        is_productive_action = False
        action_category = "unproductive"
        
        # Giocare una mano (massima priorità)
        if 'hand_type' in info and info.get('hand_type') is not None:
            is_productive_action = True
            action_category = "play_hand"
            reward += 30.0  # AUMENTATO bonus fisso per giocare una mano (da 20.0 a 30.0)
            if hasattr(self, 'consecutive_null_actions'):
                self.consecutive_null_actions = 0
        
        elif info.get('blind_beaten'):
            is_productive_action = True
            action_category = "blind_beaten"
            reward += 50.0  # Bonus significativo per battere un blind
            if hasattr(self, 'consecutive_null_actions'):
                self.consecutive_null_actions = 0
        
        # ACQUISTO JOKER: Azione strategica SOLO quando necessario ed economicamente sensata
        elif info.get('shop_action', '').startswith('bought_'):
            is_productive_action = True
            action_category = "joker_purchase"
            joker_purchased = info['shop_action'].replace('bought_', '')
            money_spent = info.get('money_spent', 0)
            
            # ECONOMIA INTELLIGENTE: Base reward dipende dalla situazione economica
            money_after_purchase = self.money  # Money after purchase
            
            # 1. VALUTAZIONE ECONOMICA: Penalizza acquisti che lasciano troppo pochi soldi
            if money_after_purchase < 2:
                economic_penalty = -15.0  # PENALITÀ per rimanere senza soldi
                joker_purchase_reward = 5.0  # Reward molto ridotto
            elif money_after_purchase < 4:
                economic_penalty = -5.0   # Penalità minore ma presente
                joker_purchase_reward = 15.0  # Reward ridotto
            else:
                economic_penalty = 0.0    # Nessuna penalità economica
                joker_purchase_reward = 25.0  # Full reward per acquisto sicuro
            
            # 2. VALUTAZIONE NECESSITÀ STRATEGICA: Joker necessari SOLO se non riesco a battere il blind
            # Calcola il potenziale massimo delle mani attuali
            estimated_hand_power = self._estimate_current_hand_power()
            remaining_score_needed = max(0, self.chips_needed - self.score)
            
            # Stima se possiamo battere il blind con le mani rimaste
            total_estimated_power = estimated_hand_power * self.hands_left
            blind_difficulty_ratio = remaining_score_needed / max(total_estimated_power, 1)
            
            # NECESSITÀ STRATEGICA basata su difficoltà del blind
            if blind_difficulty_ratio > 1.5:
                # MOLTO DIFFICILE: Joker fortemente necessario
                strategic_necessity = 20.0
                necessity_reason = "blind_molto_difficile"
            elif blind_difficulty_ratio > 1.0:
                # DIFFICILE: Joker moderatamente necessario  
                strategic_necessity = 10.0
                necessity_reason = "blind_difficile"
            elif blind_difficulty_ratio > 0.7:
                # GESTIBILE: Joker leggermente utile
                strategic_necessity = 3.0
                necessity_reason = "blind_gestibile"
            elif blind_difficulty_ratio > 0.5:
                # FACILE: Joker poco utile
                strategic_necessity = -5.0
                necessity_reason = "blind_facile"
            else:
                # MOLTO FACILE: Joker spreco di soldi
                strategic_necessity = -15.0
                necessity_reason = "blind_molto_facile"
            
            # VALUTAZIONE CONTEGGIO JOKER: Penalità per troppi joker
            joker_count = len(self.jokers)
            if joker_count >= 3:
                count_penalty = -10.0 * (joker_count - 2)  # Penalità crescente
            else:
                count_penalty = 0.0
            
            # Combina necessità strategica con conteggio
            necessity_bonus = strategic_necessity + count_penalty
            joker_purchase_reward += necessity_bonus
            
            # Removed debug printing for clean training
            
            # 3. VALUTAZIONE ANTE: Primi ante richiedono meno joker
            if self.current_ante <= 1:
                # Ante 1: joker non sempre necessari
                if joker_count >= 2:
                    ante_penalty = -8.0   # Penalità per troppi joker in ante 1
                    joker_purchase_reward += ante_penalty
            elif self.current_ante <= 2:
                # Ante 2: 1-2 joker sufficienti
                if joker_count >= 3:
                    ante_penalty = -5.0   # Penalità per troppi joker in ante 2
                    joker_purchase_reward += ante_penalty
            # Ante 3+: joker più necessari, nessuna penalità aggiuntiva
            
            # 4. VALUTAZIONE PREZZO: Penalizza acquisti troppo costosi per la situazione
            reasonable_price = 3 + self.current_ante  # Prezzo ragionevole aumenta con ante
            if money_spent > reasonable_price:
                price_penalty = -(money_spent - reasonable_price) * 2.0
                joker_purchase_reward += price_penalty
            
            # 5. BONUS QUALITÀ JOKER (solo se l'acquisto è economicamente sensato)
            if economic_penalty == 0.0:  # Solo se economia è buona
                strategic_bonuses = {
                    'joker': 3.0,              # Basic joker - reliable
                    'chip_joker': 4.0,         # Good early game
                    'mult_joker': 6.0,         # Strong late game
                    'bonus_joker': 8.0,        # Dual effect - excellent
                    'wild_joker': 5.0,         # Scaling potential
                    'lucky_joker': 7.0         # High risk/reward
                }
                joker_purchase_reward += strategic_bonuses.get(joker_purchased, 2.0)
            
            # Applica penalità economica
            joker_purchase_reward += economic_penalty
            
            # 6. RISULTATO FINALE: Assicura che reward sia ragionevole
            joker_purchase_reward = max(joker_purchase_reward, -20.0)  # Cap penalità
            joker_purchase_reward = min(joker_purchase_reward, 40.0)   # Cap reward
                
            reward += joker_purchase_reward
            # Removed debug printing for clean training
            
            if hasattr(self, 'consecutive_null_actions'):
                self.consecutive_null_actions = 0
        
        # 2. PENALITÀ SEMPLIFICATA: Solo per inattività
        if not is_productive_action and not info.get('invalid_action'):
            if not hasattr(self, 'consecutive_null_actions'):
                self.consecutive_null_actions = 0
            self.consecutive_null_actions += 1
            
            # Penalità fissa per ogni azione non produttiva
            unproductive_penalty = -0.1
            reward += unproductive_penalty
            info['unproductive_penalty'] = unproductive_penalty
            info['consecutive_unproductive'] = self.consecutive_null_actions
        
        # 3. NESSUN ALTRO REWARD/PENALTY per massima semplicità
        
        # 🚀 TEMPORAL PRESSURE: Penalità crescente per incentivare azioni rapide
        if not self.in_shop_phase:
            # Penalità temporale leggera ma crescente per steps senza giocare mani
            temporal_penalty = -0.02 * self.steps_since_last_hand
            reward += temporal_penalty
            
            if self.steps_since_last_hand > 30:  # Warning threshold
                reward -= 1.0  # Penalità aggiuntiva
                info['temporal_warning'] = True
        
        # Logging per debug
        info['action_category'] = action_category
        
        # Clean training - no reward logging
        return self._get_observation(), reward, terminated, truncated, info
    
    def _internal_reset(self):
        """Reset interno per training continuo - mantiene statistiche globali"""
        # Reset game state
        self.current_ante = 1
        self.current_blind = 0
        self.money = self.starting_money
        self.hands_left = 4
        self.discards_left = 3
        self.jokers = []
        self.shop_items = []
        self.shop_refreshes_left = 2
        self.in_shop_phase = False
        self.shop_steps = 0
        self.shop_refreshes = 0
        self.score = 0
        self.played_cards_this_round = []
        self.game_won = False
        self.game_failed = False
        
        # Reset run-specific tracking, mantieni statistiche globali
        self.best_score_this_run = 0
        self.episode_max_ante = 1  # Reset per nuovo episodio
        self.current_run_hands = []
        self.current_run_jokers = []
        
        # Reset forced play system
        self.steps_since_last_hand = 0
        self.forced_play_count = 0
        
        # Reset anti-spam systems
        self.recent_actions = []
        self.consecutive_discards = 0
        if hasattr(self, 'consecutive_null_actions'):
            self.consecutive_null_actions = 0
        if hasattr(self, 'consecutive_high_cards'):
            self.consecutive_high_cards = 0
        
        # Reset stagnation detection
        self.last_score = 0
        self.stagnation_steps = 0
        
        # NON resettare: highest_ante_reached, total_hands_played, hand_type_stats
        # Queste statistiche si accumulano nel training continuo
        
        # Create new deck and deal hand
        self._create_deck()
        self._deal_hand()
        self._set_blind()
    
    def _execute_play(self, selected_cards: List[Card], info: Dict) -> Tuple[float, Dict]:
        """Execute a play action"""
        if not (1 <= len(selected_cards) <= 5):
            info['action_type'] = 'invalid_play'
            info['joker_effect'] = None
            info['invalid_reason'] = f'need_1_to_5_cards_got_{len(selected_cards)}'
            # Stronger penalty for very wrong selections
            penalty = -0.5 if len(selected_cards) == 0 else -0.2
            return penalty, info
        if self.hands_left == 0:
            info['action_type'] = 'invalid_play_no_hands'
            info['joker_effect'] = None
            info['invalid_reason'] = 'no_hands_left'
            return -0.5, info
        
        # Valid play
        self.hands_left -= 1
        
        # 🚀 RESET FORCED PLAY COUNTER: Hand was played successfully
        self.steps_since_last_hand = 0
        
        # 🚀 NUOVO: Trova automaticamente la migliore combinazione di 5 carte dalla mano completa!
        # Invece di usare solo le carte selezionate, trova la migliore combinazione possibile
        if len(self.hand) >= 5:
            # Se abbiamo almeno 5 carte, trova la migliore combinazione di 5
            best_cards, hand_type, chips, mult = self._find_best_5_card_hand(self.hand)
            # Aggiorna selected_cards per logging/info
            actual_played_cards = best_cards
            info['cards_analyzed'] = len(self.hand)
            info['best_hand_found'] = hand_type.value
            info['original_selection'] = len(selected_cards)
        else:
            # Se abbiamo meno di 5 carte, usa tutte le carte disponibili
            hand_type, chips, mult = PokerEvaluator.evaluate_hand(self.hand)
            actual_played_cards = self.hand.copy()
            info['cards_analyzed'] = len(self.hand)
            info['best_hand_found'] = hand_type.value
            info['original_selection'] = len(selected_cards)
        
        # Track hand type statistics - GLOBAL e RECENT
        self.hand_type_stats[hand_type.value] += 1
        self.total_hands_played += 1
        
        # CONTINUOUS TRAINING: Track cumulative hands
        if hasattr(self, 'continuous_hands_played'):
            self.continuous_hands_played += 1
        
        total_bonus_chips = 0
        total_bonus_mult = 0
        joker_effect = None
        for joker in self.jokers:
            bonus_chips, bonus_mult = joker.get_effect(actual_played_cards, hand_type)
            total_bonus_chips += bonus_chips
            total_bonus_mult += bonus_mult
            
        # All joker effects now handled in get_effect() method
        if any(joker.joker_type == JokerType.LUCKY_JOKER for joker in self.jokers):
            joker_effect = 'lucky_joker_proc'  # Will show when lucky triggers
            
        # Calculate final score with joker bonuses
        final_chips = chips + total_bonus_chips
        final_mult = mult + total_bonus_mult
        hand_score = final_chips * final_mult
        self.score += hand_score

        # 🚀 DEBUG: Messaggio dettagliato per ogni mano giocata
        blind_needed = self.chips_needed
        current_score = self.score
        will_beat_blind = current_score >= blind_needed
        
        debug_msg = (f"🎮 MANO #{self.total_hands_played + 1}: "
                    f"{hand_type.value.upper()} | "
                    f"Score: {hand_score} ({final_chips}×{final_mult}) | "
                    f"Totale: {current_score}/{blind_needed} | "
                    f"{'✅ BLIND BATTUTO!' if will_beat_blind else '❌ Non sufficiente'} | "
                    f"Ante {self.current_ante}, Blind {self.current_blind + 1}")
        
        # Stampa debug solo se debug=True nell'environment
        if getattr(self, 'debug', False):
            print(debug_msg)

        # HIGH ANTE TRACKING: Record this hand for potential ante achievement
        hand_record = {
            'hand_type': hand_type.value,
            'chips': chips,
            'mult': mult,
            'score': final_chips * final_mult,
            'ante': self.current_ante,
            'blind': self.current_blind,
            'cards_played': [str(card) for card in actual_played_cards],
            'jokers_active': [j.joker_type.value for j in self.jokers],
            'money': self.money,
            'debug_info': debug_msg  # Aggiungi info debug al record
        }
        self.current_run_hands.append(hand_record)

        # Track recent hands for convergence monitoring
        self.recent_hand_types.append(hand_type.value)
        if len(self.recent_hand_types) > self.recent_window_size:
            self.recent_hand_types.pop(0)  # Keep only last 10k hands
        
        # 🚀 NUOVO: Rimuovi le carte effettivamente giocate (migliore combinazione) dalla mano
        for card in actual_played_cards:
            if card in self.hand:
                self.hand.remove(card)
        self._deal_hand()
        info.update({
            'action_type': 'play',
            'hand_score': hand_score,
            'hand_type': hand_type.value,
            'chips': final_chips,
            'mult': final_mult,
            'joker_effect': joker_effect,
            'hand_type_stats': dict(self.hand_type_stats),
            'total_hands_played': self.total_hands_played
        })
        
        # Base reward starts POSITIVE - reward per ogni azione valida per evitare negatività costante
        base_reward = 2.0  # INCREASED: Reward base positivo per ogni azione per bilanciare il sistema
        
        # 🚀 NUOVO: Bonus basato su carte effettivamente giocate (migliore combinazione)
        cards_bonus = len(actual_played_cards) * 2.0  # Usa le carte della migliore combinazione
        
        # GRADUATED PRESSURE: Smart reward shaping for learning
        best_possible_hand = self._get_best_possible_hand_from_all_cards()
        hand_rankings = {
            HandType.HIGH_CARD: 0, HandType.PAIR: 1, HandType.TWO_PAIR: 2,
            HandType.THREE_OF_A_KIND: 3, HandType.STRAIGHT: 4, HandType.FLUSH: 5,
            HandType.FULL_HOUSE: 6, HandType.FOUR_OF_A_KIND: 7,
            HandType.STRAIGHT_FLUSH: 8, HandType.ROYAL_FLUSH: 9
        }
        
        current_rank = hand_rankings[hand_type]
        best_rank = hand_rankings[best_possible_hand]
        
        # SMART GRADUATED REWARDS
        if current_rank == best_rank:
            # Excellent: Found the optimal hand - RIDOTTO
            strategy_bonus = 2.0  # Ridotto da 4.0
        elif current_rank >= best_rank - 1:
            # Good: Found a very good hand - RIDOTTO
            strategy_bonus = 1.0  # Ridotto da 2.0
        elif current_rank >= best_rank - 2:
            # Okay: Found a decent hand
            strategy_bonus = 0.3  # Ridotto da 0.5
        elif current_rank >= best_rank - 3:
            # Poor: Could do much better
            strategy_bonus = -0.5  # Ridotto da -1.0
        else:
            # Very poor: Missing obvious opportunities
            strategy_bonus = -1.0  # Ridotto da -2.0
            
        # EDUCATIONAL PENALTY: MOLTO PIÙ AGGRESSIVO - punire high card quando disponibili combinazioni
        if hand_type == HandType.HIGH_CARD and best_possible_hand != HandType.HIGH_CARD:
            # Penalità graduate molto più severe per spingere l'apprendimento
            missed_opportunity = best_rank
            if missed_opportunity >= 3:  # Missed tris or better
                strategy_bonus -= 8.0  # MOLTO AUMENTATO da 2.0
            elif missed_opportunity >= 2:  # Missed two pair
                strategy_bonus -= 6.0  # MOLTO AUMENTATO da 1.5
            elif missed_opportunity >= 1:  # Missed pair
                strategy_bonus -= 4.0  # MOLTO AUMENTATO da 1.0
                
        # ANTI-HIGH-CARD PRESSURE: Penalità aggiuntiva se gioca troppe high card consecutive
        if hand_type == HandType.HIGH_CARD:
            # Traccia high card consecutive per penalizzarle
            if not hasattr(self, 'consecutive_high_cards'):
                self.consecutive_high_cards = 0
            self.consecutive_high_cards += 1
            # Penalità crescente per high card consecutive
            if self.consecutive_high_cards >= 3:
                strategy_bonus -= 2.0 * self.consecutive_high_cards  # Penalità esponenziale
        else:
            # Reset counter se gioca una mano diversa da high card
            if hasattr(self, 'consecutive_high_cards'):
                self.consecutive_high_cards = 0
                
        # EXPLORATION BONUS: Bilanciato per strategia vs gioco
        exploration_bonus = 0.5 * len(actual_played_cards)  # 🚀 NUOVO: Usa carte effettivamente giocate
        
        # ANTI-DISCARD BONUS: Bilanciato
        play_action_bonus = 1.0  # Aumentato a 1.0
        
        # GRADUATED PRESSURE: Update learning progress
        self.learning_progress['total_selections'] += 1
        if strategy_bonus > 0:  # Good selection
            self.learning_progress['smart_selections'] += 1
        
        # Update success rate
        if self.learning_progress['total_selections'] > 0:
            self.learning_progress['success_rate'] = (
                self.learning_progress['smart_selections'] / 
                self.learning_progress['total_selections']
            )
            
        # Track best hands found
        self.learning_progress['best_hands_found'][hand_type] += 1
        
        # HAND TYPE REWARDS: FOCUS PROGRESSIONE, non mani perfette
        hand_type_rewards = {
            HandType.HIGH_CARD: 0.0,
            HandType.PAIR: 0.5,
            HandType.TWO_PAIR: 1.0,
            HandType.THREE_OF_A_KIND: 2.0,
            HandType.STRAIGHT: 3.0,
            HandType.FLUSH: 3.5,
            HandType.FULL_HOUSE: 4.0,
            HandType.FOUR_OF_A_KIND: 5.0,
            HandType.STRAIGHT_FLUSH: 6.0,
            HandType.ROYAL_FLUSH: 7.0
        }
        hand_type_reward = hand_type_rewards.get(hand_type, 0.0)
        
        base_reward += cards_bonus + strategy_bonus + exploration_bonus + play_action_bonus + hand_type_reward
        
        # NEW HIGH SCORE REWARD: Reward scaling with improvement and ante difficulty
        if self.score > self.best_score_this_run:
            score_improvement = self.score - self.best_score_this_run
            
            # Ante scaling: More reward for improvements in early antes - RIDOTTO
            # Ante 1: 0.5x, Ante 2: 0.4x, Ante 3: 0.3x, Ante 4+: 0.2x (dimezzati)
            if self.current_ante == 1:
                ante_multiplier = 0.5  # Ridotto da 1.0
            elif self.current_ante == 2:
                ante_multiplier = 0.4  # Ridotto da 0.7
            elif self.current_ante == 3:
                ante_multiplier = 0.3  # Ridotto da 0.5
            else:
                ante_multiplier = 0.2  # Ridotto da 0.3
            
            # Reward scales with percentage improvement, ante difficulty, capped - PIÙ CONSERVATIVO
            high_score_reward = min((score_improvement / 200.0) * ante_multiplier, 2.0)  # Ridotto da /100 e cap 5.0
            base_reward += high_score_reward
            self.best_score_this_run = self.score
            
            # Also update session high score with ante consideration - RIDOTTO
            if self.score > self.run_high_score:
                self.run_high_score = self.score
                # Session record bonus also scales with ante difficulty - MODERATO
                session_bonus = 1.0 * ante_multiplier  # Ridotto da 2.0
                base_reward += session_bonus
            
        score_bonus = min(hand_score / 300.0, 0.5)  # Bonus più contenuto (da /200 a /300, cap da 1.0 a 0.5)
        
        # Add detailed reward breakdown to info
        info.update({
            'reward_breakdown': {
                'cards_bonus': cards_bonus,
                'strategy_bonus': strategy_bonus, 
                'exploration_bonus': exploration_bonus,
                'play_action_bonus': play_action_bonus,
                'hand_type_reward': hand_type_reward,
                'score_bonus': score_bonus,
                'total_reward': base_reward + score_bonus
            },
            # 🚀 NUOVO: Info sulla ricerca automatica della migliore combinazione
            'auto_hand_analysis': {
                'total_cards_in_hand': len(self.hand) + len(actual_played_cards),  # Prima di rimuovere
                'cards_played': len(actual_played_cards),
                'best_hand_found': hand_type.value,
                'original_user_selection': len(selected_cards),
                'improvement_found': hand_type != HandType.HIGH_CARD or len(actual_played_cards) > 1
            }
        })
        
        return base_reward + score_bonus, info
    
    def _execute_discard(self, selected_cards: List[Card], info: Dict) -> Tuple[float, Dict]:
        """Execute a discard action with SMART STRATEGIC REWARDS"""
        if not (1 <= len(selected_cards) <= self.hand_size):
            info['action_type'] = 'invalid_discard_selection'
            return -0.1, info
        
        if self.discards_left == 0:
            info['action_type'] = 'invalid_discard_no_discards'
            return -0.5, info
        
        # SMART DISCARD SYSTEM: Track hand state before discard
        self.pre_discard_hand_state = {
            'best_hand_type': self._get_best_possible_hand_from_all_cards(),
            'hand_potential': self._evaluate_hand_potential(self.hand),
            'hand_copy': self.hand.copy()
        }
        
        # ANALYZE STRATEGIC POTENTIAL BEFORE discarding
        strategic_reward = 0.0 # self._calculate_strategic_discard_reward(selected_cards)
        
        # Valid discard
        self.discards_left -= 1
        self.discard_attempts += 1
        
        # Remove discarded cards and refill hand
        for card in selected_cards:
            if card in self.hand:
                self.hand.remove(card)
        
        self._deal_hand()
        
        # SMART REWARD: Check if we actually improved after discard
        post_discard_improvement = self._calculate_post_discard_improvement()
        
        info['action_type'] = 'discard'
        info['strategic_discard'] = strategic_reward > 0
        info['strategic_reward'] = strategic_reward
        info['improvement_reward'] = post_discard_improvement
        info['hand_improved'] = post_discard_improvement > 0
        
        # BASE REWARD per discard valido - incoraggia l'uso strategico 
        base_penalty = 1.0  # INCREASED: Cambiato da penalità a reward per discard validi
        
        # PENALITÀ LIEVE per discard non strategici (molto ridotta)
        non_strategic_penalty = 0.0
        if strategic_reward <= 0 and post_discard_improvement <= 0:
            # Discard non ottimale - penalità molto lieve
            non_strategic_penalty = -0.5  # DECREASED: Ridotta drasticamente da -3.0
            if self.discards_left <= 1:  # Ultima discard sprecata
                non_strategic_penalty = -1.0  # DECREASED: Ridotta drasticamente da -5.0
        
        total_reward = base_penalty + strategic_reward + post_discard_improvement + non_strategic_penalty
        
        return total_reward, info
    
    def _calculate_joker_purchase_reward(self, joker: Joker, price: int) -> float:
        """Calculate strategic reward for purchasing a joker based on game state - BILANCIATO"""
        base_reward = 5.0  # INCREASED: Aumentato per rendere acquisto joker sempre positivo
        strategic_bonus = 0.0
        
        # 1. EARLY GAME ECONOMY: Joker economici sono buoni early - BILANCIATO
        if self.current_ante <= 2:
            if price <= 3:  # Ridotto da 4
                strategic_bonus += 0.5
            elif price <= 5:  # Ridotto da 6
                strategic_bonus += 0.3
            else:
                strategic_bonus -= 0.2  # Meno penalità per joker costosi
        
        # 2. LATE GAME POWER: Joker costosi diventano necessari - BILANCIATO
        elif self.current_ante >= 4:
            if price >= 5:  # Ridotto da 6
                strategic_bonus += 1.0
            elif price >= 3:  # Ridotto da 4
                strategic_bonus += 0.5
            else:
                strategic_bonus += 0.2
        
        # 3. JOKER TYPE SPECIFIC BONUSES - RIDOTTI PER NUOVI VALORI
        joker_type = joker.joker_type
        
        # BONUS_JOKER: Dual effect ancora prezioso ma meno potente (+8/+2 invece di +15/+3)
        if joker_type == JokerType.BONUS_JOKER:
            # Extra bonus se siamo in difficoltà (ante alti + pochi soldi)
            if self.current_ante >= 2 and self.money <= 10:
                strategic_bonus += 2.0  # Ridotto da 3.0 (dual effect meno potente)
            else:
                strategic_bonus += 1.5  # Ridotto da 2.0
            
        # WILD/LUCKY: Joker con potenziale ridotto (WILD: +1 ogni 2 carte, LUCKY: 30% invece 50%)
        elif joker_type in [JokerType.WILD_JOKER, JokerType.LUCKY_JOKER]:
            strategic_bonus += 1.5  # Ridotto da 2.5 (meno potenti ora)
            
        # MULT_JOKER: Più debole (+3 invece di +6)
        elif joker_type == JokerType.MULT_JOKER:
            if self.current_ante <= 3:
                strategic_bonus += 1.0  # Ridotto da 2.0 (meno potente)
            else:
                strategic_bonus += 0.8  # Ridotto da 1.5
                
        # CHIP_JOKER: Più debole (+15 invece di +30)
        elif joker_type == JokerType.CHIP_JOKER:
            strategic_bonus += 0.8  # Ridotto da 1.2 (meno potente)
            
        # Basic JOKER: Più debole (+2 invece di +4)
        elif joker_type == JokerType.JOKER:
            strategic_bonus += 0.5  # Ridotto da 0.7 (meno potente)
            
        # PREMIUM_JOKER: Most powerful but expensive
        elif joker_type == JokerType.PREMIUM_JOKER:
            # Premium joker gives huge value but costs a lot
            if self.current_ante >= 3:  # Late game where power is critical
                strategic_bonus += 3.0  # Massive bonus for late game power
            elif self.current_ante >= 2:  # Mid game
                strategic_bonus += 2.0  # Good value
            else:  # Early game - expensive but still good
                strategic_bonus += 1.0  # Decent even early
            
            # Extra bonus if we have money to spare (can afford premium)
            if self.money >= 15:
                strategic_bonus += 1.0  # Can afford luxury
        
        # 4. ECONOMIC SITUATION BONUS/PENALTY
        money_after_purchase = self.money - price
        
        # Penalizza se questo acquisto ci lascia al verde
        if money_after_purchase < 2:
            strategic_bonus -= 1.0  # Non andare al verde! (da -2.0 a -1.0)
        elif money_after_purchase < 5:
            strategic_bonus -= 0.5  # Attenzione ai soldi (da -1.0 a -0.5)
        elif money_after_purchase >= 10:
            strategic_bonus += 0.5  # Possiamo permettercelo facilmente (da 1.0 a 0.5)
        
        # 5. JOKER COLLECTION SYNERGY
        existing_types = [j.joker_type for j in self.jokers]
        
        # Avoid too many duplicates unless it's a great joker
        if joker_type in existing_types:
            if joker_type == JokerType.BONUS_JOKER:
                strategic_bonus += 1.0  # Multiple bonus jokers are good
            elif joker_type == JokerType.PREMIUM_JOKER:
                strategic_bonus += 2.0  # Multiple premium jokers are amazing
            elif joker_type == JokerType.JOKER:
                strategic_bonus += 0.5  # Basic jokers stack okay
            else:
                strategic_bonus -= 1.0  # Usually avoid duplicates
        
        # 6. URGENCY BONUS: Need power for harder blinds
        chips_ratio = self.chips_needed / max(150, 150)  # Normalized to early blind difficulty
        if chips_ratio > 2.0:  # Much harder than early game
            strategic_bonus += 2.0  # Need more power!
        elif chips_ratio > 1.5:
            strategic_bonus += 1.0  # Getting harder
        
        total_reward = base_reward + strategic_bonus
        
        # Cap reward in range ragionevole e bilanciato
        return max(0.2, min(total_reward, 8.0))  # Da (0.5, 15.0) a (0.2, 8.0)
    
    def _refresh_shop(self, info: Dict) -> Tuple[float, Dict]:
        """Refresh shop items (costs $2) with DYNAMIC PENALTIES based on game phase"""
        if not self.in_shop_phase:
            return 0.0, info
        
        # Penalità crescente se siamo in shop da troppo tempo
        penalty = -0.1 * self.shop_steps
        
        # Limitiamo la penalità per refresh a un massimo di -2.0
        if penalty < -2.0:
            penalty = -2.0
        
        # Rimuovi tutti gli oggetti attuali nello shop
        self.shop_items = []
        # Aggiungi nuovi oggetti (fino a 4) con logica semplificata
        num_items = random.randint(1, 4)
        for _ in range(num_items):
            item_type = random.choice(["joker", "card", "pack"])
            content = None
            price = random.randint(2, 10)
            description = ""
            
            if item_type == "joker":
                content = Joker(JokerType.JOKER)  # Joker di base per semplificare
            elif item_type == "card":
                content = Card(random.choice(list(Suit)), random.choice(list(Rank)))
            elif item_type == "pack":
                content = "Standard Pack"  # Tipo di pack fisso per semplificare
            
            self.shop_items.append(ShopItem(item_type, content, price))
        
        # Reset contatore refresh
        self.shop_refreshes_left = 2
        
        return penalty, info
    
    def _calculate_strategic_discard_reward(self, selected_cards: List[Card]) -> float:
        """DUMMY: Calculate strategic reward for discarding cards."""
        return 0.0

    def _get_hand_type_reward(self, hand_type: HandType) -> float:
        """DUMMY: Get reward for a specific hand type."""
        return 0.0

    def _enter_shop_phase(self):
        """Enter the shop phase between blinds."""
        self.in_shop_phase = True
        self.shop_steps = 0
        self.shop_refreshes = 0  # Conta i refresh in questa sessione shop
        self.shop_items = self._generate_shop_items()
        joker_count = len([item for item in self.shop_items if item.item_type == 'joker'])
        if self.debug:
            print(f"[SHOP] Entrato nello shop: ${self.money}, {joker_count} joker, {len(self.jokers)}/{self.max_jokers} owned")
        # NON advance blind automaticamente - l'agente deve scegliere

    def _exit_shop_phase(self):
        """Exit the shop phase and advance to next blind."""
        self.in_shop_phase = False
        self.shop_steps = 0
        
        # Only advance blind if game is not ended
        if not self.game_won and not self.game_failed:
            # Validate current state before advancing
            if (self.current_ante <= self.max_ante and 
                self.current_ante in self.blinds):
                self._advance_blind()
            else:
                print(f"[WARNING] Cannot advance blind from invalid state: ante {self.current_ante}, blind {self.current_blind}")
                # Force end game if in invalid state
                self.game_failed = True

    def _leave_shop(self, info: Dict) -> Tuple[float, Dict]:
        """Leave the shop and advance to next blind."""
        reward = 5.0  # INCREASED: Base reward positivo per uscire dal negozio
        
        # REWARD INTELLIGENTE per uscita dal negozio
        base_exit_reward = 15.0  # RIDOTTO - uscita è importante ma non troppo premiata
        
        # ECONOMIA REWARD: Premia uscita con soldi rimasti
        money_conservation_bonus = min(self.money * 1.5, 12.0) # Più soldi = più reward
        
        # JOKER COUNT EVALUATION: Premia uscita con numero appropriato di joker
        joker_count = len(self.jokers)
        if joker_count == 0:
            # Nessun joker - penalità se siamo oltre ante 1
            if self.current_ante >= 2:
                joker_penalty = -8.0  # Penalità per non aver comprato joker necessari
            else:
                joker_penalty = -2.0  # Penalità leggera per ante 1
            reward += joker_penalty
        elif joker_count == 1:
            # 1 joker - buono per primi ante
            if self.current_ante <= 2:
                joker_bonus = 5.0
            else:
                joker_bonus = 2.0  # Forse ne servono di più per ante alti
            reward += joker_bonus
        elif joker_count == 2:
            # 2 joker - ottimo per la maggior parte delle situazioni
            joker_bonus = 8.0
            reward += joker_bonus
        elif joker_count == 3:
            # 3 joker - buono per ante alti
            if self.current_ante >= 3:
                joker_bonus = 6.0
            else:
                joker_bonus = 3.0  # Forse troppi per ante bassi
            reward += joker_bonus
        else:
            # 4+ joker - potrebbe essere eccessivo
            joker_penalty = -5.0 * (joker_count - 3)  # Penalità crescente
            reward += joker_penalty
        
        # STRATEGIC SHOP EXIT REWARD basato sui step
        if self.shop_steps <= 1:
            # Uscita troppo rapida - potrebbe aver perso opportunità
            if joker_count == 0:
                quick_exit_penalty = -5.0  # Penalità per uscita rapida senza acquisti
            else:
                quick_exit_penalty = 0.0   # OK se ha già joker
            reward += quick_exit_penalty
        elif self.shop_steps <= 3:
            # Uscita ragionevole - ha valutato le opzioni
            strategic_bonus = 5.0
            reward += strategic_bonus
        elif self.shop_steps <= 4:
            # Uscita accettabile
            strategic_bonus = 2.0
            reward += strategic_bonus
        # else: nessun bonus per uscite troppo lente
        
        reward += base_exit_reward + money_conservation_bonus
        
        # ECONOMIC WISDOM: Bonus for non sprecare soldi in refresh
        if self.shop_refreshes == 0:  # NESSUN refresh = economicamente saggio!
            reward += 5.0  # AUMENTATO da 1.0
            info['economic_bonus'] = 'no_refresh_bonus'
        elif self.shop_refreshes <= 1:  # Solo 1 refresh = accettabile
            reward += 2.0  # AUMENTATO da 0.5
        # else: troppi refresh = nessun bonus (la perdita dei $2 è già punizione)
        
        # ECONOMIC WISDOM REWARD: Balance money conservation vs joker investment
        joker_count = len(self.jokers)
        if joker_count >= 2 and self.money >= 2:  # Good balance: some jokers + some money
            reward += 10.0  # AUMENTATO da 5.0
            if self.debug:
                print(f"[REWARD] Buon bilanciamento economia: {joker_count} joker, ${self.money} money")
        elif joker_count >= 3:  # Good joker collection
            reward += 8.0  # AUMENTATO da 3.0
        elif joker_count >= 1:  # Almeno un joker comprato
            reward += 5.0  # NUOVO: reward per almeno fare qualcosa
        elif self.money >= 6:  # Conservative but risky - might need more jokers
            reward += 2.0  # AUMENTATO da 1.0
        
        info['shop_action'] = 'leave'
        info['final_money'] = self.money
        info['final_jokers'] = len(self.jokers)
        
        if self.debug:
            print(f"[SHOP EXIT] REWARD TOTALE: +{reward:.1f} per uscire dal negozio!")
        
        self._exit_shop_phase()
        return reward, info

    def _execute_shop_action(self, action_id: int, info: Dict) -> Tuple[float, Dict]:
        """Execute a shop action: 0-3=buy_item, 4=refresh, 5=leave"""
        self.shop_steps += 1
        reward = 1.0  # INCREASED: Base reward positivo per azioni shop
        
        # REDUCED LOGGING: Only log significant actions
        log_action = action_id in [0, 1, 2, 3, 4, 5] and (action_id <= 3 and action_id < len(self.shop_items)) or action_id >= 4
        
        if action_id <= 3:  # Buy item 0-3
            if action_id < len(self.shop_items):
                item = self.shop_items[action_id]
                
                # Check if can afford
                if self.money >= item.price:
                    # Check if can fit joker
                    if item.item_type == 'joker' and len(self.jokers) < self.max_jokers:
                        # SUCCESSFUL PURCHASE
                        self.money -= item.price
                        self.jokers.append(item.content)
                        self.shop_items.pop(action_id)  # Remove bought item
                        reward += 5.0  # Reward for successful purchase
                        
                        # HIGH ANTE TRACKING: Record joker acquisition
                        joker_acquisition = {
                            'joker_type': item.content.joker_type.value,
                            'price': item.price,
                            'ante': self.current_ante,
                            'blind': self.current_blind,
                            'money_after': self.money,
                            'total_jokers': len(self.jokers)
                        }
                        self.current_run_jokers.append(joker_acquisition)
                        
                        if self.debug:
                            print(f"[SHOP] Acquistato {item.content.joker_type.value} per ${item.price}. Money rimasti: ${self.money}")
                        info['shop_action'] = f'bought_{item.content.joker_type.value}'
                        info['money_spent'] = item.price
                    elif item.item_type == 'joker':
                        # No space for joker
                        reward -= 1.0
                        info['shop_action'] = 'buy_failed_no_space'
                    else:
                        # Other item types not implemented yet
                        reward -= 0.5
                        info['shop_action'] = 'buy_failed_unsupported'
                else:
                    # Can't afford
                    reward -= 2.0
                    info['shop_action'] = 'buy_failed_no_money'
            else:
                # Invalid item index
                reward -= 1.0
                info['shop_action'] = 'buy_failed_invalid_item'
                
        elif action_id == 4:  # Refresh shop
            if self.money >= 2:  # Refresh costs $2
                self.money -= 2
                self.shop_refreshes += 1  # Conta i refresh
                self.shop_items = self._generate_shop_items()
                # NO penalità artificiale - il costo di $2 È il deterrente naturale
                # Il modello deve imparare che perdere $2 è costoso dall'economia del gioco
                info['shop_action'] = 'refresh'
                info['shop_refreshes'] = self.shop_refreshes
                if self.debug:
                    print(f"[SHOP] Shop refreshed per $2. Money rimasti: ${self.money}")
            else:
                reward -= 1.0
                info['shop_action'] = 'refresh_failed_no_money'
                
        elif action_id == 5:  # Leave shop
            reward, info = self._leave_shop(info)
            
        else:
            # Invalid action
            reward -= 1.0
            info['shop_action'] = 'invalid_action'
        
        # ECONOMY PRESSURE: Penalità moderate per i primi step, poi aggressive
        if self.shop_steps > 3:  # Dai tempo per 2-3 acquisti
            reward -= (self.shop_steps - 3) * 10.0  # MASSIVA penalità solo dopo 3 step
        elif self.shop_steps > 2:  # Piccola penalità dopo 2 step
            reward -= 2.0
            
        # FORCE EXIT: Auto-exit if stuck in shop too long
        if self.shop_steps > 5:  # Aumentato da 3 a 5
            print(f"[SHOP] FORCED EXIT: Too many shop steps ({self.shop_steps})")
            reward, info = self._leave_shop(info)
            
        return reward, info

    def _get_best_possible_hand_from_all_cards(self) -> HandType:
        """DUMMY: Returns the simplest hand type to avoid crashes."""
        return HandType.HIGH_CARD

    def _evaluate_hand_potential(self, cards: List[Card]) -> Dict:
        """DUMMY: Returns a default potential dictionary to avoid crashes."""
        return {'potential_type': HandType.HIGH_CARD, 'strength': 0, 'draw_type': 'none'}

    def _find_best_5_card_hand(self, cards: List[Card]) -> Tuple[List[Card], HandType, int, int]:
        """Find the best possible 5-card hand from the given cards"""
        if len(cards) < 5:
            # If we have fewer than 5 cards, use all of them
            hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
            return cards, hand_type, chips, mult
        
        # Try all combinations of 5 cards and find the best one
        from itertools import combinations
        best_hand = None
        best_type = HandType.HIGH_CARD
        best_chips = 0
        best_mult = 0
        best_score = 0
        
        for combo in combinations(cards, 5):
            hand_type, chips, mult = PokerEvaluator.evaluate_hand(list(combo))
            # Calculate score for comparison (chips * mult)
            score = chips * mult
            
            # Hand type hierarchy for comparison
            type_values = {
                HandType.HIGH_CARD: 1,
                HandType.PAIR: 2,
                HandType.TWO_PAIR: 3,
                HandType.THREE_OF_A_KIND: 4,
                HandType.STRAIGHT: 5,
                HandType.FLUSH: 6,
                HandType.FULL_HOUSE: 7,
                HandType.FOUR_OF_A_KIND: 8,
                HandType.STRAIGHT_FLUSH: 9,
                HandType.ROYAL_FLUSH: 10
            }
            
            type_value = type_values.get(hand_type, 0)
            
            # Compare by hand type first, then by score
            if (type_value > type_values.get(best_type, 0)) or \
               (type_value == type_values.get(best_type, 0) and score > best_score):
                best_hand = list(combo)
                best_type = hand_type
                best_chips = chips
                best_mult = mult
                best_score = score
        
        return best_hand, best_type, best_chips, best_mult

    def _estimate_current_hand_power(self) -> float:
        """Stima il potenziale punteggio della migliore mano giocabile dalla mano attuale.
        
        Questo valuta:
        1. La migliore combinazione di 5 carte (o meno se disponibili)
        2. Gli effetti dei joker attuali
        3. Ritorna una stima del punteggio potenziale per una singola mano
        """
        if not self.hand:
            return 0.0
        
        # Trova la migliore combinazione possibile
        if len(self.hand) >= 5:
            best_cards, hand_type, chips, mult = self._find_best_5_card_hand(self.hand)
        else:
            hand_type, chips, mult = PokerEvaluator.evaluate_hand(self.hand)
            best_cards = self.hand.copy()
        
        # Applica gli effetti dei joker come nella vera giocata
        total_bonus_chips = 0
        total_bonus_mult = 0
        
        for joker in self.jokers:
            bonus_chips, bonus_mult = joker.get_effect(best_cards, hand_type)
            total_bonus_chips += bonus_chips
            total_bonus_mult += bonus_mult
        
        # Calcola il punteggio finale stimato
        final_chips = chips + total_bonus_chips
        final_mult = mult + total_bonus_mult
        estimated_score = final_chips * final_mult
        
        return float(estimated_score)

    def _calculate_post_discard_improvement(self) -> float:
        """DUMMY: Returns zero improvement to avoid crashes."""
        return 0.0
        
    def _generate_shop_items(self) -> List[ShopItem]:
        """Generate shop items: 3-5 jokers, balanced prices"""
        items = []
        
        # Generate 3-5 jokers with strategic pricing (max_jokers=5)
        num_jokers = random.choice([2, 3, 4])  # Meno joker disponibili (da 3-5 a 2-4)
        available_jokers = [
            JokerType.JOKER,           # Basic +2 mult sempre - economico (ridotto)
            JokerType.CHIP_JOKER,      # +15 chips sempre - economico (ridotto)
            JokerType.MULT_JOKER,      # +3 mult sempre - medio (ridotto)
            JokerType.BONUS_JOKER,     # +8 chips +2 mult - medio (ridotto)
            JokerType.WILD_JOKER,      # Scaling con carte - scalabile (ridotto)
            JokerType.LUCKY_JOKER,     # RNG potente (ridotto da 10 a 6)
            JokerType.PREMIUM_JOKER    # Most powerful joker - premium (costoso)
        ]
        
        # Remove jokers already owned to encourage variety
        owned_types = {j.joker_type for j in self.jokers}
        available_jokers = [jt for jt in available_jokers if jt not in owned_types]
        
        if not available_jokers:  # If all owned, reset to all types
            available_jokers = list(JokerType)
            
        selected_jokers = random.sample(available_jokers, min(num_jokers, len(available_jokers)))
        
        for joker_type in selected_jokers:
            # Strategic pricing based on power level and ante - PREZZI RIDOTTI MA BILANCIATI
            base_prices = {
                JokerType.JOKER: 3,           # Base +2 mult (ridotto da 4)
                JokerType.CHIP_JOKER: 3,      # +15 chips (ridotto da 4)
                JokerType.MULT_JOKER: 4,      # +3 mult (ridotto da 6)
                JokerType.BONUS_JOKER: 5,     # Doppio effetto (ridotto da 8)
                JokerType.WILD_JOKER: 4,      # Scaling con carte (ridotto da 6)
                JokerType.LUCKY_JOKER: 6,     # RNG potente (ridotto da 10 a 6)
                JokerType.PREMIUM_JOKER: 10   # Most powerful - premium price
            }
            
            # Prices scale less aggressively with ante (more accessible economy)
            price_multiplier = 1.0 + (self.current_ante - 1) * 0.15  # Ridotto da 0.2 a 0.15
            base_price = base_prices.get(joker_type, 4)  # Default ridotto da 5 a 4
            final_price = max(2, int(base_price * price_multiplier))  # Prezzo minimo rimane 2
            
            joker = Joker(joker_type)
            items.append(ShopItem(
                item_type='joker',
                content=joker,
                price=final_price,
                description=f"{joker_type.value} - ${final_price}"
            ))
        
        return items

    def _get_total_blinds_beaten(self) -> int:
        """Get total number of blinds beaten"""
        # Calculate total blinds beaten based on current progression
        total = 0
        
        # Count completed antes (each ante has 3 blinds)
        completed_antes = max(0, self.current_ante - 1)
        total += completed_antes * 3
        
        # Add blinds beaten in current ante
        total += self.current_blind
        
        return total
        return (self.current_ante - 1) * 3 + self.current_blind

    def _advance_blind(self):
        """Advance to the next blind"""
        self.current_blind += 1
        if self.current_blind >= len(self.blinds[self.current_ante]):
            # Completato un ante, passa al successivo
            self.current_blind = 0
            self.current_ante += 1
            
            # CORREZIONE: Se abbiamo superato il massimo ante, termina il gioco
            # Il gioco termina solo DOPO aver completato il max_ante
            if self.current_ante > self.max_ante:
                self.current_ante = self.max_ante
                self.current_blind = len(self.blinds[self.max_ante]) - 1
                self.game_won = True
                return
        
        # CORREZIONE CRITICA: Reset score SOLO per il nuovo blind (non tra le mani dello stesso blind)
        # Il punteggio deve essere cumulativo tra le mani dello stesso blind
        self.score = 0  # Reset per il nuovo blind
        self.hands_left = 4
        self.discards_left = 3
        
        # Imposta il nuovo blind
        self._set_blind()
        
        # Debug disabled for clean training output
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation as a flat array"""
        # For simplicity, return a dummy observation
        return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def render(self, mode="human"):
        """Render the environment (dummy implementation)"""
        pass
    
    def close(self):
        """Close the environment (dummy implementation)"""
        pass