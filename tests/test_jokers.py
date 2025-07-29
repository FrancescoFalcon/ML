"""
Test joker functionality in Balatro environment
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from balatro_env import BalatroEnv, Joker, JokerType, Card, Suit, Rank, HandType
from balatro_env import PokerEvaluator


class TestJokerFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up a fresh environment for each test"""
        self.env = BalatroEnv()
        self.env.reset()

    def test_basic_joker_effect(self):
        """Test that basic JOKER adds +4 mult"""
        # Add a basic joker
        basic_joker = Joker(JokerType.JOKER)
        self.env.jokers.append(basic_joker)
        
        # Create a simple pair hand
        test_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.TWO),
        ]
        
        # Test joker effect calculation
        chips, mult = basic_joker.get_effect(test_hand, HandType.PAIR)
        self.assertEqual(chips, 0, "Basic joker should not add chips")
        self.assertEqual(mult, 4, "Basic joker should add +4 mult")

    def test_greedy_joker_effect(self):
        """Test GREEDY_JOKER effect (+3 mult if no face cards)"""
        greedy_joker = Joker(JokerType.GREEDY_JOKER)
        
        # Test with no face cards (should get bonus)
        no_face_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),
            Card(Suit.CLUBS, Rank.FOUR),
        ]
        chips, mult = greedy_joker.get_effect(no_face_hand, HandType.HIGH_CARD)
        self.assertEqual(mult, 3, "Greedy joker should add +3 mult with no face cards")
        
        # Test with face cards (should get no bonus)
        with_face_hand = [
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.SPADES, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        chips, mult = greedy_joker.get_effect(with_face_hand, HandType.HIGH_CARD)
        self.assertEqual(mult, 0, "Greedy joker should add no mult with face cards")

    def test_jolly_joker_pair_effect(self):
        """Test JOLLY_JOKER special +8 mult for pairs"""
        jolly_joker = Joker(JokerType.JOLLY_JOKER)
        self.env.jokers.append(jolly_joker)
        
        # Set up hand with a pair
        self.env.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
        ]
        
        # Action space uses discrete integers for play actions
        # Binary pattern for selecting first two cards: 11000000 = 3 (bit positions 0 and 1)
        play_action = 3  # Binary 11000000 for selecting first two cards
        
        obs, reward, done, truncated, info = self.env.step(play_action)
        
        # Check that the JOLLY_JOKER effect was triggered
        self.assertEqual(info.get('joker_effect'), 'jolly_joker_pair', 
                        "JOLLY_JOKER should trigger special effect for pairs")

    def test_lusty_joker_flush_effect(self):
        """Test LUSTY_JOKER effect (+3 mult for flush/single suit)"""
        lusty_joker = Joker(JokerType.LUSTY_JOKER)
        
        # Test with all same suit
        flush_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.HEARTS, Rank.FOUR),
            Card(Suit.HEARTS, Rank.SIX),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.TEN),
        ]
        chips, mult = lusty_joker.get_effect(flush_hand, HandType.FLUSH)
        self.assertEqual(mult, 3, "Lusty joker should add +3 mult for flush")
        
        # Test with mixed suits
        mixed_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.CLUBS, Rank.SIX),
        ]
        chips, mult = lusty_joker.get_effect(mixed_hand, HandType.HIGH_CARD)
        self.assertEqual(mult, 0, "Lusty joker should add no mult for mixed suits")

    def test_wrathful_joker_same_rank_effect(self):
        """Test WRATHFUL_JOKER effect (+3 mult for all same rank)"""
        wrathful_joker = Joker(JokerType.WRATHFUL_JOKER)
        
        # Test with all same rank (should be impossible with normal hand, but test logic)
        same_rank_hand = [
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.SPADES, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.SEVEN),
            Card(Suit.DIAMONDS, Rank.SEVEN),
        ]
        chips, mult = wrathful_joker.get_effect(same_rank_hand, HandType.FOUR_OF_A_KIND)
        self.assertEqual(mult, 3, "Wrathful joker should add +3 mult for all same rank")
        
        # Test with mixed ranks
        mixed_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.THREE),
        ]
        chips, mult = wrathful_joker.get_effect(mixed_hand, HandType.PAIR)
        self.assertEqual(mult, 0, "Wrathful joker should add no mult for mixed ranks")

    def test_gluttonous_joker_constant_effect(self):
        """Test GLUTTONOUS_JOKER effect (always +3 mult)"""
        gluttonous_joker = Joker(JokerType.GLUTTONOUS_JOKER)
        
        # Test with any hand
        test_hand = [
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.SPADES, Rank.KING),
        ]
        chips, mult = gluttonous_joker.get_effect(test_hand, HandType.HIGH_CARD)
        self.assertEqual(mult, 3, "Gluttonous joker should always add +3 mult")

    def test_multiple_jokers_stacking(self):
        """Test that multiple jokers stack their effects properly"""
        # Add multiple jokers
        basic_joker = Joker(JokerType.JOKER)  # +4 mult
        gluttonous_joker = Joker(JokerType.GLUTTONOUS_JOKER)  # +3 mult
        self.env.jokers = [basic_joker, gluttonous_joker]
        
        # Set up a simple hand
        self.env.hand = [
            Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.SPADES, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.DIAMONDS, Rank.JACK),
            Card(Suit.HEARTS, Rank.KING),
        ]
        
        # Play first card only (action 1 = binary 00000001)
        play_action = 1  
        
        obs, reward, done, truncated, info = self.env.step(play_action)
        
        # Both jokers should contribute: base mult (1) + basic joker (4) + gluttonous (3) = 8
        expected_mult = 8
        self.assertEqual(info.get('mult'), expected_mult, 
                        f"Multiple jokers should stack: expected {expected_mult}, got {info.get('mult')}")

    def test_joker_purchase_system(self):
        """Test buying jokers from shop"""
        # Enter shop phase
        self.env.in_shop_phase = True
        self.env.shop_items = self.env._generate_shop_items()
        
        # Check if shop has jokers
        joker_items = [i for i, item in enumerate(self.env.shop_items) if item.item_type == "joker"]
        
        if joker_items:
            slot = joker_items[0]
            item = self.env.shop_items[slot]
            initial_money = self.env.money
            initial_jokers = len(self.env.jokers)
            
            # Give enough money to buy
            self.env.money = max(self.env.money, item.price + 5)
            
            # Shop actions start after base actions
            base_actions = 2**self.env.hand_size + 2**self.env.hand_size
            buy_action = base_actions + slot  # Buy action for this slot
            
            obs, reward, done, truncated, info = self.env.step(buy_action)
            
            # Verify purchase
            self.assertEqual(len(self.env.jokers), initial_jokers + 1, 
                           "Should have one more joker after purchase")
            self.assertEqual(info.get('action_type'), 'buy_item', 
                           "Action should be recognized as buy_item")
            self.assertEqual(info.get('item_type'), 'joker', 
                           "Should recognize item as joker")

    def test_joker_space_limit(self):
        """Test that joker purchase respects max_jokers limit"""
        # Environment starts with 1 joker after reset, so fill remaining slots
        while len(self.env.jokers) < self.env.max_jokers:
            self.env.jokers.append(Joker(JokerType.JOKER))
        
        # Enter shop phase
        self.env.in_shop_phase = True
        self.env.shop_items = self.env._generate_shop_items()
        self.env.money = 100  # Plenty of money
        
        # Find a joker in shop
        joker_items = [i for i, item in enumerate(self.env.shop_items) if item.item_type == "joker"]
        
        if joker_items:
            slot = joker_items[0]
            
            # Shop actions start after base actions
            base_actions = 2**self.env.hand_size + 2**self.env.hand_size
            buy_action = base_actions + slot
            
            obs, reward, done, truncated, info = self.env.step(buy_action)
            
            # Should fail
            self.assertEqual(info.get('action_type'), 'invalid_buy_no_joker_space',
                           "Should reject purchase when joker slots are full")
            self.assertEqual(len(self.env.jokers), self.env.max_jokers,
                           "Joker count should remain at max")

    def test_joker_bonus_attributes(self):
        """Test jokers with bonus chips and mult attributes"""
        # Create joker with bonus attributes
        enhanced_joker = Joker(JokerType.JOKER, level=2, bonus_chips=10, bonus_mult=2)
        
        test_hand = [Card(Suit.HEARTS, Rank.ACE)]
        chips, mult = enhanced_joker.get_effect(test_hand, HandType.HIGH_CARD)
        
        # Should get base effect (4 mult) + bonus (10 chips, 2 mult)
        expected_chips = 10
        expected_mult = 4 + 2  # base + bonus
        
        self.assertEqual(chips, expected_chips, f"Expected {expected_chips} chips, got {chips}")
        self.assertEqual(mult, expected_mult, f"Expected {expected_mult} mult, got {mult}")

    def test_joker_effects_in_gameplay(self):
        """Integration test: jokers affecting actual gameplay scoring"""
        # Clear existing jokers and add specific ones
        self.env.jokers = [Joker(JokerType.JOKER)]  # +4 mult
        
        # Set up a known hand
        self.env.hand = [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.DIAMONDS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK),
        ]
        
        # Play the pair of aces (action 3 = binary 11000000)
        play_action = 3  
        
        obs, reward, done, truncated, info = self.env.step(play_action)
        
        # Verify joker contributed to scoring
        base_pair_mult = 2  # Base mult for pair
        joker_mult = 4     # Basic joker adds +4
        expected_mult = base_pair_mult + joker_mult
        
        self.assertEqual(info.get('mult'), expected_mult,
                        f"Expected mult {expected_mult} (base {base_pair_mult} + joker {joker_mult}), got {info.get('mult')}")
        
        # Verify hand was scored correctly
        # ACE pair: base pair chips (10) + 2 ace chips (11 each) = 32 total chips
        base_pair_chips = 32  # Corrected: 10 base + 11 + 11 for ace pair
        expected_score = base_pair_chips * expected_mult
        self.assertEqual(info.get('hand_score'), expected_score,
                        f"Expected score {expected_score}, got {info.get('hand_score')}")


if __name__ == '__main__':
    unittest.main()
