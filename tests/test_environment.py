
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from balatro_env import BalatroEnv, Card, Suit, Rank, PokerEvaluator, HandType, Joker, JokerType

class TestBalatroEnv(unittest.TestCase):
    
    def setUp(self):
        self.env = BalatroEnv(max_ante=2, starting_money=10, hand_size=8, max_jokers=1)
    
    def test_environment_creation(self):
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.max_ante, 2)
        self.assertEqual(self.env.starting_money, 10)
        self.assertEqual(self.env.hand_size, 8)
        self.assertEqual(self.env.max_jokers, 1)
    
    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertEqual(self.env.current_ante, 1)
        self.assertEqual(self.env.current_blind, 0)
        self.assertEqual(self.env.money, 10)
        self.assertEqual(self.env.hands_left, 4)
        self.assertEqual(self.env.discards_left, 3)
        self.assertEqual(len(self.env.hand), 8)
        self.assertEqual(len(self.env.deck), 52 - 8)
        self.assertEqual(len(self.env.jokers), 1)
        self.assertGreater(self.env.chips_needed, 0)
        self.assertEqual(info['ante_reached'], 1)
        self.assertEqual(info['blinds_beaten'], 0)
    
    def test_action_space(self):
        self.assertEqual(self.env.action_space.n, 2**self.env.hand_size + 2**self.env.hand_size)
        self.assertEqual(self.env.action_space.n, 2**8 + 2**8)

    def test_observation_space(self):
        self.assertEqual(self.env.observation_space.shape, (200,))

    def test_play_valid_hand(self):
        self.env.reset()
        initial_hands_left = self.env.hands_left
        initial_score = self.env.score
        
        self.env.hand = [
            Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
            Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
            Card(Suit.SPADES, Rank.ACE), 
            Card(Suit.CLUBS, Rank.TWO), Card(Suit.DIAMONDS, Rank.THREE), 
            Card(Suit.HEARTS, Rank.FOUR)
        ]
        action_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        action = action_mask

        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertEqual(self.env.hands_left, initial_hands_left - 1)
        self.assertGreater(self.env.score, initial_score)
        self.assertEqual(info['action_type'], 'play')
        self.assertGreater(reward, 0)

    def test_play_invalid_hand_size(self):
        self.env.reset()
        initial_hands_left = self.env.hands_left
        
        action_mask = (1 << 0) | (1 << 1) | (1 << 2)
        action = action_mask

        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertEqual(self.env.hands_left, initial_hands_left) 
        self.assertLess(reward, 0)
        self.assertEqual(info['action_type'], 'invalid_play')

    def test_discard_cards(self):
        self.env.reset()
        initial_discards_left = self.env.discards_left
        initial_hand_size = len(self.env.hand)
        
        action_mask = (1 << 0) | (1 << 1) | (1 << 2)
        action = action_mask + (2**self.env.hand_size)

        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertEqual(self.env.discards_left, initial_discards_left - 1)
        self.assertEqual(len(self.env.hand), initial_hand_size) 
        self.assertEqual(info['action_type'], 'discard')
        self.assertGreaterEqual(reward, -0.1)

    def test_advance_blind(self):
        self.env.reset()
        self.env.score = self.env.chips_needed
        initial_ante = self.env.current_ante
        initial_blind = self.env.current_blind
        
        self.env.hand = [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
                         Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
                         Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO), 
                         Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
        action = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertTrue(info['blind_beaten'])
        if initial_blind < 2:
            self.assertEqual(self.env.current_blind, initial_blind + 1)
            self.assertEqual(self.env.current_ante, initial_ante)
        else:
            self.assertEqual(self.env.current_blind, 0)
            self.assertEqual(self.env.current_ante, initial_ante + 1)
        
        self.assertGreater(self.env.money, self.env.starting_money)
        self.assertEqual(self.env.hands_left, 4)
        self.assertEqual(self.env.discards_left, 3)
        self.assertEqual(self.env.score, 0)

    def test_game_win_condition(self):
        self.env = BalatroEnv(max_ante=1, starting_money=10)
        self.env.reset()
        
        self.env.score = self.env.chips_needed
        self.env.hand = [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
                         Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
                         Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO), 
                         Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
        action = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

        obs, reward, done, truncated, info = self.env.step(action)
        self.assertFalse(done)
        self.assertEqual(self.env.current_blind, 1)

        self.env.score = self.env.chips_needed
        self.env.hand = [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
                         Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
                         Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO), 
                         Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertFalse(done)
        self.assertEqual(self.env.current_blind, 2)
        
        self.env.score = self.env.chips_needed
        self.env.hand = [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
                         Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
                         Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO), 
                         Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertTrue(done)
        self.assertTrue(info['won'])
        self.assertGreater(reward, 5.0)

    def test_game_lose_condition(self):
        self.env.reset()
        self.env.hands_left = 1
        self.env.score = 0
        
        self.env.hand = [
            Card(Suit.SPADES, Rank.TWO), Card(Suit.CLUBS, Rank.THREE),
            Card(Suit.DIAMONDS, Rank.FOUR), Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.SPADES, Rank.SEVEN), Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.DIAMONDS, Rank.NINE), Card(Suit.HEARTS, Rank.TEN)
        ]
        action = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

        obs, reward, done, truncated, info = self.env.step(action)
        
        self.assertTrue(done)
        self.assertTrue(info['failed'])
        self.assertLess(reward, -0.5)

    def test_joker_effect(self):
        self.env.reset()
        self.env.jokers = [Joker(JokerType.JOLLY_JOKER)]
        self.env.hands_left = 1
        self.env.score = 0
        
        self.env.hand = [
            Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE),
            Card(Suit.SPADES, Rank.FOUR), Card(Suit.CLUBS, Rank.FIVE),
            Card(Suit.DIAMONDS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN)
        ]
        action_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        action = action_mask

        obs, reward, done, truncated, info = self.env.step(action)
        
        _, base_chips, base_mult = PokerEvaluator.evaluate_hand(self.env.hand[0:5])
        expected_mult_without_joker = base_mult
        expected_chips_without_joker = base_chips
        
        self.assertEqual(info['hand_type'], HandType.PAIR.value)
        self.assertAlmostEqual(info['hand_score'], (expected_chips_without_joker) * (expected_mult_without_joker + 8), places=2)
        self.assertEqual(self.env.score, (expected_chips_without_joker) * (expected_mult_without_joker + 8))


class TestPokerEvaluator(unittest.TestCase):
    
    def test_high_card_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.SIX), Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.TEN)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.HIGH_CARD)
        self.assertEqual(chips, 2+4+6+8+10 + 5)
        self.assertEqual(mult, 1)

    def test_pair_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING), Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.PAIR)
        self.assertEqual(chips, 10+10+10+10+10 + 10)
        self.assertEqual(mult, 2)
    
    def test_two_pair_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING), Card(Suit.CLUBS, Rank.KING),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.TWO_PAIR)
        self.assertEqual(chips, 10+10+10+10+10 + 20)
        self.assertEqual(mult, 2)

    def test_three_of_a_kind_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.THREE_OF_A_KIND)
        self.assertEqual(chips, 10+10+10+10+10 + 30)
        self.assertEqual(mult, 3)

    def test_straight_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.TEN), Card(Suit.SPADES, Rank.JACK),
            Card(Suit.DIAMONDS, Rank.QUEEN), Card(Suit.CLUBS, Rank.KING),
            Card(Suit.HEARTS, Rank.ACE)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.STRAIGHT)
        self.assertEqual(chips, 10+10+10+10+10 + 30)
        self.assertEqual(mult, 4)

    def test_straight_evaluation_ace_low(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.CLUBS, Rank.FOUR),
            Card(Suit.HEARTS, Rank.FIVE)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.STRAIGHT)
        self.assertEqual(chips, 10+2+3+4+5 + 30)
        self.assertEqual(mult, 4)
    
    def test_flush_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.TWO), Card(Suit.HEARTS, Rank.FIVE),
            Card(Suit.HEARTS, Rank.SEVEN), Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.HEARTS, Rank.KING)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FLUSH)
        self.assertEqual(chips, 2+5+7+10+10 + 35)
        self.assertEqual(mult, 4)

    def test_full_house_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.KING),
            Card(Suit.HEARTS, Rank.KING)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FULL_HOUSE)
        self.assertEqual(chips, 10+10+10+10+10 + 40)
        self.assertEqual(mult, 4)

    def test_four_of_a_kind_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FOUR_OF_A_KIND)
        self.assertEqual(chips, 10+10+10+10+2 + 60)
        self.assertEqual(mult, 7)

    def test_straight_flush_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.NINE), Card(Suit.HEARTS, Rank.TEN),
            Card(Suit.HEARTS, Rank.JACK), Card(Suit.HEARTS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.KING)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.STRAIGHT_FLUSH)
        self.assertEqual(chips, 9+10+10+10+10 + 100)
        self.assertEqual(mult, 8)

    def test_royal_flush_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.TEN), Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.HEARTS, Rank.QUEEN), Card(Suit.HEARTS, Rank.KING),
            Card(Suit.HEARTS, Rank.ACE)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.ROYAL_FLUSH)
        self.assertEqual(chips, 10+10+10+10+10 + 100)
        self.assertEqual(mult, 8)
    
    def test_less_than_5_cards_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.TWO)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.HIGH_CARD)
        self.assertEqual(chips, 0)
        self.assertEqual(mult, 0)

        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.HIGH_CARD)
        self.assertEqual(chips, 0)
        self.assertEqual(mult, 0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
