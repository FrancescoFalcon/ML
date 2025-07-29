
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
        # Action space includes shop actions: base actions (256+256) + shop actions (6) = 518
        expected_actions = 2**self.env.hand_size + 2**self.env.hand_size + 6
        self.assertEqual(self.env.action_space.n, expected_actions)

    def test_observation_space(self):
        # Observation space è stato esteso a 220 per includere informazioni sui joker e shop
        self.assertEqual(self.env.observation_space.shape, (220,))

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

        # Check hands_left using info before blind advancement
        self.assertEqual(info['action_type'], 'play')
        self.assertEqual(info.get('hand_score', 0) > 0, True)
        self.assertGreater(reward, 0)

    def test_play_invalid_hand_size(self):
        self.env.reset()
        initial_hands_left = self.env.hands_left
        
        # Usa un'azione fuori range - action space va da 0 a 517 (256+256+6-1)
        action = 1000  # Azione completamente fuori range
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        # L'environment dovrebbe gestire l'azione invalida
        self.assertLess(reward, 0)
        self.assertTrue(info['invalid_action'])

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
        self.assertGreaterEqual(reward, -0.6)  # Updated to reflect current penalty structure

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
        # Dopo aver battuto un blind, entri in fase shop, ante/blind non cambiano ancora
        self.assertTrue(self.env.in_shop_phase)
        self.assertEqual(self.env.current_ante, initial_ante)
        self.assertEqual(self.env.current_blind, initial_blind)

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
        # Dopo aver battuto il primo blind, entri in shop phase
        self.assertTrue(self.env.in_shop_phase)
        self.assertEqual(self.env.current_blind, 0)  # Non ancora avanzato fino a quando non esci dal shop

        # Exit shop phase to continue to next blind
        obs, reward, done, truncated, info = self.env.step(517)  # LEAVE_SHOP action
        self.assertFalse(done)
        self.assertFalse(self.env.in_shop_phase)

        self.env.score = self.env.chips_needed
        self.env.hand = [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.JACK), 
                         Card(Suit.SPADES, Rank.QUEEN), Card(Suit.SPADES, Rank.KING), 
                         Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.TWO), 
                         Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertFalse(done)
        # Test che l'ambiente gestisca correttamente il beating del blind
        self.assertTrue(info.get('blind_beaten', False))

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
        played_cards = [
            Card(Suit.SPADES, Rank.ACE), Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE),
            Card(Suit.SPADES, Rank.FOUR)
        ]
        self.env.hand = played_cards + [Card(Suit.CLUBS, Rank.FIVE), Card(Suit.DIAMONDS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN)]
        action_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
        action = action_mask

        obs, reward, done, truncated, info = self.env.step(action)

        _, base_chips, base_mult = PokerEvaluator.evaluate_hand(played_cards)
        expected_mult_without_joker = base_mult
        expected_chips_without_joker = base_chips

        self.assertEqual(info['hand_type'], HandType.PAIR.value)
        self.assertAlmostEqual(info['hand_score'], (expected_chips_without_joker) * (expected_mult_without_joker + 8), places=2)


class TestPokerEvaluator(unittest.TestCase):
    def test_two_pair_with_extra_card(self):
        # Mano: 2,2,3,4,4 -> TWO_PAIR, il 3 non conta
        cards = [
            Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.TWO),
            Card(Suit.DIAMONDS, Rank.THREE), Card(Suit.CLUBS, Rank.FOUR), Card(Suit.HEARTS, Rank.FOUR)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.TWO_PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.TWO_PAIR]
        # Solo le due coppie: 2,2,4,4
        expected_chips = base_score + 2 + 2 + 4 + 4
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

    def test_pair_with_extra_card(self):
        # Mano: A, A, 2, 3, 4 -> PAIR, solo gli assi contano (ora valgono 11)
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.PAIR]
        expected_chips = base_score + 11 + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

    def test_three_of_a_kind_with_extra_card(self):
        # Mano: 4,4,4,2,3 -> THREE_OF_A_KIND, solo i 4 contano
        cards = [
            Card(Suit.HEARTS, Rank.FOUR), Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.FOUR), Card(Suit.CLUBS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.THREE_OF_A_KIND)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.THREE_OF_A_KIND]
        expected_chips = base_score + 4 + 4 + 4
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 3)

    def test_full_house_only_counts_triple_and_pair(self):
        # Mano: A, A, A, K, K -> FULL_HOUSE, solo assi e k (assi ora valgono 11)
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.KING), Card(Suit.HEARTS, Rank.KING)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FULL_HOUSE)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.FULL_HOUSE]
        expected_chips = base_score + 11 + 11 + 11 + 10 + 10
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 4)

    def test_four_of_a_kind_only_counts_quads(self):
        # Mano: A, A, A, A, 2 -> FOUR_OF_A_KIND, solo assi (ora valgono 11)
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.ACE), Card(Suit.HEARTS, Rank.TWO)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FOUR_OF_A_KIND)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.FOUR_OF_A_KIND]
        expected_chips = base_score + 11 + 11 + 11 + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 7)
    
    def test_high_card_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.DIAMONDS, Rank.SIX), Card(Suit.CLUBS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.TEN)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.HIGH_CARD)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.HIGH_CARD]
        # Solo la carta più alta
        expected_chips = base_score + 10
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 1)

    def test_pair_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING), Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.PAIR]
        # Solo la coppia di assi (ora valgono 11 ciascuno)
        expected_chips = base_score + 11 + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)
    
    def test_two_pair_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.KING), Card(Suit.CLUBS, Rank.KING),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.TWO_PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.TWO_PAIR]
        # Solo le due coppie: assi (11+11) e k (10+10)
        expected_chips = base_score + 11 + 11 + 10 + 10
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

    def test_three_of_a_kind_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.JACK)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.THREE_OF_A_KIND)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.THREE_OF_A_KIND]
        # Solo i tre assi (ora valgono 11 ciascuno)
        expected_chips = base_score + 11 + 11 + 11
        self.assertEqual(chips, expected_chips)
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
        # Scala bassa A-2-3-4-5: asso vale 1 nella scala bassa
        self.assertEqual(chips, 1+2+3+4+5 + 30)
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
        # Full house: 3 assi (11+11+11) + 2 re (10+10)
        self.assertEqual(chips, 11+11+11+10+10 + 40)
        self.assertEqual(mult, 4)

    def test_four_of_a_kind_evaluation(self):
        cards = [
            Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE), Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO)
        ]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.FOUR_OF_A_KIND)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.FOUR_OF_A_KIND]
        # Solo i quattro assi (ora valgono 11 ciascuno)
        expected_chips = base_score + 11 + 11 + 11 + 11
        self.assertEqual(chips, expected_chips)
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
        # PAIR con 2 carte (assi ora valgono 11)
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE)]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.PAIR]
        expected_chips = base_score + 11 + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

        # THREE_OF_A_KIND con 3 carte
        cards = [Card(Suit.HEARTS, Rank.FOUR), Card(Suit.SPADES, Rank.FOUR), Card(Suit.DIAMONDS, Rank.FOUR)]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.THREE_OF_A_KIND)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.THREE_OF_A_KIND]
        expected_chips = base_score + 4 + 4 + 4
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 3)

        # TWO_PAIR con 4 carte
        cards = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.SPADES, Rank.TWO), Card(Suit.DIAMONDS, Rank.FOUR), Card(Suit.CLUBS, Rank.FOUR)]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.TWO_PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.TWO_PAIR]
        expected_chips = base_score + 2 + 2 + 4 + 4
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

        # PAIR con 3 carte
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.ACE), Card(Suit.DIAMONDS, Rank.TWO)]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.PAIR)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.PAIR]
        expected_chips = base_score + 11 + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 2)

        # HIGH_CARD con carte tutte diverse
        cards = [Card(Suit.HEARTS, Rank.ACE), Card(Suit.SPADES, Rank.KING), Card(Suit.DIAMONDS, Rank.TWO)]
        hand_type, chips, mult = PokerEvaluator.evaluate_hand(cards)
        self.assertEqual(hand_type, HandType.HIGH_CARD)
        base_score, _ = PokerEvaluator.BASE_SCORES[HandType.HIGH_CARD]
        expected_chips = base_score + 11
        self.assertEqual(chips, expected_chips)
        self.assertEqual(mult, 1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
