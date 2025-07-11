"""
Balatro RL Project Setup
========================

Setup script and requirements for the Balatro RL project.
"""

import os
import subprocess
import sys

# =============================================================================
# REQUIREMENTS
# =============================================================================

REQUIREMENTS = [
    "stable-baselines3[extra]>=2.0.0",
    "gymnasium>=0.29.0", # Changed from gym to gymnasium
    "torch>=1.13.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "wandb>=0.13.0",
    "tensorboard>=2.10.0",
    "opencv-python>=4.6.0", # Not strictly required for the core RL, but good to have for image processing
    "tqdm>=4.64.0",
    "seaborn>=0.12.0",
    "pandas>=1.5.0",
    "scipy>=1.9.0",
    "plotly>=5.11.0",
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
    "pyyaml>=6.0" # Added for YAML config files
]

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

PROJECT_STRUCTURE = {
    "src/": [
        "balatro_env.py",
        "training.py", 
        "evaluation.py", # This file might be merged into training.py or kept separate
        "utils.py",
        "models/", # For internal models (e.g., card embeddings if used)
        "callbacks/" # New directory for callbacks
    ],
    "data/": [
        "logs/",
        "models/",
        "results/",
        "plots/"
    ],
    "notebooks/": [
        "exploration.ipynb",
        "analysis.ipynb",
        "visualization.ipynb"
    ],
    "configs/": [
        "training_config.yaml",
        "model_config.yaml"
    ],
    "tests/": [
        "test_environment.py",
        "test_training.py" # Placeholder for future training tests
    ]
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def create_project_structure():
    """Create the project directory structure"""
    print("Creating project structure...")
    
    for directory, files in PROJECT_STRUCTURE.items():
        # Create directory
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
        
        # Create files/subdirectories
        for file_or_dir in files:
            full_path = os.path.join(directory, file_or_dir)
            if file_or_dir.endswith('/'):
                # It's a subdirectory
                os.makedirs(full_path, exist_ok=True)
                print(f"Created subdirectory: {full_path}")
            else:
                # It's a file
                if not os.path.exists(full_path):
                    with open(full_path, 'w') as f:
                        f.write(f"# {file_or_dir}\n")
                    print(f"Created file: {full_path}")

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    
    # Create requirements.txt
    requirements_file = 'requirements.txt'
    with open(requirements_file, 'w') as f:
        for req in REQUIREMENTS:
            f.write(f"{req}\n")
    
    # Install packages
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print(f"Please install manually: pip install -r {requirements_file}")

def create_config_files():
    """Create configuration files"""
    print("Creating configuration files...")
    
    # Training configuration
    training_config = """
# Training Configuration for Balatro RL
# =====================================

environment:
  max_ante: 8
  starting_money: 4
  deck_size: 52 # Not directly used by env but for info
  hand_size: 8 # Max cards in hand
  max_jokers: 5 # Max jokers player can have

training:
  algorithm: "PPO"
  total_timesteps: 500000
  learning_rate: 3e-4
  batch_size: 64
  n_steps: 2048 # Number of steps to run for each environment per update
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  n_envs: 8 # Number of parallel environments

evaluation:
  eval_freq: 10000 # Evaluate every N timesteps
  n_eval_episodes: 50
  deterministic: true

logging:
  log_dir: "./data/logs/"
  model_dir: "./data/models/"
  use_wandb: true
  wandb_project: "balatro-rl"
  wandb_entity: null # Replace with your wandb entity if applicable
  save_freq: 50000 # How often to save the model (for wandb)

curriculum:
  enabled: true
  stages:
    - max_ante: 2
      timesteps: 100000
    - max_ante: 4
      timesteps: 150000
    - max_ante: 6
      timesteps: 200000
    - max_ante: 8
      timesteps: 250000
"""
    
    with open('configs/training_config.yaml', 'w') as f:
        f.write(training_config)
    print("Created configs/training_config.yaml")
    
    # Model configuration
    model_config = """
# Model Configuration for Balatro RL
# ==================================

network:
  policy_type: "MlpPolicy"
  net_arch:
    - 512
    - 256
    - 128
  activation_fn: "relu" # Options: relu, tanh, gelu
  ortho_init: true
  
features:
  normalize_obs: true
  normalize_reward: true
  clip_obs: 10.0
  
advanced:
  use_sde: false # Use Stochastic Differential Equations for exploration
  sde_sample_freq: -1
  target_kl: null # If not null, clip the KL divergence between old and new policy

reward_function: # These are illustrative for a more complex reward, not directly used by SB3 PPO
  weights:
    blind_completion: 1.0
    ante_progression: 2.0
    hand_score: 0.1
    efficiency: 0.5 # Reward for fewer hands/discards
    survival: 0.2
"""
    
    with open('configs/model_config.yaml', 'w') as f:
        f.write(model_config)
    print("Created configs/model_config.yaml")

def create_analysis_notebook():
    """Create Jupyter notebook for analysis"""
    print("Creating notebooks/analysis.ipynb")
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Balatro RL Analysis\\n",
    "\\n",
    "This notebook provides analysis and visualization of the Balatro RL training results."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from stable_baselines3 import PPO\\n",
    "import sys\\n",
    "import os\\n",
    "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src')))\\n",
    "\\n",
    "from balatro_env import BalatroEnv\\n",
    "from utils import load_results, BalatroMetrics, plot_training_progress, create_performance_report, analyze_hand_preferences\\n",
    "\\n",
    "# Set style\\n",
    "plt.style.use('seaborn-v0_8-darkgrid') # Updated style\\n",
    "sns.set_palette('deep')\\n",
    "\\n",
    "print(\"Libraries loaded and paths configured.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Analyze Training Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Load logs from Monitor wrapper (if using direct log files, not WandB)\\n",
    "# You would typically use tensorboard or wandb UI for live monitoring.\\n",
    "# For analysis after training, you might load saved evaluation results.\\n",
    "\\n",
    "# Assuming evaluation results are saved to 'data/results/evaluation_results.pkl'\\n",
    "results_path = '../data/results/evaluation_results.pkl'\\n",
    "metrics_obj = BalatroMetrics()\\n",
    "\\n",
    "try:\\n",
    "    # In a real scenario, you'd populate BalatroMetrics from raw log data or a structured eval output\\n",
    "    # For simplicity here, if evaluate_agent saves a dict, you'd convert it.\\n",
    "    # Let's assume 'evaluate_agent' from training.py was run and its output saved.\\n",
    "    # The evaluate_agent function already uses BalatroMetrics internally.\\n",
    "    # If you want to load raw logs from monitor, you would use: \\n",
    "    # from stable_baselines3.common.results_plotter import load_results as sb3_load_results\\n",
    "    # log_dir = '../data/logs/' # Or the specific stage log dir\\n",
    "    # agent_results = sb3_load_results(log_dir)\\n",
    "    # print(agent_results.head())\\n",
    "    \\n",
    "    # For now, let's just create some dummy data or load a placeholder\\n",
    "    # In practice, you'd save the metrics_tracker object directly if you wanted to reload it.\\n",
    "    # As 'evaluate_agent' saves plots and reports, you'd look at those files.\\n",
    "    \\n",
    "    print(\"To analyze, ensure you have run 'python src/training.py --mode eval' first.\\n\")\\n",
    "    print(\"Check 'evaluation_progress.png', 'evaluation_distributions.png', and 'evaluation_report.md' in your project root or data/plots.\")\\n",
    "    \\n",
    "    # If you saved a raw BalatroMetrics object:\\n",
    "    # metrics_obj = load_results(results_path) # Needs to be saved as pickle of BalatroMetrics object\\n",
    "    # stats = metrics_obj.get_statistics()\\n",
    "    # print(\"Loaded Evaluation Statistics:\")\\n",
    "    # for k, v in stats.items():\\n",
    "    #     print(f\"- {k}: {v}\")\\n",
    "    \\n",
    "except FileNotFoundError:\\n",
    "    print(f\"Warning: Results file not found at {results_path}. Run evaluation first.\")\\n",
    "    # Create dummy data for demonstration if file not found\\n",
    "    metrics_obj.episode_rewards = np.random.rand(100) * 100 - 20 # Example rewards\\n",
    "    metrics_obj.antes_reached = np.random.randint(1, 9, 100) # Example antes\\n",
    "    metrics_obj.blinds_beaten = np.random.randint(0, 24, 100) # Example blinds\\n",
    "    metrics_obj.scores_achieved = np.random.rand(100) * 1000 # Example scores\\n",
    "    metrics_obj.hand_types_played = np.random.choice(['high_card', 'pair', 'flush'], 200).tolist()\\n",
    "    metrics_obj.jokers_active_per_episode = [['joker'], ['jolly_joker'], ['greedy_joker']] * 33 # Example jokers\\n",
    "    print(\"Using dummy data for analysis.\")\\n",
    "    \\n",
    "stats = metrics_obj.get_statistics()\\n",
    "print(\"\\n--- Overall Statistics ---\")\\n",
    "for k, v in stats.items():\\n",
    "    print(f\"- {k}: {v}\")\\n",
    "\\n",
    "if metrics_obj.episode_rewards:\\n",
    "    plot_training_progress(metrics_obj.episode_rewards, metrics_obj.antes_reached, metrics_obj.blinds_beaten, save_path=None) # Display in notebook\\n",
    "    metrics_obj.plot_distributions(save_path=None) # Display in notebook\\n",
    "    analyze_hand_preferences(metrics_obj.hand_types_played, save_path=None) # Display in notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model Performance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load trained model (ensure it's saved from training script)\\n",
    "model_path = '../data/models/final_model' # Or path to a specific stage model\\n",
    "normalize_path = '../data/models/vecnormalize.pkl'\\n",
    "\\n",
    "try:\\n",
    "    model = PPO.load(model_path)\\n",
    "    print(f\"Model loaded from {model_path}\")\\n",
    "    \\n",
    "    # To inspect model policy architecture\\n",
    "    print(\"\\nModel Policy Network:\\n\", model.policy)\\n",
    "    \\n",
    "    # You can also run a single evaluation episode directly here if needed\\n",
    "    # For comprehensive evaluation, use the 'eval' mode of training.py\\n",
    "    # env = BalatroEnv()\\n",
    "    # if os.path.exists(normalize_path):\\n",
    "    #    from stable_baselines3.common.vec_env import VecNormalize\\n",
    "    #    env = Monitor(env, 'temp_log')\\n",
    "    #    env = make_vec_env(lambda: env, n_envs=1)\\n",
    "    #    env = VecNormalize.load(normalize_path, env)\\n",
    "    #    env.training = False\\n",
    "    #    env.norm_reward = False\\n",
    "    # obs, _ = env.reset()\\n",
    "    # for _ in range(100):\\n",
    "    #    action, _states = model.predict(obs, deterministic=True)\\n",
    "    #    obs, reward, done, truncated, info = env.step(action)\\n",
    "    #    env.render()\\n",
    "    #    if done or truncated:\\n",
    "    #        break\\n",
    "    \\n",
    "except Exception as e:\\n",
    "    print(f\"Error loading model: {e}. Ensure the model has been trained and saved correctly.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Strategy Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# This section would involve deeper analysis of agent behavior.\\n",
    "# E.g., analyzing decision points like when to discard vs. play, which hands are preferred.\\n",
    "# This requires saving more granular data during evaluation or logging specific actions.\\n",
    "\\n",
    "# Example: Plotting hand type distribution from BalatroMetrics\\n",
    "if metrics_obj.hand_types_played:\\n",
    "    print(\"\\n--- Hand Type Play Frequency ---\")\\n",
    "    print(metrics_obj._get_most_common(metrics_obj.hand_types_played))\\n",
    "\\n",
    "# You could extend BalatroMetrics or add functions to `utils.py` to log and analyze:\\n",
    "# - Number of discards per blind\\n",
    "# - Money spent in shop (if shop is implemented in env)\\n",
    "# - Value of cards discarded vs. played\\n",
    "# - Correlation between joker types and performance\\n",
    "\\n",
    "print(\"\\nStrategy analysis requires more detailed logging within the environment or custom callbacks.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('notebooks/analysis.ipynb', 'w') as f:
        f.write(notebook_content)
    print("Created notebooks/analysis.ipynb")

def create_test_files():
    """Create test files"""
    print("Creating test files...")
    
    # test_environment.py (already provided updated content above)
    test_env_content = """
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
"""
    
    with open('tests/test_environment.py', 'w') as f:
        f.write(test_env_content)
    print("Created tests/test_environment.py")

    # test_training.py (placeholder)
    test_training_content = '''
"""
Tests for Balatro training
"""

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# No specific tests implemented yet, but structure is here
class TestTraining(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
'''
    with open('tests/test_training.py', 'w') as f:
        f.write(test_training_content)
    print("Created tests/test_training.py")


# =============================================================================
# MAIN SETUP FUNCTION
# =============================================================================

def setup_project():
    """Main setup function"""
    print("Setting up Balatro RL project...")
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    install_requirements()
    
    # Create configuration files
    create_config_files()
    
    # Create utility files
    # The content of utils.py, balatro_env.py, training.py, and balatro_callbacks.py
    # are generated directly below or are expected to be updated/pasted.
    # The setup script will create empty files, then you'd paste the content.
    # For a real setup script, you might embed the content or copy from templates.
    create_utils_file() # This function is defined below in the original setup script
    create_balatro_env_file() # New helper function
    create_training_file() # New helper function
    create_balatro_callbacks_file() # New helper function

    # Create analysis notebook
    create_analysis_notebook()
    
    # Create test files
    create_test_files()
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Run: python -m unittest discover tests  # Run all tests to verify setup")
    print("2. Run: python src/training.py --mode train  # Start training")
    print("3. Run: python src/training.py --mode eval --model_path ./data/models/final_model --normalize_path ./data/models/vecnormalize.pkl  # Evaluate model")
    print("4. Open: notebooks/analysis.ipynb  # Analyze results")
    print("\nProject structure created successfully!")


# Helper functions to write the detailed file contents during setup.
# In a real distribution, these might be loaded from template files.
# For this exercise, we'll embed them.

def create_utils_file():
    # Content of src/utils.py (pasted directly)
    utils_content = """
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import pickle
import json
import yaml
from collections import Counter
import os

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def load_results(filepath: str) -> Dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_training_progress(rewards: List[float], 
                         antes: List[int], 
                         blinds: List[int],
                         save_path: Optional[str] = None):
    if not rewards:
        print("No rewards data to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    window = min(100, len(rewards) // 10 if len(rewards) >= 10 else len(rewards))
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
    else:
        axes[0, 1].set_visible(False)
    
    axes[1, 0].plot(antes)
    axes[1, 0].set_title('Antes Reached')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Ante')
    axes[1, 0].set_yticks(range(1, max(antes + [2])))

    axes[1, 1].plot(blinds)
    axes[1, 1].set_title('Total Blinds Beaten')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Blinds')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training progress plot to {save_path}")

def analyze_hand_preferences(hand_types: List[str], save_path: Optional[str] = None):
    if not hand_types:
        print("No hand type data to analyze.")
        return

    hand_counts = Counter(hand_types)
    labels = hand_counts.keys()
    sizes = hand_counts.values()

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    plt.axis('equal')
    plt.title('Agent Hand Type Preferences')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved hand preferences plot to {save_path}")

def create_performance_report(results: Dict, save_path: str):
    report = f'''
# Balatro RL Performance Report

## Overall Performance
- **Win Rate**: {results.get('win_rate', 0):.1%}
- **Average Reward**: {results.get('avg_reward', 0):.2f}
- **Average Ante Reached**: {results.get('avg_ante', 0):.2f}
- **Average Blinds Beaten**: {results.get('avg_blinds', 0):.2f}

## Training Statistics
- **Total Episodes**: {results.get('total_episodes', 'N/A')}
- **Total Timesteps**: {results.get('total_timesteps', 'N/A')}
- **Training Time**: {results.get('training_time', 'N/A')}

## Model Configuration (Extracted from training run or config)
- **Algorithm**: {results.get('algorithm', 'PPO')}
- **Learning Rate**: {results.get('learning_rate', 'N/A')}
- **Batch Size**: {results.get('batch_size', 'N/A')}
'''
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"Created performance report at {save_path}")

class BalatroMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_rewards = []
        self.antes_reached = []
        self.blinds_beaten = []
        self.hand_types_played = []
        self.jokers_active_per_episode = []
        self.scores_achieved = []
    
    def add_episode(self, reward: float, ante: int, blinds: int, 
                   hand_types: List[str], jokers: List[str], score: int):
        self.episode_rewards.append(reward)
        self.antes_reached.append(ante)
        self.blinds_beaten.append(blinds)
        self.hand_types_played.extend(hand_types)
        self.jokers_active_per_episode.append(jokers)
        self.scores_achieved.append(score)
    
    def get_statistics(self) -> Dict:
        if not self.episode_rewards:
            return {}
        
        all_jokers_flat = [joker for sublist in self.jokers_active_per_episode for joker in sublist]

        return {
            'win_rate': np.mean([1 if ante >= 8 else 0 for ante in self.antes_reached]),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_ante': np.mean(self.antes_reached),
            'avg_blinds': np.mean(self.blinds_beaten),
            'avg_score': np.mean(self.scores_achieved),
            'reward_std': np.std(self.episode_rewards),
            'most_common_hands': self._get_most_common(self.hand_types_played),
            'most_common_jokers': self._get_most_common(all_jokers_flat),
            'progression_rate': np.mean(self.antes_reached) / 8.0
        }
    
    def _get_most_common(self, items: List[Any], top_k: int = 5) -> List[Tuple[Any, int]]:
        return Counter(items).most_common(top_k)
    
    def plot_distributions(self, save_path: Optional[str] = None):
        if not self.episode_rewards:
            print("No data to plot distributions.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(self.episode_rewards, bins=30, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Episode Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        
        sns.histplot(self.antes_reached, bins=range(1, max(self.antes_reached + [2]) + 2), kde=False, discrete=True, ax=axes[0, 1])
        axes[0, 1].set_title('Ante Reached Distribution')
        axes[0, 1].set_xlabel('Ante')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xticks(range(1, max(self.antes_reached + [2]) + 1))
        
        sns.histplot(self.blinds_beaten, bins=30, kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Blinds Beaten Distribution')
        axes[1, 0].set_xlabel('Total Blinds Beaten')
        axes[1, 0].set_ylabel('Frequency')
        
        sns.histplot(self.scores_achieved, bins=30, kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Score Distribution (Last Blind)')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved distribution plots to {save_path}")
"""
    with open('src/utils.py', 'w') as f:
        f.write(utils_content)
    print("Created src/utils.py")

def create_balatro_env_file():
    # Content of src/balatro_env.py (pasted directly)
    balatro_env_content = """
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
        print(f"\\n--- Balatro Game State ---")
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
"""
    with open('src/balatro_env.py', 'w') as f:
        f.write(balatro_env_content)
    print("Created src/balatro_env.py")

def create_training_file():
    # Content of src/training.py (pasted directly)
    training_content = """
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.balatro_env import BalatroEnv
from src.utils import load_config, BalatroMetrics, plot_training_progress, create_performance_report
from src.callbacks.balatro_callbacks import BalatroCallback

class TrainingConfig:
    def __init__(self, config_path="configs/training_config.yaml"):
        config = load_config(config_path)
        
        self.n_envs = config['training']['n_envs']
        self.max_ante = config['environment']['max_ante']
        self.starting_money = config['environment']['starting_money']
        self.hand_size = config['environment'].get('hand_size', 8)
        self.max_jokers = config['environment'].get('max_jokers', 5)
        
        self.algorithm = config['training']['algorithm']
        self.total_timesteps = config['training']['total_timesteps']
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.n_steps = config['training']['n_steps']
        self.gamma = config['training']['gamma']
        self.gae_lambda = config['training']['gae_lambda']
        self.clip_range = config['training']['clip_range']
        self.ent_coef = config['training']['ent_coef']
        self.vf_coef = config['training']['vf_coef']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        self.eval_freq = config['evaluation']['eval_freq']
        self.n_eval_episodes = config['evaluation']['n_eval_episodes']
        self.deterministic_eval = config['evaluation']['deterministic']
        
        self.log_dir = config['logging']['log_dir']
        self.model_dir = config['logging']['model_dir']
        self.use_wandb = config['logging']['use_wandb']
        self.wandb_project = config['logging']['wandb_project']
        self.wandb_entity = config['logging'].get('wandb_entity', None)
        self.save_freq = config['logging'].get('save_freq', 50000)

        self.curriculum_enabled = config['curriculum']['enabled']
        self.curriculum_stages = config['curriculum']['stages']


class ModelConfig:
    def __init__(self, config_path="configs/model_config.yaml"):
        config = load_config(config_path)

        self.policy_type = config['network']['policy_type']
        self.net_arch = config['network']['net_arch']
        self.activation_fn = config['network']['activation_fn']
        self.ortho_init = config['network']['ortho_init']

        self.normalize_obs = config['features']['normalize_obs']
        self.normalize_reward = config['features']['normalize_reward']
        self.clip_obs = config['features']['clip_obs']

        self.use_sde = config['advanced']['use_sde']
        self.sde_sample_freq = config['advanced']['sde_sample_freq']
        self.target_kl = config['advanced']['target_kl']

        self.reward_weights = config['reward_function']['weights']

def create_env(config: TrainingConfig, eval_env: bool = False):
    def _init():
        env_max_ante = config.max_ante if not eval_env else 8
        env = BalatroEnv(max_ante=env_max_ante,
                         starting_money=config.starting_money,
                         hand_size=config.hand_size,
                         max_jokers=config.max_jokers)
        env = Monitor(env, config.log_dir)
        return env
    
    return _init

def setup_wandb(config: TrainingConfig):
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=config.__dict__,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            name=f"{config.algorithm}_{config.total_timesteps}"
        )

def train_agent(config: TrainingConfig, model_config: ModelConfig, previous_model=None, previous_vec_env=None):
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    if wandb.run is None:
        setup_wandb(config)
    
    env = make_vec_env(
        create_env(config),
        n_envs=config.n_envs,
        seed=42
    )
    
    if previous_vec_env:
        env = VecNormalize.load(previous_vec_env, env)
        print("Loaded previous VecNormalize statistics.")
    else:
        env = VecNormalize(env, norm_obs=model_config.normalize_obs,
                           norm_reward=model_config.normalize_reward,
                           clip_obs=model_config.clip_obs)
    
    eval_env = make_vec_env(
        create_env(config, eval_env=True),
        n_envs=1,
        seed=123
    )
    eval_env = VecNormalize(eval_env, norm_obs=model_config.normalize_obs,
                            norm_reward=model_config.normalize_reward,
                            clip_obs=model_config.clip_obs)
    
    if previous_model:
        model = previous_model
        model.set_env(env)
        print("Continuing training with loaded model.")
    else:
        policy_kwargs = dict(
            net_arch=model_config.net_arch
        )
        if model_config.activation_fn == 'relu':
            policy_kwargs['activation_fn'] = getattr(__import__('torch.nn', fromlist=['ReLU']), 'ReLU')
        elif model_config.activation_fn == 'tanh':
            policy_kwargs['activation_fn'] = getattr(__import__('torch.nn', fromlist=['Tanh']), 'Tanh')

        model = PPO(
            model_config.policy_type,
            env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=1,
            tensorboard_log=config.log_dir,
            policy_kwargs=policy_kwargs,
            use_sde=model_config.use_sde,
            sde_sample_freq=model_config.sde_sample_freq,
            target_kl=model_config.target_kl
        )
    
    callbacks = [BalatroCallback()]
    
    if config.use_wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=config.save_freq,
            model_save_path=f"{config.model_dir}/wandb_model",
            verbose=2,
            save_replay_buffer=True,
            save_vecnormalize=True
        ))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.model_dir, "best_model"),
        log_path=config.log_dir,
        eval_freq=max(config.eval_freq // config.n_envs, 1),
        n_eval_episodes=config.n_eval_episodes,
        deterministic=config.deterministic_eval,
        render=False
    )
    callbacks.append(eval_callback)
    
    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO_Balatro_Run"
    )
    
    model.save(f"{config.model_dir}/final_model")
    env.save(f"{config.model_dir}/vecnormalize.pkl")
    
    print("Training completed!")
    return model, env

def evaluate_agent(model_path: str, n_episodes: int = 100, normalize_path: Optional[str] = None):
    model = PPO.load(model_path)
    
    eval_config = TrainingConfig()
    env = BalatroEnv(max_ante=8, starting_money=eval_config.starting_money)
    
    if normalize_path and os.path.exists(normalize_path):
        env = Monitor(env, "./temp_eval_log")
        env = make_vec_env(lambda: env, n_envs=1)
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = Monitor(env, "./temp_eval_log")
        env = make_vec_env(lambda: env, n_envs=1)

    metrics_tracker = BalatroMetrics()
    
    print(f"\\nStarting evaluation for {n_episodes} episodes...")
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        episode_hand_types = []
        episode_jokers = [j.joker_type.value for j in env.unwrapped.envs[0].jokers]

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward[0]
            
            current_info = info[0] if isinstance(info, tuple) else info
            
            if current_info.get('action_type') == 'play' and 'hand_type' in current_info:
                episode_hand_types.append(current_info['hand_type'])

        metrics_tracker.add_episode(
            reward=episode_reward,
            ante=env.unwrapped.envs[0].current_ante,
            blinds=env.unwrapped.envs[0]._get_total_blinds_beaten(),
            hand_types=episode_hand_types,
            jokers=episode_jokers,
            score=env.unwrapped.envs[0].score
        )
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Ante={env.unwrapped.envs[0].current_ante}, Blinds Beaten={env.unwrapped.envs[0]._get_total_blinds_beaten()}")
    
    stats = metrics_tracker.get_statistics()
    
    print(f"\\nEvaluation Results ({n_episodes} episodes):")
    print(f"Win Rate: {stats.get('win_rate', 0):.1%}")
    print(f"Average Reward: {stats.get('avg_reward', 0):.2f}")
    print(f"Average Ante Reached: {stats.get('avg_ante', 0):.2f}")
    print(f"Average Blinds Beaten: {stats.get('avg_blinds', 0):.2f}")
    print(f"Average Score: {stats.get('avg_score', 0):.2f}")
    print(f"Most Common Hands: {stats.get('most_common_hands', [])}")

    plot_training_progress(metrics_tracker.episode_rewards,
                           metrics_tracker.antes_reached,
                           metrics_tracker.blinds_beaten,
                           save_path="./data/plots/evaluation_progress.png")
    
    metrics_tracker.plot_distributions(save_path="./data/plots/evaluation_distributions.png")

    report_data = {
        'win_rate': stats.get('win_rate', 0),
        'avg_reward': stats.get('avg_reward', 0),
        'avg_ante': stats.get('avg_ante', 0),
        'avg_blinds': stats.get('avg_blinds', 0),
        'total_episodes': n_episodes,
        'algorithm': model.__class__.__name__,
        'learning_rate': model.lr_schedule(1),
        'batch_size': model.batch_size
    }
    create_performance_report(report_data, "./data/results/evaluation_report.md")

    return stats

def curriculum_training():
    main_config = TrainingConfig()
    model_config = ModelConfig()

    if not main_config.curriculum_enabled:
        print("Curriculum learning is disabled in config. Running single training session.")
        train_agent(main_config, model_config)
        return
        
    stages = main_config.curriculum_stages
    
    model = None
    vec_env_stats_path = None
    
    for i, stage in enumerate(stages):
        print(f"\\n=== Curriculum Stage {i+1}: Max Ante {stage['max_ante']} ({stage['timesteps']} timesteps) ===")
        
        stage_config = TrainingConfig()
        stage_config.max_ante = stage['max_ante']
        stage_config.total_timesteps = stage['timesteps']
        stage_config.wandb_project = f"{main_config.wandb_project}-stage-{i+1}"
        stage_config.log_dir = os.path.join(main_config.log_dir, f"stage_{i+1}")
        stage_config.model_dir = os.path.join(main_config.model_dir, f"stage_{i+1}")
        os.makedirs(stage_config.log_dir, exist_ok=True)
        os.makedirs(stage_config.model_dir, exist_ok=True)

        if main_config.use_wandb:
            wandb.finish()
            setup_wandb(stage_config)

        model, env = train_agent(stage_config, model_config, previous_model=model, previous_vec_env=vec_env_stats_path)
        
        stage_model_path = os.path.join(stage_config.model_dir, "final_stage_model")
        stage_vecnorm_path = os.path.join(stage_config.model_dir, "vecnormalize.pkl")
        model.save(stage_model_path)
        env.save(stage_vecnorm_path)
        vec_env_stats_path = stage_vecnorm_path

        print(f"Evaluating Stage {i+1} model...")
        results = evaluate_agent(stage_model_path, n_episodes=stage_config.n_eval_episodes, normalize_path=stage_vecnorm_path)
        print(f"Stage {i+1} Results: Win Rate = {results['win_rate']:.1%}, Avg Ante = {results['avg_ante']:.2f}")

    if main_config.use_wandb:
        wandb.finish()
"""
    with open('src/training.py', 'w') as f:
        f.write(training_content)
    print("Created src/training.py")

def create_balatro_callbacks_file():
    # Content of src/callbacks/balatro_callbacks.py (pasted directly)
    callbacks_content = """
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import List, Dict, Any
import os

class BalatroCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.antes_reached = []
        self.blinds_beaten = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    reward = self.locals['rewards'][i]

                    self.episode_rewards.append(reward)
                    self.episode_lengths.append(self.locals['env'].buf_timesteps[i])

                    if 'ante_reached' in info:
                        self.antes_reached.append(info['ante_reached'])
                    if 'blinds_beaten' in info:
                        self.blinds_beaten.append(info['blinds_beaten'])
                    
                    if wandb.run is not None:
                        wandb.log({
                            'rollout/episode_reward': reward,
                            'rollout/episode_length': self.locals['env'].buf_timesteps[i],
                            'balatro/antes_reached': info.get('ante_reached', 0),
                            'balatro/blinds_beaten': info.get('blinds_beaten', 0),
                            'global_step': self.num_timesteps
                        })
        
        return True
    
    def _on_training_end(self) -> None:
        if self.episode_rewards:
            print("\\nGenerating final training progress plot...")
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(self.episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(1, 3, 2)
            plt.plot(self.episode_lengths)
            plt.title('Episode Lengths (Timesteps)')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.subplot(1, 3, 3)
            if self.antes_reached:
                plt.plot(self.antes_reached)
                plt.title('Antes Reached')
                plt.xlabel('Episode')
                plt.ylabel('Ante')
            
            plt.tight_layout()
            output_dir = self.model_save_path if hasattr(self, 'model_save_path') else './'
            plot_path = os.path.join(output_dir, 'final_training_progress.png')
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path)
            print(f"Saved final training progress plot to {plot_path}")
"""
    os.makedirs('src/callbacks', exist_ok=True) # Ensure directory exists
    with open('src/callbacks/balatro_callbacks.py', 'w') as f:
        f.write(callbacks_content)
    print("Created src/callbacks/balatro_callbacks.py")


if __name__ == "__main__":
    setup_project()