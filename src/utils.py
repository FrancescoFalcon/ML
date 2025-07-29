
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
                   hand_types: list, jokers: list, score: int, win: bool = None):
        self.episode_rewards.append(reward)
        self.antes_reached.append(ante)
        self.blinds_beaten.append(blinds)
        self.hand_types_played.extend(hand_types)
        self.jokers_active_per_episode.append(jokers)
        self.scores_achieved.append(score)
        if not hasattr(self, 'wins'):
            self.wins = []
        if win is not None:
            self.wins.append(win)
        else:
            # fallback: consider win if ante >= 8 (legacy behavior)
            self.wins.append(ante >= 8)
    
    def get_statistics(self) -> dict:
        if not self.episode_rewards:
            return {}
        all_jokers_flat = [joker for sublist in self.jokers_active_per_episode for joker in sublist]
        win_rate = np.mean(self.wins) if hasattr(self, 'wins') and len(self.wins) == len(self.episode_rewards) else np.mean([1 if ante >= 8 else 0 for ante in self.antes_reached])
        return {
            'win_rate': win_rate,
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
