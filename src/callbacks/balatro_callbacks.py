
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
            print("\nGenerating final training progress plot...")
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
