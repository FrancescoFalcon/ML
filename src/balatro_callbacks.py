from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import List, Dict, Any
import os
from collections import defaultdict

class BalatroCallback(BaseCallback):
    def __init__(self, verbose=1, training_env=None):  # Set verbose=1 by default
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.antes_reached = []
        self.blinds_beaten = []
        
        # Reference to training environment for continuous training info
        self.balatro_training_env = training_env  # Rename to avoid conflicts
        
        # Hand type tracking - CON TRACKING RECENTE
        self.hand_type_counts = defaultdict(int)
        self.total_hands_played = 0
        self.hand_type_history = []  # Track over time for plotting
        
        # Recent hand tracking per statistiche
        self.recent_hands = []  # Store recent hand types
        self.recent_window = 5000  # Track last 5000 hands
        
        # Joker acquisition tracking
        self.joker_purchases = defaultdict(int)  # Track joker types purchased
        self.total_jokers_purchased = 0
        self.joker_cost_history = []  # Track costs over time
        
        # NUOVO: Tracking progressione ante nel training
        self.training_max_ante_reached = 1  # Ante massimo raggiunto in tutto il training
        self.ante_progression_points = []  # [(hands_played, max_ante_reached), ...]
        self.ante_milestone_hands = []  # Mani giocate quando viene raggiunto un nuovo ante record
        
        # FIX: Previeni duplicazione del grafico finale
        self.training_ended = False  # Flag per evitare multiple chiamate
        
    def _on_step(self) -> bool:
        # Track hand types on every step (not just episode end)
        if self.locals.get('infos') is not None:
            for i, info in enumerate(self.locals['infos']):
                # Check if a hand was played this step
                if info.get('action_type') == 'play' and 'hand_type' in info:
                    hand_type_name = info['hand_type']
                    self.hand_type_counts[hand_type_name] += 1
                    self.total_hands_played += 1
                    
                    # 🔍 DEBUG: Stampa info dettagliate per ogni mano giocata
                    hand_score = info.get('hand_score', 0)  # Score di questa singola mano
                    total_score = info.get('score', 0)  # Score totale cumulativo per il blind
                    winning_score = info.get('winning_score', None)  # Score che ha battuto il blind
                    blind_requirement = info.get('chips_needed', 0)  # FIXED: usa chips_needed invece di blind_requirement
                    success = info.get('hand_success', False) or info.get('blind_beaten', False)
                    money_left = info.get('money', 0)
                    ante_current = info.get('ante', 1)  # FIXED: usa 'ante' invece di 'current_ante'
                    blind_current = info.get('blind', 0)  # Aggiungi anche il blind corrente
                    
                    # TRACKING ANTE: Aggiorna il massimo ante raggiunto durante il training
                    if ante_current > self.training_max_ante_reached:
                        self.training_max_ante_reached = ante_current
                        # Registra a quante mani è avvenuto questo nuovo record
                        self.ante_milestone_hands.append(self.total_hands_played)
                        print(f"🎯 NUOVO ANTE RECORD: {ante_current} raggiunto dopo {self.total_hands_played:,} mani!")
                    
                    # Aggiorna la progressione ante (ogni 1000 mani per non riempire memoria)
                    if len(self.ante_progression_points) == 0 or self.total_hands_played - self.ante_progression_points[-1][0] >= 1000:
                        self.ante_progression_points.append((self.total_hands_played, self.training_max_ante_reached))
                    
                    # Mostra winning score se il blind è stato battuto
                    if winning_score is not None:
                        score_display = f"VITTORIA! {winning_score}/{blind_requirement}"
                    elif money_left == 0 and blind_requirement > 0:
                        score_display = f"{total_score}/{blind_requirement} (Senza soldi)"
                    elif blind_requirement == 0:
                        score_display = f"{total_score}/??? (Blind sconosciuto)"
                    else:
                        score_display = f"{total_score}/{blind_requirement}"
                    
                    # TRAINING VELOCE: Solo milestone ogni 1000 mani per non rallentare
                    if self.total_hands_played % 1000 == 0:
                        # print(f"📊 PROGRESSO: {self.total_hands_played:,} mani | Episodi: {len(self.episode_rewards)} | Ante max: {self.training_max_ante_reached}")
                        pass
                    
                    # Track recent hands for convergence
                    self.recent_hands.append(hand_type_name)
                    if len(self.recent_hands) > self.recent_window:
                        self.recent_hands.pop(0)  # Keep only last 5000 hands
                
                # Check if a joker was purchased this step
                if info.get('shop_action', '').startswith('bought_'):
                    joker_type = info['shop_action'].replace('bought_', '')
                    self.joker_purchases[joker_type] += 1
                    self.total_jokers_purchased += 1
                    
                    # Track joker purchase (removed debug logging)
                    money_spent = info.get('money_spent', 0)
                    
                    # Track cost if available
                    if 'money_spent' in info:
                        self.joker_cost_history.append(info['money_spent'])
                
                # FIXED: Move logging outside joker purchase block (removed verbose logging)
                if self.verbose >= 2 and self.total_hands_played > 0 and self.total_hands_played % 50000 == 0:
                    print(f"[TRACKING] Total hands played: {self.total_hands_played}")
                    print(f"[TRACKING] Total jokers purchased: {self.total_jokers_purchased}")
                    # Print current distribution every 20000 hands
                    print("  Current hand distribution:")
                    total = sum(self.hand_type_counts.values())
                    if total > 0:
                        for hand_type, count in sorted(self.hand_type_counts.items(), key=lambda x: x[1], reverse=True):
                            if count > 0:
                                percentage = count / total * 100
                                print(f"    {hand_type:15}: {count:3} hands ({percentage:4.1f}%)")
                                
                        # Show recent distribution as well
                        if len(self.recent_hands) >= 1000:
                            print(f"  Recent {len(self.recent_hands)} hands distribution:")
                            recent_counts = defaultdict(int)
                            for hand in self.recent_hands:
                                recent_counts[hand] += 1
                            for hand_type, count in sorted(recent_counts.items(), key=lambda x: x[1], reverse=True):
                                if count > 0:
                                    percentage = count / len(self.recent_hands) * 100
                                    print(f"    {hand_type:15}: {count:3} hands ({percentage:4.1f}%)")
        
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    reward = self.locals['rewards'][i]

                    self.episode_rewards.append(reward)
                    
                    # ANTI-DUPLICAZIONE ROBUSTA: Usa reward + ante + hands per identificare episodi unici
                    episode_unique_id = f"reward_{reward}_ante_{info.get('ante_reached', 1)}_hands_{self.total_hands_played}"
                    if not hasattr(self, '_printed_episodes'):
                        self._printed_episodes = set()
                    
                    # TRAINING VELOCE: Solo episodi positivi o milestone significativi
                    if reward > 5 and episode_unique_id not in self._printed_episodes:
                        # print(f"✅ SUCCESSO: Reward={reward:.1f} | Ante={info.get('ante_reached', 1)} | Mani={self.total_hands_played:,}")
                        self._printed_episodes.add(episode_unique_id)
                    
                    # SB3 non garantisce buf_timesteps, quindi salta episode_lengths o usa info se presente
                    if 'episode_length' in info:
                        self.episode_lengths.append(info['episode_length'])

                    # NUOVO: Traccia progressione ante nel training
                    current_ante_reached = info.get('ante_reached', 1)
                    self.antes_reached.append(current_ante_reached)
                    
                    # Aggiorna il massimo ante raggiunto nel training
                    if current_ante_reached > self.training_max_ante_reached:
                        self.training_max_ante_reached = current_ante_reached
                        # Registra a quante mani è avvenuto questo nuovo record
                        self.ante_milestone_hands.append(self.total_hands_played)
                        
                        # � COLORIZED NEW ANTE RECORD - SOLO PER VERI NUOVI RECORD! 🌟
                        # ANSI color codes for bright, attention-grabbing output
                        RESET = '\033[0m'
                        BRIGHT_GREEN = '\033[92m'
                        BRIGHT_YELLOW = '\033[93m'
                        BRIGHT_CYAN = '\033[96m'
                        BRIGHT_MAGENTA = '\033[95m'
                        BRIGHT_RED = '\033[91m'
                        BOLD = '\033[1m'
                        UNDERLINE = '\033[4m'
                        
                        # Create spectacular colorful border for NEW RECORDS ONLY
                        border_top = f"{BRIGHT_YELLOW}{'�' * 25}{RESET}"
                        border_bottom = f"{BRIGHT_CYAN}{'⭐' * 25}{RESET}"
                        
                        print(f"\n{border_top}")
                        print(f"{BOLD}{BRIGHT_GREEN}� NUOVO RECORD DI ANTE! �{RESET}")
                        print(f"{BOLD}{BRIGHT_CYAN}📈 ANTE {current_ante_reached} RAGGIUNTO PER LA PRIMA VOLTA! 📈{RESET}")
                        print(f"{BOLD}{BRIGHT_MAGENTA}� Mani giocate totali: {self.total_hands_played:,}{RESET}")
                        
                        # Add achievement level celebration for new records
                        if current_ante_reached >= 7:
                            print(f"{BOLD}{BRIGHT_RED}💎 RECORD LEGGENDARIO! INCREDIBILE ACHIEVEMENT! 💎{RESET}")
                        elif current_ante_reached >= 5:
                            print(f"{BOLD}{BRIGHT_YELLOW}👑 RECORD DI ALTO LIVELLO! ECCELLENTE! 👑{RESET}")
                        elif current_ante_reached >= 3:
                            print(f"{BOLD}{BRIGHT_GREEN}� GRANDE PROGRESSO! NUOVO TRAGUARDO! 🚀{RESET}")
                        else:
                            print(f"{BOLD}{BRIGHT_CYAN}🎯 NUOVO MILESTONE RAGGIUNTO! 🎯{RESET}")
                        
                        print(f"{border_bottom}\n")
                        
                        # Also log the plain text version for compatibility
                        print(f"[ANTE NEW RECORD] Primo raggiungimento Ante {current_ante_reached} dopo {self.total_hands_played:,} mani giocate")
                    
                    # Registra sempre la progressione attuale (ogni 1000 mani per non riempire memoria)
                    if len(self.ante_progression_points) == 0 or self.total_hands_played - self.ante_progression_points[-1][0] >= 1000:
                        self.ante_progression_points.append((self.total_hands_played, self.training_max_ante_reached))
                    
                    if 'blinds_beaten' in info:
                        self.blinds_beaten.append(info['blinds_beaten'])
                    
                    # Log total reward
                    total_reward = info.get('total_reward', 0)

                    # Save current hand type statistics snapshot
                    if self.total_hands_played > 0:
                        self.hand_type_history.append(dict(self.hand_type_counts))

                    if wandb.run is not None:
                        wandb.log({
                            'rollout/episode_reward': reward,
                            'rollout/total_reward': total_reward,
                            # 'rollout/episode_length': info.get('episode_length', 0),
                            'balatro/antes_reached': info.get('ante_reached', 0),
                            'balatro/blinds_beaten': info.get('blinds_beaten', 0),
                            'global_step': self.num_timesteps
                        })
                        
                        # Log hand distribution to wandb
                        if self.total_hands_played > 0:
                            hand_dist_log = {}
                            
                            # Log overall distribution
                            total = sum(self.hand_type_counts.values())
                            for hand_type, count in self.hand_type_counts.items():
                                hand_dist_log[f'hand_dist/{hand_type}'] = count / total
                            
                            # Log recent distribution
                            if len(self.recent_hands) > 0:
                                recent_counts = defaultdict(int)
                                for hand in self.recent_hands:
                                    recent_counts[hand] += 1
                                for hand_type, count in recent_counts.items():
                                    hand_dist_log[f'recent_hand_dist/{hand_type}'] = count / len(self.recent_hands)
                            
                            wandb.log(hand_dist_log)

        return True

    def _on_training_end(self) -> None:
        # FIX: Previeni duplicazione del grafico se già chiamato
        if self.training_ended:
            if self.verbose >= 1:
                print("[CALLBACK] Training end già gestito, evito duplicazione grafico")
            return
        self.training_ended = True
        if self.episode_rewards:
            print("\nGenerating final training progress plot...")
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            # ...existing code for plotting...
            plt.tight_layout()
            plot_dir = os.path.join(os.getcwd(), "data", "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "training_progress_with_hands.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {plot_path}")

            if self.hand_type_counts and self.total_hands_played > 0:
                stats_dir = os.path.join(os.getcwd(), "data", "results")
                os.makedirs(stats_dir, exist_ok=True)
                stats_path = os.path.join(stats_dir, "training_statistics.txt")
                with open(stats_path, 'w') as f:
                    f.write(f"TRAINING STATISTICS\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"HAND TYPE STATISTICS (Total hands played: {self.total_hands_played})\n")
                    f.write("-" * 60 + "\n")
                    for hand_type, count in sorted(self.hand_type_counts.items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            percentage = count / self.total_hands_played * 100
                            f.write(f"{hand_type:15}: {count:6} hands ({percentage:5.1f}%)\n")
                    f.write("-" * 60 + "\n\n")
                    if self.joker_purchases and self.total_jokers_purchased > 0:
                        f.write(f"JOKER PURCHASE STATISTICS (Total jokers purchased: {self.total_jokers_purchased})\n")
                        f.write("-" * 60 + "\n")
                        for joker_type, count in sorted(self.joker_purchases.items(), key=lambda x: x[1], reverse=True):
                            percentage = count / self.total_jokers_purchased * 100
                            clean_name = joker_type.replace('_', ' ').title()
                            if clean_name == 'Joker':
                                clean_name = 'Basic Joker'
                            f.write(f"{clean_name:15}: {count:6} purchases ({percentage:5.1f}%)\n")
                        if self.joker_cost_history:
                            avg_cost = sum(self.joker_cost_history) / len(self.joker_cost_history)
                            f.write(f"\nAverage joker cost: ${avg_cost:.1f}\n")
                            f.write(f"Cost range: ${min(self.joker_cost_history)} - ${max(self.joker_cost_history)}\n")
                        f.write("-" * 60 + "\n")
                    f.write("=" * 60 + "\n")
                print(f"Training statistics saved to {stats_path}")
            
            # Hand type distribution (recent)
            if self.recent_hands:
                recent_counts = defaultdict(int)
                for hand in self.recent_hands:
                    recent_counts[hand] += 1
                
                hand_types = list(recent_counts.keys())
                counts = list(recent_counts.values())
                
                axes[1, 0].pie(counts, labels=hand_types, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title(f'Recent {len(self.recent_hands)} Hands Distribution')

            # Print statistics to console
            print(f"\n🃏 HAND TYPE STATISTICS (Total hands played: {self.total_hands_played}):")
            print("=" * 60)
            for hand_type, count in sorted(self.hand_type_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = count / self.total_hands_played * 100
                    print(f"{hand_type:15}: {count:6} hands ({percentage:5.1f}%)")
            print("=" * 60)
            
            # Print recent hand stats
            if len(self.recent_hands) >= 100:
                print(f"\n📊 RECENT {len(self.recent_hands)} HANDS STATISTICS:")
                print("=" * 60)
                recent_counts = defaultdict(int)
                for hand in self.recent_hands:
                    recent_counts[hand] += 1
                for hand_type, count in sorted(recent_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(self.recent_hands) * 100
                    print(f"{hand_type:15}: {count:6} hands ({percentage:5.1f}%)")
                print("=" * 60)
            
            # Print joker purchase stats
            if self.joker_purchases and self.total_jokers_purchased > 0:
                print(f"\n🃏 JOKER PURCHASE STATISTICS (Total jokers purchased: {self.total_jokers_purchased}):")
                print("=" * 60)
                for joker_type, count in sorted(self.joker_purchases.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / self.total_jokers_purchased * 100
                    clean_name = joker_type.replace('_', ' ').title()
                    if clean_name == 'Joker':
                        clean_name = 'Basic Joker'
                    print(f"{clean_name:15}: {count:6} purchases ({percentage:5.1f}%)")
                
                if self.joker_cost_history:
                    avg_cost = sum(self.joker_cost_history) / len(self.joker_cost_history)
                    print(f"\nAverage joker cost: ${avg_cost:.1f}")
                    print(f"Cost range: ${min(self.joker_cost_history)} - ${max(self.joker_cost_history)}")
                print("=" * 60)

            # Antes reached - SEMPRE MOSTRA GRAFICO ANTE!
            # Prima prova a mostrare la progressione cumulativa
            if self.ante_progression_points and len(self.ante_progression_points) > 1:
                # Estrai i dati per il grafico della progressione
                hands_played = [point[0] for point in self.ante_progression_points]
                max_antes = [point[1] for point in self.ante_progression_points]
                
                axes[1, 1].plot(hands_played, max_antes, 'b-', linewidth=2, label='Ante massimo raggiunto')
                axes[1, 1].set_title('Progressione Ante Massimo nel Training')
                axes[1, 1].set_xlabel('Mani Giocate (Totale)')
                axes[1, 1].set_ylabel('Ante Massimo Raggiunto')
                axes[1, 1].set_ylim(0.5, max(max_antes) + 0.5)
                axes[1, 1].set_yticks(range(1, max(max_antes) + 1))
                axes[1, 1].grid(True, alpha=0.3)
                
                # Aggiungi markers per i milestone
                if self.ante_milestone_hands:
                    milestone_antes = []
                    for i, hands in enumerate(self.ante_milestone_hands):
                        # Trova l'ante corrispondente a questo milestone
                        corresponding_ante = i + 2  # Primo milestone è ante 2, secondo è ante 3, etc.
                        if corresponding_ante <= max(max_antes):
                            milestone_antes.append(corresponding_ante)
                            axes[1, 1].scatter([hands], [corresponding_ante], 
                                             color='red', s=100, zorder=5, 
                                             label=f'Ante {corresponding_ante} raggiunto' if i == 0 else '')
                
                axes[1, 1].legend()
            else:
                # Se non abbiamo progressione cumulativa, mostra antes per episodio
                if self.antes_reached:
                    axes[1, 1].plot(self.antes_reached, 'g-', linewidth=1, alpha=0.7)
                    axes[1, 1].set_title('Antes Reached per Episodio')
                    axes[1, 1].set_xlabel('Episode')
                    axes[1, 1].set_ylabel('Ante Level')
                    axes[1, 1].set_ylim(0.5, max(self.antes_reached) + 0.5)
                    axes[1, 1].set_yticks(range(1, max(self.antes_reached) + 2))
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Aggiungi statistiche come testo
                    max_ante = max(self.antes_reached)
                    avg_ante = np.mean(self.antes_reached)
                    axes[1, 1].text(0.02, 0.98, 
                                   f'Max Ante: {max_ante}\nAvg Ante: {avg_ante:.2f}\nEpisodi: {len(self.antes_reached)}',
                                   transform=axes[1, 1].transAxes, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    # Se non abbiamo nemmeno dati per episodio, mostra almeno un messaggio
                    axes[1, 1].text(0.5, 0.5, 'Nessun dato ante disponibile\n(Agente rimasto ad Ante 1)', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes,
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    axes[1, 1].set_title('Progressione Ante - Nessun Progresso')
                    axes[1, 1].set_xlim(0, 1)
                    axes[1, 1].set_ylim(0, 1)
            
            # SEMPRE stampa statistiche dei milestone e progressione ante
            print(f"\n🎯 STATISTICHE PROGRESSIONE ANTE:")
            print("=" * 60)
            print(f"Ante massimo raggiunto nel training: {self.training_max_ante_reached}")
            if self.ante_milestone_hands:
                for i, hands in enumerate(self.ante_milestone_hands):
                    ante = i + 2  # Primo milestone è ante 2
                    print(f"Ante {ante} raggiunto dopo {hands:,} mani giocate")
            else:
                print("Nessun nuovo ante raggiunto - agente rimasto ad Ante 1")
            
            if self.antes_reached:
                print(f"Episodi totali: {len(self.antes_reached)}")
                print(f"Ante medio per episodio: {np.mean(self.antes_reached):.2f}")
                print(f"Ante massimo in singolo episodio: {max(self.antes_reached)}")
                
                # Conta quanti episodi per ogni ante
                ante_counts = defaultdict(int)
                for ante in self.antes_reached:
                    ante_counts[ante] += 1
                print("Distribuzione episodi per ante:")
                for ante in sorted(ante_counts.keys()):
                    count = ante_counts[ante]
                    percentage = count / len(self.antes_reached) * 100
                    print(f"  Ante {ante}: {count} episodi ({percentage:.1f}%)")
            print("=" * 60)
            
            plt.tight_layout()
            plt.savefig(os.path.join("./data/plots/", "training_progress_with_hands.png"), dpi=300, bbox_inches='tight')
            print("Training progress plot saved to ./data/plots/training_progress_with_hands.png")
            
            # Also save hand type statistics to a text file
            if self.hand_type_counts and self.total_hands_played > 0:
                stats_path = "./data/results/training_statistics.txt"
                os.makedirs(os.path.dirname(stats_path), exist_ok=True)
                with open(stats_path, 'w') as f:
                    f.write(f"TRAINING STATISTICS\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # Hand type statistics
                    f.write(f"HAND TYPE STATISTICS (Total hands played: {self.total_hands_played})\n")
                    f.write("-" * 60 + "\n")
                    for hand_type, count in sorted(self.hand_type_counts.items(), key=lambda x: x[1], reverse=True):
                        if count > 0:
                            percentage = count / self.total_hands_played * 100
                            f.write(f"{hand_type:15}: {count:6} hands ({percentage:5.1f}%)\n")
                    f.write("-" * 60 + "\n\n")
                    
                    # Joker purchase statistics
                    if self.joker_purchases and self.total_jokers_purchased > 0:
                        f.write(f"JOKER PURCHASE STATISTICS (Total jokers purchased: {self.total_jokers_purchased})\n")
                        f.write("-" * 60 + "\n")
                        for joker_type, count in sorted(self.joker_purchases.items(), key=lambda x: x[1], reverse=True):
                            percentage = count / self.total_jokers_purchased * 100
                            clean_name = joker_type.replace('_', ' ').title()
                            if clean_name == 'Joker':
                                clean_name = 'Basic Joker'
                            f.write(f"{clean_name:15}: {count:6} purchases ({percentage:5.1f}%)\n")
                        
                        if self.joker_cost_history:
                            avg_cost = sum(self.joker_cost_history) / len(self.joker_cost_history)
                            f.write(f"\nAverage joker cost: ${avg_cost:.1f}\n")
                            f.write(f"Cost range: ${min(self.joker_cost_history)} - ${max(self.joker_cost_history)}\n")
                        f.write("-" * 60 + "\n")
                    
                    f.write("=" * 60 + "\n")
                print(f"Training statistics saved to {stats_path}")
            
            # plt.show() # Disabilitato: nessun grafico intermedio, solo salvataggio file
