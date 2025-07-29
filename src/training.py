import os
import platform
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback

# Impostazioni globali per PyTorch: usa quasi tutti i core CPU
import torch
max_threads = max(1, os.cpu_count() - 1)
torch.set_num_threads(max_threads)
torch.set_num_interop_threads(1)

# Disable wandb symlink creation on Windows to avoid OSError
if platform.system() == "Windows":
    os.environ["WANDB_SYMLINK_DISABLE"] = "true"

import sys

# Fix: importa evaluate_agent se esiste in src.utils
try:
    from src.utils import evaluate_agent
except ImportError:
    def evaluate_agent(*args, **kwargs):
        print("[WARN] evaluate_agent non implementata, placeholder chiamato.")
        return {'win_rate': 0, 'avg_reward': 0, 'avg_ante': 0, 'avg_blinds': 0}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.balatro_env import BalatroEnv
from src.utils import load_config, BalatroMetrics, plot_training_progress, create_performance_report
from src.balatro_callbacks import BalatroCallback

class TrainingConfig:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Hardcoded configuration for maximum stability. Ignores YAML files."""
        # OVERRIDE COMPLETO per ultra-stabilità
        self.n_envs = 1
        self.max_ante = 8
        self.starting_money = 4
        self.hand_size = 8
        self.max_jokers = 5

        self.algorithm = 'PPO'
        self.total_timesteps = 1800000  # 3 stage × 600k timesteps
        self.learning_rate = 3e-4  # Standard PPO LR
        self.batch_size = 256      # Standard PPO batch size
        self.n_steps = 2048        # Standard PPO n_steps
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2      # Standard PPO clip range
        self.ent_coef = 0.01       # RIDOTTO da 0.1 per maggiore stabilità
        self.vf_coef = 0.5         # AUMENTATO da 0.1 per migliorare value function
        self.max_grad_norm = 0.5   # Standard PPO max_grad_norm
        self.device = 'cpu'

        self.eval_freq = 10000
        self.n_eval_episodes = 100
        self.deterministic_eval = True

        self.log_dir = './data/logs/'
        self.model_dir = './data/models/'
        self.use_wandb = True
        self.wandb_project = 'balatro-rl-stability-test'
        self.wandb_entity = None
        self.save_freq = 50000
        
        self.curriculum_enabled = True
        self.curriculum_stages = [
            {'max_ante': 8, 'timesteps': 600000},  # Stage 1: training completo  
            {'max_ante': 8, 'timesteps': 600000},  # Stage 2: training completo
            {'max_ante': 8, 'timesteps': 600000}   # Stage 3: training completo
        ]
        
        self.phased_training_enabled = False
        
        self.num_cpu_workers = os.cpu_count()

class ModelConfig:
    def __init__(self, config_path="configs/model_config.yaml"):
        config = load_config(config_path)

        self.policy_type = config['network']['policy_type']
        self.net_arch = [int(x) for x in config['network']['net_arch']]
        self.activation_fn = config['network']['activation_fn']
        self.ortho_init = bool(config['network']['ortho_init'])

        self.normalize_obs = bool(config['features']['normalize_obs'])
        self.normalize_reward = True  # FORCE TRUE per stabilizzare training
        self.clip_obs = float(config['features']['clip_obs'])

        self.use_sde = bool(config['advanced']['use_sde'])
        self.sde_sample_freq = int(config['advanced']['sde_sample_freq'])
        self.target_kl = (float(config['advanced']['target_kl']) if config['advanced']['target_kl'] is not None else None)

        # Ensure all reward weights are float
        self.reward_weights = {k: float(v) for k, v in config['reward_function']['weights'].items()}

def create_env(config: TrainingConfig, eval_env: bool = False):
    def _init():
        env_max_ante = config.max_ante if not eval_env else 8
        # USE FIXED MAX_JOKERS FROM CONFIG instead of dynamic
        # Dynamic was causing max_jokers=3 instead of 5
        env = BalatroEnv(max_ante=env_max_ante,
                         starting_money=config.starting_money,
                         hand_size=config.hand_size,
                         max_jokers=config.max_jokers,  # FIXED: use config value directly
                         debug=not eval_env)  # Debug only for training env, not eval env
        env = Monitor(env, config.log_dir)
        # No more dynamic max_jokers - use fixed config value for stability
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
    print("[INFO] Using CPU only (PPO MlpPolicy) - massima potenza")
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Forza wandb disattivato per evitare errori su Windows/test
    config.use_wandb = False
    # Non chiamare mai setup_wandb se wandb è disattivato
    # if wandb.run is None:
    #     setup_wandb(config)
    
    # Configura VecEnv per ottimizzazioni CPU/GPU
    if config.n_envs == 1:
        # Single environment: usa DummyVecEnv (più semplice, no multiprocessing overhead)
        env = make_vec_env(
            create_env(config),
            n_envs=config.n_envs,
            seed=42
        )
    else:
        # Multiple environments: usa SubprocVecEnv per parallelismo reale
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = make_vec_env(
            create_env(config),
            n_envs=config.n_envs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={'start_method': 'spawn'},  # Windows/CUDA compatibility
            seed=42
        )
    
    if previous_vec_env:
        # Carica VecNormalize stats dal stage precedente
        if os.path.exists(previous_vec_env):
            env = VecNormalize.load(previous_vec_env, env)
            print(f"✅ Caricato VecNormalize da {previous_vec_env}")
        else:
            # Crea nuovo VecNormalize
            env = VecNormalize(env, norm_obs=model_config.normalize_obs, 
                              norm_reward=True, gamma=config.gamma,
                              clip_obs=model_config.clip_obs, clip_reward=10.0)
    else:
        # ABILITA VecNormalize anche per nuovi environment
        env = VecNormalize(env, norm_obs=model_config.normalize_obs, 
                          norm_reward=True, gamma=config.gamma,
                          clip_obs=model_config.clip_obs, clip_reward=10.0)

    eval_env = make_vec_env(
        create_env(config, eval_env=True),
        n_envs=1,
        seed=123
    )
    # ABILITA VecNormalize anche per eval_env per consistenza
    eval_env = VecNormalize(eval_env, norm_obs=model_config.normalize_obs, 
                           norm_reward=False, gamma=config.gamma,  # No reward norm for eval
                           clip_obs=model_config.clip_obs, training=False)

    if previous_model is not None:
        # Resuming training from previous model instance
        model = previous_model
        model.set_env(env)
        print("Continuing training with loaded model.")
    else:
        # Initializing new PPO model (no previous model provided)
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
            target_kl=model_config.target_kl,
            device=config.device  # Explicit device setting
        )
    

    # === LOG MANI GIOCATE ===
    class HandCounterCallback(BalatroCallback):
        def __init__(self, log_every_steps=20480): # Log every 10 rollouts
            super().__init__()  # Remove training_env parameter
            self.total_hands = 0
            self.last_logged_step = 0
            self.log_every_steps = log_every_steps
        def _on_step(self) -> bool:
            # Conta le mani effettivamente giocate (hand_type presente in info)
            infos = self.locals.get('infos', [])
            for info in infos:
                if info.get('hand_type') is not None:
                    self.total_hands += 1
            # Logga ogni log_every_steps (SOLO essenziale)
            num_timesteps = self.num_timesteps if hasattr(self, 'num_timesteps') else self.locals.get('num_timesteps', 0)
            if num_timesteps - self.last_logged_step >= self.log_every_steps:
                # Clean training output - only essential logging
                self.last_logged_step = num_timesteps
            return super()._on_step()

    hand_counter_callback = HandCounterCallback()
    
    # Add BalatroCallback for detailed statistics and final plot
    balatro_callback = BalatroCallback(verbose=1)
    
    callbacks = [hand_counter_callback, balatro_callback]

    # Disabilitato di nuovo - causa blocchi durante evaluation
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=os.path.join(config.model_dir, "best_model"),
    #     log_path=config.log_dir,
    #     eval_freq=max(config.eval_freq // config.n_envs, 1),
    #     n_eval_episodes=config.n_eval_episodes,
    #     deterministic=config.deterministic_eval,
    #     render=False
    # )
    # callbacks.append(eval_callback)

    # Non aggiungere mai WandbCallback
    # if config.use_wandb:
    #     callbacks.append(WandbCallback(
    #         gradient_save_freq=config.save_freq,
    #         model_save_path=f"{config.model_dir}/wandb_model",
    #         verbose=2
    #     ))
    

    print("Starting training...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO_Balatro_Run"
    )
    
    model.save(f"{config.model_dir}/final_model.zip")
    # Salva VecNormalize stats per consistenza
    env.save(f"{config.model_dir}/vecnormalize.pkl")
    
    print("Training completed!")
    return model, env, hand_counter_callback, balatro_callback

def evaluate_trained_model(config: TrainingConfig, model_config: ModelConfig, model, n_episodes: int = 25):
    """Evaluate a trained model for final comparison"""
    print("Evaluating trained model for final comparison...")
    
    # Create environment IDENTICA al training per consistenza
    def make_env():
        env = BalatroEnv(max_ante=config.max_ante,
                         starting_money=config.starting_money,
                         hand_size=config.hand_size,
                         max_jokers=config.max_jokers,
                         debug=True)  # 🔍 ABILITA DEBUG anche per evaluation!
        env = Monitor(env)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    # Set the model's environment 
    model.set_env(env)
    
    # Esegui MULTIPLE runs per statistiche più robuste (come baseline)
    total_reward = 0
    all_hand_types = []
    total_blinds_beaten = 0
    all_ante_progression = []
    total_steps = 0
    runs_completed = 0
    
    for run in range(n_episodes):
        obs = env.reset()
        done = False
        run_reward = 0
        run_hand_types = []
        run_blinds_beaten = 0
        run_ante_progression = []
        run_steps = 0
        
        while not done and run_steps < 1000:  # Max 1000 steps per run per evitare loop infiniti
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            run_reward += reward[0]
            done = done[0]
            current_info = info[0] if isinstance(info, list) else info
            run_steps += 1
            
            if current_info.get('hand_type'):
                run_hand_types.append(current_info['hand_type'])
            if current_info.get('blind_beaten', False):
                run_blinds_beaten += 1
                ante = current_info.get('ante', None)
                blind = current_info.get('blind', None)
                run_ante_progression.append((ante, blind))
        
        # Accumula statistiche
        total_reward += run_reward
        all_hand_types.extend(run_hand_types)
        total_blinds_beaten += run_blinds_beaten
        all_ante_progression.extend(run_ante_progression)
        total_steps += run_steps
        runs_completed += 1
        
        # Log del progresso
       #     print(f"  Run {run+1}: {run_steps} steps, reward {run_reward:.2f}, blinds {run_blinds_beaten}")
    
    # Calcola medie
    avg_reward = total_reward / runs_completed if runs_completed > 0 else 0
    avg_blinds = total_blinds_beaten / runs_completed if runs_completed > 0 else 0
    avg_steps = total_steps / runs_completed if runs_completed > 0 else 0
    
    from collections import Counter
    hand_counter = Counter(all_hand_types)
    # BUG FIX: Calcola il vero massimo ante raggiunto dall'agente
    max_ante_reached = max([a for a, b in all_ante_progression] + [1])
    # Calcola anche la media degli ante raggiunti per una metrica più accurata
    all_antes_reached = [a for a, b in all_ante_progression if a is not None]
    avg_ante_reached = sum(all_antes_reached) / len(all_antes_reached) if all_antes_reached else 1.0
    
    print(f"\n=== TRAINED MODEL EVALUATION ({runs_completed} runs) ===")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average blinds beaten: {avg_blinds:.2f}")
    print(f"Max ante reached: {max_ante_reached}")
    print(f"Average ante reached: {avg_ante_reached:.2f}")
    print(f"Total hands played: {len(all_hand_types)}")
    print(f"Most common hands: {hand_counter.most_common(5)}")
    env.close()
    
    return {
        'total_steps': total_steps,
        'total_reward': total_reward,
        'blinds_beaten': total_blinds_beaten,
        'ante_progression': all_ante_progression,
        'most_common_hands': hand_counter.most_common(5),
        'win_rate': (total_blinds_beaten / runs_completed) / 3.0 if runs_completed > 0 else 0.0,  # Normalizzato per 3 blinds
        'avg_reward': avg_reward,
        'avg_ante': avg_ante_reached,  # BUG FIX: Usa il vero average ante raggiunto
        'max_ante_reached': max_ante_reached,  # Aggiungi anche il massimo
        'avg_blinds': avg_blinds
    }

def evaluate_untrained_model(config: TrainingConfig, model_config: ModelConfig, n_episodes: int = 25):
    """Evaluate a completely untrained model for baseline comparison"""
    print("Creating untrained model for baseline evaluation...")
    
    # Create environment
    def make_env():
        env = BalatroEnv(max_ante=config.max_ante)
        env = Monitor(env)
        return env
    
    env = make_vec_env(make_env, n_envs=1)
    
    if model_config.normalize_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=config.gamma)
    
    # Create untrained model
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
        device=config.device,
        verbose=0,
        policy_kwargs=dict(net_arch=model_config.net_arch)
    )
    
    # Esegui una singola run lunga, stampa progressione ante/blind e reward
    obs = env.reset()
    done = False
    total_reward = 0
    hand_types = []
    blinds_beaten = 0
    ante_progression = []
    step_count = 0
    def get_env_attr(attr, default=None):
        if hasattr(env, 'env_method'):
            try:
                return env.env_method('__getattribute__', attr)[0]
            except:
                return default
        return default
    while not done:
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        done = done[0]
        current_info = info[0] if isinstance(info, list) else info
        step_count += 1
        if current_info.get('hand_type'):
            hand_types.append(current_info['hand_type'])
        if current_info.get('blind_beaten', False):
            blinds_beaten += 1
            ante = current_info.get('ante', None)
            blind = current_info.get('blind', None)
            ante_progression.append((ante, blind))
    from collections import Counter
    hand_counter = Counter(hand_types)
    print("\n=== SINGLE RUN UNTRAINED MODEL ===")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Blinds beaten: {blinds_beaten}")
    print(f"Ante progression (ante, blind): {ante_progression}")
    print(f"Most common hands: {hand_counter.most_common(5)}")
    env.close()
    
    # BUG FIX: Calcola il vero massimo ante raggiunto per untrained model
    max_ante_reached = max([a for a, b in ante_progression] + [1])
    
    return {
        'total_steps': step_count,
        'total_reward': total_reward,
        'blinds_beaten': blinds_beaten,
        'ante_progression': ante_progression,
        'most_common_hands': hand_counter.most_common(5),
        'win_rate': 1.0 if blinds_beaten >= 3 else 0.0,  # Win se batte almeno 3 blinds
        'avg_reward': total_reward,
        'avg_ante': max_ante_reached,  # BUG FIX: Usa il vero massimo ante raggiunto
        'avg_blinds': blinds_beaten
    }


def phased_training():
    """Execute 5-phase training with individual evaluations and progress comparison"""
    main_config = TrainingConfig("configs/phased_training_config.yaml")
    model_config = ModelConfig()
    main_config.use_wandb = False  # Force disable wandb
    
    if not main_config.phased_training_enabled:
        print("Phased training is disabled. Running standard curriculum training.")
        curriculum_training()
        return
    
    print("🚀 Starting 5-Phase Training (50 minutes total)")
    print("=" * 70)
    print(f"Total Duration: {main_config.phased_config['total_duration_minutes']} minutes")
    print(f"Number of Phases: {main_config.phased_config['num_phases']}")
    print("=" * 70)
    
    # Store results for each phase
    phase_results = []
    
    model = None
    vec_env_stats_path = None
    
    # Execute each phase
    for i, phase_config in enumerate(main_config.phased_config['phases']):
        phase_num = i + 1
        print(f"\n{'='*20} PHASE {phase_num}: {phase_config['name']} {'='*20}")
        print(f"Duration: {phase_config['duration_minutes']} minutes")
        print(f"Timesteps: {phase_config['timesteps_per_minute'] * phase_config['duration_minutes']:,}")
        print(f"Max Ante: {phase_config['max_ante']}")
        
        # Configure phase-specific settings
        phase_training_config = TrainingConfig("configs/phased_training_config.yaml")
        phase_training_config.max_ante = phase_config['max_ante']
        phase_training_config.total_timesteps = phase_config['timesteps_per_minute'] * phase_config['duration_minutes']
        phase_training_config.n_eval_episodes = phase_config['evaluation_episodes']
        phase_training_config.log_dir = os.path.join(main_config.log_dir, f"phase_{phase_num}")
        phase_training_config.model_dir = os.path.join(main_config.model_dir, f"phase_{phase_num}")
        phase_training_config.use_wandb = False
        
        os.makedirs(phase_training_config.log_dir, exist_ok=True)
        os.makedirs(phase_training_config.model_dir, exist_ok=True)
        
        print(f"⏰ Starting Phase {phase_num} training...")
        import time
        phase_start_time = time.time()
        
        # Train the model
        model, env = train_agent(phase_training_config, model_config, 
                                previous_model=model, previous_vec_env=vec_env_stats_path)
        
        phase_end_time = time.time()
        phase_duration = (phase_end_time - phase_start_time) / 60  # in minutes
        print(f"⏱️ Phase {phase_num} completed in {phase_duration:.1f} minutes")
        
        # Save phase model
        phase_model_path = os.path.join(phase_training_config.model_dir, f"phase_{phase_num}_model.zip")
        phase_vecnorm_path = os.path.join(phase_training_config.model_dir, "vecnormalize.pkl")
        model.save(phase_model_path)
        env.save(phase_vecnorm_path)
        vec_env_stats_path = phase_vecnorm_path
        
        # Evaluate phase
        # RIMOSSA valutazione automatica del modello dopo ogni fase
        pass
    
    # === FINAL ANALYSIS ===
    print(f"\n{'='*70}")
    print("🏆 PHASED TRAINING COMPLETE - FINAL ANALYSIS")
    print(f"{'='*70}")
    
    # Display all phase results
    print("\n📈 PHASE-BY-PHASE RESULTS:")
    print("-" * 70)
    for phase_name, results in phase_results:
        print(f"{phase_name:12} | Win Rate: {results['win_rate']:6.1%} | "
              f"Avg Reward: {results['avg_reward']:6.2f} | "
              f"Avg Ante: {results['avg_ante']:6.2f} | "
              f"Avg Blinds: {results['avg_blinds']:6.2f}")
    print("-" * 70)
    
    # Compare first vs final phase
    first_data = phase_results[0][1]   # First phase  
    final_data = phase_results[-1][1]  # Last phase
    
    # Calculate improvements (CORRECTED: final - first per mostrare miglioramenti positivi)
    print(f"\n📈 IMPROVEMENTS:")
    win_rate_improvement = final_data['win_rate'] - first_data['win_rate']
    reward_improvement = final_data['avg_reward'] - first_data['avg_reward']
    ante_improvement = final_data['avg_ante'] - first_data['avg_ante']
    blinds_improvement = final_data['avg_blinds'] - first_data['avg_blinds']
    
    print(f"  - Win Rate: {win_rate_improvement:+.1%}")
    print(f"  - Average Reward: {reward_improvement:+.2f}")
    print(f"  - Average Ante: {ante_improvement:+.2f}")
    print(f"  - Average Blinds: {blinds_improvement:+.2f}")
    
    # Calculate percentage improvements
    if first_data['avg_reward'] != 0:
        reward_pct = (reward_improvement / abs(first_data['avg_reward'])) * 100
        print(f"  - Reward Improvement: {reward_pct:+.1f}%")
    
    if first_data['avg_blinds'] != 0:
        blinds_pct = (blinds_improvement / first_data['avg_blinds']) * 100
        print(f"  - Blinds Improvement: {blinds_pct:+.1f}%")
    
    # Save detailed comparison report
    report_path = "./data/results/phased_training_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Phased Training Report (50 minutes)\n\n")
        
        f.write("## Phase-by-Phase Results\n")
        f.write("| Phase | Win Rate | Avg Reward | Avg Ante | Avg Blinds |\n")
        f.write("|-------|----------|------------|----------|------------|\n")
        for phase_name, results in phase_results:
            f.write(f"| {phase_name} | {results['win_rate']:.1%} | "
                   f"{results['avg_reward']:.2f} | {results['avg_ante']:.2f} | "
                   f"{results['avg_blinds']:.2f} |\n")
        
        f.write(f"\n## Improvements\n")
        f.write(f"- Win Rate: {win_rate_improvement:+.1%}\n")
        f.write(f"- Average Reward: {reward_improvement:+.2f}\n")
        f.write(f"- Average Ante: {ante_improvement:+.2f}\n")
        f.write(f"- Average Blinds: {blinds_improvement:+.2f}\n")
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    print("=" * 70)
    print("🎉 Phased training completed successfully!")


def curriculum_training():
    main_config = TrainingConfig()
    model_config = ModelConfig()
    # Forza wandb disattivato per evitare errori nei test/Windows
    main_config.use_wandb = False

    if not main_config.curriculum_enabled:
        print("Curriculum learning is disabled in config. Running single training session.")
        train_agent(main_config, model_config)
        return
        
    stages = main_config.curriculum_stages
    
    model = None
    vec_env_stats_path = None
    
    for i, stage in enumerate(stages):
        print(f"\n=== Curriculum Stage {i+1}: Max Ante {stage['max_ante']} ({stage['timesteps']} timesteps) ===")
        
        stage_config = TrainingConfig()
        stage_config.max_ante = stage['max_ante']
        stage_config.total_timesteps = stage['timesteps']
        stage_config.wandb_project = f"{main_config.wandb_project}-stage-{i+1}"
        stage_config.log_dir = os.path.join(main_config.log_dir, f"stage_{i+1}")
        stage_config.model_dir = os.path.join(main_config.model_dir, f"stage_{i+1}")
        stage_config.use_wandb = False  # Forza wandb disattivato per ogni stage
        os.makedirs(stage_config.log_dir, exist_ok=True)
        os.makedirs(stage_config.model_dir, exist_ok=True)

        # Non chiamare mai setup_wandb o wandb.finish

        model, env, final_callback, balatro_callback = train_agent(stage_config, model_config, previous_model=model, previous_vec_env=vec_env_stats_path)
        
        stage_model_path = os.path.join(stage_config.model_dir, "final_stage_model.zip")
        model.save(stage_model_path)
        print(f"✅ Stage {i+1} completed and saved to {stage_model_path}")
        # Salva VecNormalize per continuità tra stage
        stage_vecnorm_path = os.path.join(stage_config.model_dir, "vecnormalize.pkl")
        env.save(stage_vecnorm_path)
        vec_env_stats_path = stage_vecnorm_path

    print("=" * 70)
    print("🎯 Training completato! Generazione grafico finale...")
    
    # Generate final training plot if we have a callback with data
    try:
        # Use the BalatroCallback with real training data for detailed plot
        print("📊 Generazione grafico di progressione finale...")
        
        # Force generation of final training plot with real data
        if hasattr(balatro_callback, 'episode_rewards') and balatro_callback.episode_rewards:
            print(f"📊 Dati trovati: {len(balatro_callback.episode_rewards)} episodi registrati")
            balatro_callback._on_training_end()
        elif hasattr(final_callback, 'total_hands') and final_callback.total_hands > 0:
            # Fallback: Create a basic completion plot
            import matplotlib.pyplot as plt
            
            os.makedirs("./data/plots/", exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'🎉 Training Completato!\n\n'
                              f'Totale mani giocate: {final_callback.total_hands:,}\n'
                              f'Curriculum stages completati: {len(stages)}\n'
                              f'Modello finale salvato!', 
                     ha='center', va='center', fontsize=14, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Balatro RL Training - Completamento', fontsize=16, fontweight='bold')
            
            plot_path = "./data/plots/training_completion.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Grafico di completamento salvato in: {plot_path}")
        else:
            print("📊 Nessun dato disponibile per il grafico")
        
    except Exception as e:
        print(f"⚠️ Impossibile generare grafico finale: {e}")
    
    # Genera automaticamente i grafici dettagliati
    print("\n📊 Generazione grafici dettagliati...")
    try:
        import sys
        import subprocess
        
        # Esegui lo script di generazione grafici
        script_path = os.path.join(os.path.dirname(__file__), '..', 'generate_detailed_plots.py')
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode == 0:
            print("✅ Grafici dettagliati generati con successo!")
            print(result.stdout)
        else:
            print(f"⚠️ Errore nella generazione grafici: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Errore nell'esecuzione automatica dei grafici: {e}")
        print("Puoi eseguire manualmente: python generate_detailed_plots.py")
        print("📊 Training completato comunque con successo!")

    if main_config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    print("🚀 Starting Balatro RL Training...")
    print("=" * 50)
    # Load configurations
    config = TrainingConfig("configs/training_config.yaml")
    model_config = ModelConfig("configs/model_config.yaml")
    
    print(f"Training Configuration:")
    print(f"  - Total timesteps: {config.total_timesteps:,}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Number of environments: {config.n_envs}")
    print(f"  - Device: {config.device}")
    print(f"  - Use WandB: {config.use_wandb}")
    
    print(f"\nModel Configuration:")
    print(f"  - Network architecture: {model_config.net_arch}")
    print(f"  - Policy type: {model_config.policy_type}")
    print(f"  - Normalize observations: {model_config.normalize_obs}")
    
    print("\n" + "=" * 50)
    
    # CURRICULUM TRAINING: esegui curriculum se abilitato, altrimenti training standard
    if config.curriculum_enabled:
        print("🎯 Starting Curriculum Training...")
        curriculum_training()
    else:
        print("🎯 Starting Single Training Session...")
        model, env = train_agent(config, model_config)
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved to: {config.model_dir}/final_model.zip")
        print(f"📊 Logs saved to: {config.log_dir}")

    # Report dettagliato solo se training standard (non curriculum)
    if not config.curriculum_enabled:
        print(f"\n📊 Report sulle ultime 5000 mani giocate (dopo training):")
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize
            
            final_model_path = f"{config.model_dir}/final_model.zip"
            vecnorm_path = f"{config.model_dir}/vecnormalize.pkl"
            
            # Controlla se i file esistono
            if not os.path.exists(final_model_path):
                print(f"❌ Modello non trovato: {final_model_path}")
                print("🎉 Training session completed!")
            else:
                def make_env():
                    env = BalatroEnv(max_ante=config.max_ante,
                                     starting_money=config.starting_money,
                                     hand_size=config.hand_size,
                                     max_jokers=config.max_jokers)
                    env = Monitor(env, config.log_dir)
                    return env
                
                eval_env = make_vec_env(make_env, n_envs=1)
                
                # Carica VecNormalize solo se il file esiste e normalize_obs è True
                if model_config.normalize_obs and os.path.exists(vecnorm_path):
                    eval_env = VecNormalize.load(vecnorm_path, eval_env)
                    print(f"✅ VecNormalize caricato da {vecnorm_path}")
                elif model_config.normalize_obs:
                    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=model_config.clip_obs)
                    print(f"✅ VecNormalize creato nuovo (normalize_obs=True)")
                else:
                    print(f"✅ Nessuna normalizzazione applicata")
                
                model = PPO.load(final_model_path, env=eval_env, device=config.device)
                
                obs = eval_env.reset()
                done = False
                hand_types = []
                hand_infos = []
                step_count = 0
                while not done and len(hand_types) < 5000:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    done = done[0]
                    current_info = info[0] if isinstance(info, list) else info
                    step_count += 1
                    if current_info.get('hand_type'):
                        hand_types.append(current_info['hand_type'])
                        hand_infos.append(current_info)
                
                from collections import Counter
                hand_counter = Counter(hand_types)
                print(f"Totale mani raccolte: {len(hand_types)} su {step_count} step")
                print(f"Ratio mani giocate: {len(hand_types) / max(1, step_count):.4f}")
                print(f"Mani più comuni (top 10): {hand_counter.most_common(10)}")
                # Statistiche aggiuntive sulle ultime 5000 mani
                blinds_beaten = sum(1 for info in hand_infos if info.get('blind_beaten', False))
                ante_progression = [(info.get('ante', None), info.get('blind', None)) for info in hand_infos if info.get('blind_beaten', False)]
                print(f"Blinds beaten: {blinds_beaten}")
                print(f"Ante progression (solo blinds battuti): {ante_progression[-10:]}")
                eval_env.close()
        except Exception as e:
            print(f"❌ Errore nel report finale: {e}")
            print("Il training è completato comunque.")
    
    print("\n🎉 Training session completed!")
    
    # Genera automaticamente i grafici dettagliati
    print("\n📊 Generazione grafici dettagliati...")
    try:
        import sys
        import subprocess
        
        # Esegui lo script di generazione grafici
        script_path = os.path.join(os.path.dirname(__file__), '..', 'generate_detailed_plots.py')
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode == 0:
            print("✅ Grafici dettagliati generati con successo!")
            print(result.stdout)
        else:
            print(f"⚠️ Errore nella generazione grafici: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Errore nell'esecuzione automatica dei grafici: {e}")
        print("Puoi eseguire manualmente: python generate_detailed_plots.py")
