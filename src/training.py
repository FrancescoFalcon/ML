
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

# Disable wandb symlink creation on Windows to avoid OSError
if platform.system() == "Windows":
    os.environ["WANDB_SYMLINK_DISABLE"] = "true"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.balatro_env import BalatroEnv
from src.utils import load_config, BalatroMetrics, plot_training_progress, create_performance_report
from src.callbacks.balatro_callbacks import BalatroCallback

class TrainingConfig:
    def __init__(self, config_path="configs/training_config.yaml"):
        config = load_config(config_path)
        
        self.n_envs = int(config['training']['n_envs'])
        self.max_ante = int(config['environment']['max_ante'])
        self.starting_money = float(config['environment']['starting_money'])
        self.hand_size = int(config['environment'].get('hand_size', 8))
        self.max_jokers = int(config['environment'].get('max_jokers', 5))

        self.algorithm = config['training']['algorithm']
        self.total_timesteps = int(config['training']['total_timesteps'])
        self.learning_rate = float(config['training']['learning_rate'])
        self.batch_size = int(config['training']['batch_size'])
        self.n_steps = int(config['training']['n_steps'])
        self.gamma = float(config['training']['gamma'])
        self.gae_lambda = float(config['training']['gae_lambda'])
        self.clip_range = float(config['training']['clip_range'])
        self.ent_coef = float(config['training']['ent_coef'])
        self.vf_coef = float(config['training']['vf_coef'])
        self.max_grad_norm = float(config['training']['max_grad_norm'])

        self.eval_freq = int(config['evaluation']['eval_freq'])
        self.n_eval_episodes = int(config['evaluation']['n_eval_episodes'])
        self.deterministic_eval = bool(config['evaluation']['deterministic'])

        self.log_dir = config['logging']['log_dir']
        self.model_dir = config['logging']['model_dir']
        self.use_wandb = bool(config['logging']['use_wandb'])
        self.wandb_project = config['logging']['wandb_project']
        self.wandb_entity = config['logging'].get('wandb_entity', None)
        self.save_freq = int(config['logging'].get('save_freq', 50000))

        self.curriculum_enabled = bool(config['curriculum']['enabled'])
        self.curriculum_stages = config['curriculum']['stages']


class ModelConfig:
    def __init__(self, config_path="configs/model_config.yaml"):
        config = load_config(config_path)

        self.policy_type = config['network']['policy_type']
        self.net_arch = [int(x) for x in config['network']['net_arch']]
        self.activation_fn = config['network']['activation_fn']
        self.ortho_init = bool(config['network']['ortho_init'])

        self.normalize_obs = bool(config['features']['normalize_obs'])
        self.normalize_reward = bool(config['features']['normalize_reward'])
        self.clip_obs = float(config['features']['clip_obs'])

        self.use_sde = bool(config['advanced']['use_sde'])
        self.sde_sample_freq = int(config['advanced']['sde_sample_freq'])
        self.target_kl = (float(config['advanced']['target_kl']) if config['advanced']['target_kl'] is not None else None)

        # Ensure all reward weights are float
        self.reward_weights = {k: float(v) for k, v in config['reward_function']['weights'].items()}

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
    import torch
    print(f"[INFO] PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("[INFO] Using CPU only")
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Forza wandb disattivato per evitare errori su Windows/test
    config.use_wandb = False
    # Non chiamare mai setup_wandb se wandb è disattivato
    # if wandb.run is None:
    #     setup_wandb(config)
    
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

    # Always add EvalCallback to save best model, regardless of wandb
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
    
    print(f"\nStarting evaluation for {n_episodes} episodes...")
    # Helper functions for robust attribute access
    def get_env_attr(attr, default=None):
        # Try VecEnv
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'envs'):
            base = env.unwrapped.envs[0]
        elif hasattr(env, 'envs'):
            base = env.envs[0]
        elif hasattr(env, 'unwrapped'):
            base = env.unwrapped
        else:
            base = env
        return getattr(base, attr, default)

    def call_env_method(method, default=None):
        # Try VecEnv
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'envs'):
            base = env.unwrapped.envs[0]
        elif hasattr(env, 'envs'):
            base = env.envs[0]
        elif hasattr(env, 'unwrapped'):
            base = env.unwrapped
        else:
            base = env
        return getattr(base, method, lambda: default)()

    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        episode_reward = 0
        done = False
        truncated = False
        episode_hand_types = []
        # Robustly get jokers for both VecEnv and non-VecEnv
        try:
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'envs'):
                # VecEnv
                episode_jokers = [j.joker_type.value for j in env.unwrapped.envs[0].jokers]
            elif hasattr(env, 'jokers'):
                episode_jokers = [j.joker_type.value for j in env.jokers]
            else:
                episode_jokers = []
        except Exception:
            episode_jokers = []

        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            # Handle both 4-value and 5-value returns
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                raise ValueError(f"Unexpected number of values returned from env.step: {len(step_result)}")
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            current_info = info[0] if isinstance(info, (list, tuple)) else info
            if current_info.get('action_type') == 'play' and 'hand_type' in current_info:
                episode_hand_types.append(current_info['hand_type'])

        metrics_tracker.add_episode(
            reward=episode_reward,
            ante=get_env_attr('current_ante', 0),
            blinds=call_env_method('_get_total_blinds_beaten', 0),
            hand_types=episode_hand_types,
            jokers=episode_jokers,
            score=get_env_attr('score', 0)
        )
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Ante={get_env_attr('current_ante', 0)}, Blinds Beaten={call_env_method('_get_total_blinds_beaten', 0)}")
    
    stats = metrics_tracker.get_statistics()
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
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

        model, env = train_agent(stage_config, model_config, previous_model=model, previous_vec_env=vec_env_stats_path)
        
        stage_model_path = os.path.join(stage_config.model_dir, "final_stage_model.zip")
        stage_vecnorm_path = os.path.join(stage_config.model_dir, "vecnormalize.pkl")
        model.save(stage_model_path)
        env.save(stage_vecnorm_path)
        vec_env_stats_path = stage_vecnorm_path

        print(f"Evaluating Stage {i+1} model...")
        results = evaluate_agent(stage_model_path, n_episodes=stage_config.n_eval_episodes, normalize_path=stage_vecnorm_path)
        print(f"Stage {i+1} Results: Win Rate = {results['win_rate']:.1%}, Avg Ante = {results['avg_ante']:.2f}")

    if main_config.use_wandb:
        wandb.finish()
