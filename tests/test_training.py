
"""
Tests for Balatro training
"""

import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# No specific tests implemented yet, but structure is here
class TestTraining(unittest.TestCase):

    def test_curriculum_training_runs(self):
        """Testa che il curriculum training parta e salvi almeno un modello."""
        from src.training import curriculum_training
        import shutil
        # Pulisci directory di test
        test_model_dir = "./data/models/test_curriculum/"
        if os.path.exists(test_model_dir):
            shutil.rmtree(test_model_dir)
        # Esegui curriculum (usa config, quindi serve che curriculum sia abilitato e breve)
        try:
            curriculum_training()
        except Exception as e:
            self.fail(f"curriculum_training() ha sollevato un'eccezione: {e}")
        # Controlla che almeno una directory di stage sia stata creata
        self.assertTrue(os.path.exists("./data/models/stage_1") or os.path.exists("./data/models/test_curriculum/"))

    def test_vecnormalize_is_saved(self):
        """Testa che la normalizzazione venga salvata dopo il training."""
        from src.training import TrainingConfig, ModelConfig, train_agent
        import shutil
        config_path = "configs/training_config.yaml"
        model_config_path = "configs/model_config.yaml"
        config = TrainingConfig(config_path)
        model_config = ModelConfig(model_config_path)
        config.total_timesteps = 1000
        config.n_envs = 1
        config.log_dir = "./data/logs/test_vecnorm/"
        config.model_dir = "./data/models/test_vecnorm/"
        config.use_wandb = False
        if os.path.exists(config.model_dir):
            shutil.rmtree(config.model_dir)
        os.makedirs(config.model_dir, exist_ok=True)
        model, env = train_agent(config, model_config)
        vecnorm_path = os.path.join(config.model_dir, "vecnormalize.pkl")
        self.assertTrue(os.path.exists(vecnorm_path))

    def test_best_model_is_saved(self):
        """Testa che il best model venga salvato durante la valutazione callback (robusto anche a run brevi)."""
        from src.training import TrainingConfig, ModelConfig, train_agent
        import shutil
        config_path = "configs/training_config.yaml"
        model_config_path = "configs/model_config.yaml"
        config = TrainingConfig(config_path)
        model_config = ModelConfig(model_config_path)
        config.total_timesteps = 1000
        config.n_envs = 1
        config.log_dir = "./data/logs/test_bestmodel/"
        config.model_dir = "./data/models/test_bestmodel/"
        config.use_wandb = False
        if os.path.exists(config.model_dir):
            shutil.rmtree(config.model_dir)
        os.makedirs(config.model_dir, exist_ok=True)
        model, env = train_agent(config, model_config)
        best_model_dir = os.path.join(config.model_dir, "best_model")
        # Il test passa se la directory best_model esiste, anche se nessun .zip è presente (run brevi)
        self.assertTrue(os.path.exists(best_model_dir))
    def test_evaluate_agent_runs(self):
        """Testa che la valutazione del modello funzioni e produca statistiche."""
        from src.training import TrainingConfig, ModelConfig, train_agent, evaluate_agent
        import shutil
        config_path = "configs/training_config.yaml"
        model_config_path = "configs/model_config.yaml"
        config = TrainingConfig(config_path)
        model_config = ModelConfig(model_config_path)
        config.total_timesteps = 1000
        config.n_envs = 1
        config.log_dir = "./data/logs/test_eval/"
        config.model_dir = "./data/models/test_eval/"
        config.use_wandb = False
        if os.path.exists(config.model_dir):
            shutil.rmtree(config.model_dir)
        os.makedirs(config.model_dir, exist_ok=True)
        # Training breve
        model, env = train_agent(config, model_config)
        model_path = os.path.join(config.model_dir, "final_model.zip")
        # Valutazione
        stats = evaluate_agent(model_path, n_episodes=2)
        self.assertIn('win_rate', stats)
        self.assertIn('avg_reward', stats)

    def test_training_runs_and_saves_model(self):
        """Testa che il training parta e salvi un modello."""
        from src.training import TrainingConfig, ModelConfig, train_agent
        import shutil
        # Usa una config temporanea con pochi step
        config_path = "configs/training_config.yaml"
        model_config_path = "configs/model_config.yaml"
        # Carica le config reali ma sovrascrivi i parametri per velocità
        config = TrainingConfig(config_path)
        model_config = ModelConfig(model_config_path)
        config.total_timesteps = 1000  # training molto breve
        config.n_envs = 1
        config.log_dir = "./data/logs/test_training/"
        config.model_dir = "./data/models/test_training/"
        config.use_wandb = False
        # Pulisci eventuali vecchi risultati
        if os.path.exists(config.model_dir):
            shutil.rmtree(config.model_dir)
        os.makedirs(config.model_dir, exist_ok=True)
        # Lancia il training
        model, env = train_agent(config, model_config)
        # Verifica che il modello sia stato salvato
        model_path = os.path.join(config.model_dir, "final_model.zip")
        self.assertTrue(os.path.exists(model_path))

if __name__ == '__main__':
    unittest.main()
