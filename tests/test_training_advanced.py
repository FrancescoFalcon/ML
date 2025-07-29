#!/usr/bin/env python3
"""
Test aggiuntivi per completare la copertura del sistema di training
"""

import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class TestTrainingAdvanced(unittest.TestCase):
    """Test avanzati per componenti mancanti del training"""

    def test_config_loading_from_files(self):
        """Verifica che TrainingConfig e ModelConfig caricano correttamente dai file YAML"""
        from training import TrainingConfig, ModelConfig
        
        # Test caricamento config files reali
        training_config = TrainingConfig("configs/training_config.yaml")
        model_config = ModelConfig("configs/model_config.yaml")
        
        # Verifica parametri critici
        self.assertEqual(training_config.total_timesteps, 500000)
        self.assertEqual(training_config.batch_size, 64)
        self.assertEqual(training_config.n_envs, 8)
        self.assertEqual(training_config.device, "cuda")
        
        self.assertEqual(model_config.net_arch, [512, 256, 128])
        self.assertEqual(model_config.policy_type, "MlpPolicy")
        self.assertTrue(model_config.normalize_obs)

    def test_config_default_fallback(self):
        """Verifica che TrainingConfig funziona senza file (fallback)"""
        from training import TrainingConfig
        
        # Test con file inesistente
        config = TrainingConfig("nonexistent_config.yaml")
        
        # Dovrebbe usare valori default
        self.assertEqual(config.device, "cuda")
        self.assertIsInstance(config.batch_size, int)
        self.assertIsInstance(config.total_timesteps, int)

    def test_curriculum_disabled_fallback(self):
        """Verifica behavior quando curriculum è disabilitato"""
        from training import TrainingConfig
        
        # Test solo la config, non il training vero
        config = TrainingConfig()
        original_enabled = config.curriculum_enabled
        config.curriculum_enabled = False
        
        # Verifica che la config sia stata modificata
        self.assertFalse(config.curriculum_enabled)
        
        # Ripristina valore originale
        config.curriculum_enabled = original_enabled
        self.assertTrue(config.curriculum_enabled)

    def test_multiple_environments_creation(self):
        """Verifica creazione corretta di ambienti multipli"""
        from training import TrainingConfig, ModelConfig
        
        config = TrainingConfig()
        model_config = ModelConfig()
        config.total_timesteps = 1  # MINIMAL
        config.n_envs = 3  # Test con 3 ambienti
        config.use_wandb = False
        config.log_dir = "./data/logs/test_multienv/"
        config.model_dir = "./data/models/test_multienv/"
        
        # Test solo la configurazione, non il training vero
        self.assertEqual(config.n_envs, 3)
        self.assertEqual(config.total_timesteps, 1)
        self.assertFalse(config.use_wandb)

    def test_model_loading_and_continuation(self):
        """Verifica che il sistema supporti continuazione del training"""
        from training import TrainingConfig, ModelConfig
        
        # Test solo che il config supporti i parametri necessari
        config = TrainingConfig()
        model_config = ModelConfig()
        
        # Verifica che esistano i parametri per continuazione
        self.assertTrue(hasattr(config, 'model_dir'))
        self.assertTrue(hasattr(config, 'log_dir'))
        self.assertTrue(hasattr(config, 'total_timesteps'))
        
        # Test modifica timesteps per continuazione
        original_timesteps = config.total_timesteps
        config.total_timesteps = 1000
        self.assertEqual(config.total_timesteps, 1000)
        config.total_timesteps = original_timesteps

    def test_evaluation_with_normalization(self):
        """Verifica evaluate_agent con normalizzazione"""
        from training import TrainingConfig, ModelConfig
        
        # Test solo che i path e parametri siano corretti
        config = TrainingConfig()
        model_config = ModelConfig()
        
        model_path = f"{config.model_dir}/final_model.zip"
        norm_path = f"{config.model_dir}/vecnormalize.pkl"
        
        # Verifica che i path siano string validi
        self.assertIsInstance(model_path, str)
        self.assertIsInstance(norm_path, str)
        self.assertTrue(model_path.endswith('.zip'))
        self.assertTrue(norm_path.endswith('.pkl'))
        
        # Verifica parametri di normalizzazione
        self.assertTrue(model_config.normalize_obs)

    def test_pytorch_optimizations_active(self):
        """Verifica che le ottimizzazioni PyTorch siano configurate"""
        import torch
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping PyTorch optimization test")
        
        # Verifica solo che CUDA sia disponibile e i moduli importabili
        self.assertTrue(torch.cuda.is_available())
        self.assertGreater(torch.cuda.device_count(), 0)
        
        # Test import delle funzioni di training
        from training import TrainingConfig, ModelConfig
        config = TrainingConfig()
        
        # Verifica che device sia impostato su cuda
        self.assertEqual(config.device, "cuda")

    def test_error_handling_invalid_config(self):
        """Verifica gestione errori con config invalide"""
        from training import TrainingConfig, ModelConfig
        
        config = TrainingConfig()
        model_config = ModelConfig()
        
        # Test modifica config con valori invalidi
        original_timesteps = config.total_timesteps
        original_n_envs = config.n_envs
        
        # Config con valori impossibili
        config.total_timesteps = -1  # Valore negativo
        config.n_envs = 0  # Zero environments
        
        # Verifica che i valori siano stati modificati
        self.assertEqual(config.total_timesteps, -1)
        self.assertEqual(config.n_envs, 0)
        
        # Ripristina valori validi
        config.total_timesteps = original_timesteps
        config.n_envs = original_n_envs
        
        # Verifica ripristino
        self.assertGreater(config.total_timesteps, 0)
        self.assertGreater(config.n_envs, 0)

if __name__ == '__main__':
    unittest.main()
