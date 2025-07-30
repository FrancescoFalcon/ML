#!/usr/bin/env python3
"""
Setup script per il progetto Balatro RL
========================================

Questo script installa automaticamente tutte le dipendenze necessarie
per eseguire il progetto di reinforcement learning per Balatro.

Uso:
    python setup.py

Oppure per installazione avanzata:
    python setup.py --dev    # Include dipendenze per sviluppo
    python setup.py --gpu    # Include PyTorch con supporto GPU
    python setup.py --all    # Include tutto
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

class BalatroSetup:
    def __init__(self):
        self.python_version = sys.version_info
        self.os_name = platform.system().lower()
        self.project_root = Path(__file__).parent
        
    def check_python_version(self):
        """Verifica che la versione Python sia compatibile"""
        if self.python_version < (3, 8):
            print("❌ ERRORE: Python 3.8+ richiesto")
            print(f"   Versione attuale: {self.python_version.major}.{self.python_version.minor}")
            print("   Aggiorna Python prima di continuare")
            sys.exit(1)
        else:
            print(f"✅ Python {self.python_version.major}.{self.python_version.minor} compatibile")
    
    def upgrade_pip(self):
        """Aggiorna pip alla versione più recente"""
        print("\n📦 Aggiornamento pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True)
            print("✅ pip aggiornato con successo")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Avviso: Impossibile aggiornare pip: {e}")
    
    def install_requirements(self, gpu=False, dev=False):
        """Installa le dipendenze dal file requirements.txt"""
        print(f"\n📚 Installazione dipendenze...")
        
        # Installa requirements base
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                             check=True)
                print("✅ Dipendenze base installate")
            except subprocess.CalledProcessError as e:
                print(f"❌ Errore installazione dipendenze: {e}")
                return False
        else:
            print(f"❌ File requirements.txt non trovato in {req_file}")
            return False
        
        # Installa PyTorch con GPU se richiesto
        if gpu:
            self.install_pytorch_gpu()
        
        # Installa dipendenze sviluppo se richiesto
        if dev:
            self.install_dev_dependencies()
        
        return True
    
    def install_pytorch_gpu(self):
        """Installa PyTorch con supporto GPU"""
        print(f"\n🚀 Installazione PyTorch GPU...")
        
        if self.os_name == "windows":
            gpu_command = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            gpu_command = [
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ]
        
        try:
            subprocess.run(gpu_command, check=True)
            print("✅ PyTorch GPU installato")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Avviso: Installazione PyTorch GPU fallita: {e}")
            print("   Verrà usata la versione CPU")
    
    def install_dev_dependencies(self):
        """Installa dipendenze aggiuntive per sviluppo"""
        print(f"\n🛠️ Installazione dipendenze sviluppo...")
        
        dev_packages = [
            "jupyter",
            "ipython",
            "pytest-xdist",  # Test paralleli
            "black",         # Code formatter
            "flake8",        # Linter
            "mypy",          # Type checker
            "pre-commit"     # Git hooks
        ]
        
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + dev_packages, 
                         check=True)
            print("✅ Dipendenze sviluppo installate")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Avviso: Alcune dipendenze sviluppo non installate: {e}")
    
    def create_directories(self):
        """Crea le directory necessarie per il progetto"""
        print(f"\n📁 Creazione directory progetto...")
        
        directories = [
            "data/logs",
            "data/models", 
            "data/plots",
            "data/results",
            "data/checkpoints"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {dir_path}")
    
    def verify_installation(self):
        """Verifica che tutte le dipendenze critiche siano installate"""
        print(f"\n🔍 Verifica installazione...")
        
        critical_packages = [
            "numpy",
            "gymnasium", 
            "stable_baselines3",
            "torch",
            "matplotlib",
            "wandb",
            "yaml"
        ]
        
        failed_imports = []
        for package in critical_packages:
            try:
                if package == "yaml":
                    __import__("yaml")
                elif package == "stable_baselines3":
                    __import__("stable_baselines3")
                else:
                    __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package}")
                failed_imports.append(package)
        
        if failed_imports:
            print(f"\n❌ Pacchetti non installati: {', '.join(failed_imports)}")
            print("   Ripeti l'installazione o installa manualmente")
            return False
        else:
            print("\n🎉 Tutte le dipendenze critiche sono installate!")
            return True
    
    def print_usage_info(self):
        """Stampa informazioni su come usare il progetto"""
        print(f"\n" + "="*60)
        print("🎮 BALATRO RL PROJECT - SETUP COMPLETATO")
        print("="*60)
        print("\n📋 Come iniziare:")
        print("   1. Training base:")
        print("      python src/training.py")
        print()
        print("   2. Genera grafici dettagliati:")
        print("      python generate_detailed_plots.py")
        print()
        print("   3. Test ambiente:")
        print("      python -m pytest tests/")
        print()
        print("   4. Test rapido:")
        print("      python tests/test_environment.py")
        print()
        print("📁 Directory create:")
        print("   - data/logs/     → Log di training")
        print("   - data/models/   → Modelli salvati")
        print("   - data/plots/    → Grafici generati")
        print("   - data/results/  → Risultati analisi")
        print()
        print("📖 File di configurazione:")
        print("   - configs/training_config.yaml")
        print("   - configs/model_config.yaml")
        print()
        print("🚨 In caso di problemi:")
        print("   - Verifica versione Python (3.8+)")
        print("   - Reinstalla: pip install -r requirements.txt")
        print("   - GPU: python setup.py --gpu")
        print("="*60)

def main():
    # Parse argomenti
    args = sys.argv[1:]
    gpu = "--gpu" in args
    dev = "--dev" in args
    all_features = "--all" in args
    
    if all_features:
        gpu = dev = True
    
    print("🎮 BALATRO RL PROJECT SETUP")
    print("="*40)
    
    setup = BalatroSetup()
    
    # 1. Verifica Python
    setup.check_python_version()
    
    # 2. Aggiorna pip
    setup.upgrade_pip()
    
    # 3. Installa dipendenze
    if not setup.install_requirements(gpu=gpu, dev=dev):
        print("\n❌ Setup fallito durante installazione dipendenze")
        sys.exit(1)
    
    # 4. Crea directory
    setup.create_directories()
    
    # 5. Verifica installazione
    if not setup.verify_installation():
        print("\n⚠️  Setup completato con avvisi")
        sys.exit(1)
    
    # 6. Stampa info utilizzo
    setup.print_usage_info()
    
    print("\n🎉 Setup completato con successo!")

if __name__ == "__main__":
    main()
