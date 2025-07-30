# 🎮 Balatro RL Project - Guida Installazione

Questo progetto implementa un agente di reinforcement learning per giocare a Balatro usando PPO (Proximal Policy Optimization).

## 📋 Requisiti di Sistema

- **Python 3.8+** (raccomandato 3.9 o 3.10)
- **8GB RAM** (minimo 4GB)
- **10GB spazio disco** (per dati e modelli)

## 🚀 Installazione Rapida

### Windows
1. **Scarica e installa Python** da [python.org](https://www.python.org/downloads/)
   - ✅ Seleziona "Add Python to PATH" durante l'installazione
2. **Doppio click** su `install.bat`
3. **Segui le istruzioni** del menu

### Linux/macOS
1. **Installa Python 3.8+** (se non presente):
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install python3 python3-pip
   
   # macOS (con Homebrew)
   brew install python3
   ```
2. **Rendi eseguibile e lancia**:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

### Installazione Manuale
```bash
# 1. Aggiorna pip
python -m pip install --upgrade pip

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Setup progetto (opzionale)
python setup.py
```

## 📦 Tipi di Installazione

### 🔹 Base (Raccomandato)
```bash
python setup.py
```
- Include solo le dipendenze essenziali
- Ideale per training e utilizzo normale

### 🔹 GPU (Per NVIDIA)
```bash
python setup.py --gpu
```
- Include PyTorch con supporto CUDA
- Training 3-5x più veloce
- Richiede driver NVIDIA aggiornati

### 🔹 Sviluppo
```bash
python setup.py --dev
```
- Include strumenti di sviluppo
- Jupyter, pytest, linting, formatting
- Per chi vuole modificare il codice

### 🔹 Completa
```bash
python setup.py --all
```
- Include tutto: GPU + sviluppo
- Installazione più pesante ma completa

## 🎯 Verifica Installazione

**Test rapido:**
```bash
python tests/test_environment.py
```

**Training di prova:**
```bash
python src/training.py
```

**Generazione grafici:**
```bash
python generate_detailed_plots.py
```

## 📁 Struttura Progetto

```
ML/
├── setup/                  # Tutti i file di installazione e documentazione
│   ├── requirements.txt    # Dipendenze Python
│   ├── setup.py            # Script setup automatico
│   ├── install.bat         # Installazione Windows
│   ├── install.sh          # Installazione Linux/macOS
│   ├── INSTALL.md          # Guida installazione dettagliata
│   ├── README.md           # Documentazione dettagliata
│   └── README_TLDR.md      # Versione sintetica
├── src/
│   ├── balatro_env.py      # Ambiente di gioco
│   ├── training.py         # Script training principale
│   ├── utils.py            # Utilità e metriche
│   └── callbacks/
│       └── balatro_callbacks.py
├── configs/
│   ├── training_config.yaml
│   └── model_config.yaml
├── tests/
│   ├── test_environment.py
│   ├── test_training.py
│   └── test_jokers.py
├── data/                   # Creata automaticamente
│   ├── logs/               # Log training
│   ├── models/             # Modelli salvati
│   ├── plots/              # Grafici generati
│   └── results/            # Risultati analisi
└── generate_detailed_plots.py
```

## 🎮 Come Usare

### Training Base
```bash
python src/training.py
```

### Training con Configurazione Custom
```bash
# Modifica configs/training_config.yaml
python src/training.py
```

### Analisi Risultati
```bash
python generate_detailed_plots.py
```

### Test Ambiente
```bash
python -m pytest tests/
```

## 🔧 Configurazione

### File `configs/training_config.yaml`:
- **total_timesteps**: Durata training (default: 25,000)
- **learning_rate**: Velocità apprendimento (default: 0.00008)
- **n_envs**: Ambienti paralleli (default: 4)
- **max_ante**: Obiettivo massimo ante (default: 8)

### File `configs/model_config.yaml`:
- Architettura rete neurale
- Parametri PPO avanzati

## 🐛 Risoluzione Problemi

### Python non trovato
```bash
# Verifica installazione
python --version
# o
python3 --version
```

### Errori pip
```bash
# Aggiorna pip
python -m pip install --upgrade pip

# Reinstalla dipendenze
pip install -r requirements.txt --force-reinstall
```

### Errori GPU
```bash
# Verifica driver NVIDIA
nvidia-smi

# Reinstalla PyTorch GPU
pip uninstall torch
python setup.py --gpu
```

### Errori spazio disco
- Elimina file temporanei in `data/logs/`
- Comprimi o sposta modelli in `data/models/`

### Errori memoria
- Riduci `n_envs` in `training_config.yaml`
- Riduci `batch_size` 

## 📈 Performance Attese
### Training Base (CPU)(Risultati per env utilizzato):
- **Velocità**: ~5,000 timesteps/minuto
- **Durata**: 25-45 minuti per training completo


### Risultati Tipici:
- **Ante 2**: Raggiunto dopo ~10,000 timesteps
- **Ante 3**: Raggiunto dopo ~20,000 timesteps
- **Ante 4+**: Dipende dalla configurazione


**Per problemi avanzati:**
- Controlla log in `data/logs/`
- Verifica configurazione in `configs/`
- Esegui test: `python -m pytest tests/ -v`

---

