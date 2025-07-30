# Balatro RL - Reinforcement Learning Poker Agent

## 📚 Overview
Balatro RL è un progetto di intelligenza artificiale che utilizza il Reinforcement Learning (RL) per insegnare a un agente a giocare a Balatro, un gioco ispirato al poker, con un sistema di shop, joker e progressione di difficoltà (ante/blind). L'obiettivo è sviluppare strategie ottimali per superare livelli sempre più difficili, gestendo risorse e sfruttando le meccaniche di gioco.

## 🏗️ Project Structure
```
ML/
├── configs/           # Configurazioni YAML per modello e training
├── data/              # Modelli, log, risultati, grafici
├── notebooks/         # Analisi, esplorazione, visualizzazione
├── src/               # Codice sorgente principale
│   ├── balatro_env.py         # Ambiente RL custom (Balatro)
│   ├── training.py            # Script di training principale
│   ├── evaluation.py          # Script di valutazione
│   ├── utils.py               # Utility varie
│   ├── callbacks/             # Callback custom per logging/statistiche
│   └── models/                # (Opzionale) Modelli custom
├── tests/             # Test automatici
├── requirements.txt   # Dipendenze Python
├── setup.py           # Installer automatico
├── install.bat/.sh    # Script di installazione Windows/Linux
├── INSTALL.md         # Guida installazione dettagliata
└── README_TLDR.md     # Versione sintetica di questo README
```

## 🚀 Installation
1. **Prerequisiti**: Python 3.9+ (consigliato 3.11), pip, ambiente virtuale consigliato
2. **Installazione automatica**:
   - Windows: `install.bat`
   - Linux/Mac: `install.sh`
   - Universale: `python setup.py`
3. **Manuale**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Problemi?** Consulta `INSTALL.md` per troubleshooting dettagliato.

## ⚙️ Configurazione
- **Training**: `configs/training_config.yaml` (timesteps, curriculum, frequenza salvataggi)
- **Modello**: `configs/model_config.yaml` (architettura, normalizzazione, PPO)
- **Ambiente**: Parametri ante, soldi iniziali, shop, anti-spam, reward shaping

## 🎮 Come Funziona
- **Ambiente RL**: Simula Balatro (carte, mani poker, blind, ante, shop, joker)
- **Agente**: Usa PPO (Stable-Baselines3) per apprendere strategie ottimali
- **Obiettivo**: Raggiungere ante/blind sempre più alti, gestendo risorse e rischi
- **Curriculum Learning**: Difficoltà crescente per facilitare l'apprendimento

## 🃏 Joker System
- **Joker Base**: Bonus semplici
- **Joker Avanzati**: Chip, Mult, Bonus, Wild, Lucky
- **Premium Joker**: Potenziamento massimo, costo elevato, effetto strategico
- **Shop**: Sistema di acquisto/refresh, economia bilanciata, penalità per acquisti inutili

## 🏆 Training & Metriche
- **Timesteps**: 25.000 (default, curriculum per ante)
- **Reward Shaping**: Premia progressione, mani ottimali, gestione shop
- **Logging**: Statistiche dettagliate su mani, joker, ante, reward, episode length
- **Visualizzazioni**: Grafici su progressione ante, distribuzione mani, acquisti joker, durata episodi
- **Monitoraggio**: Supporto per Weights & Biases (wandb) e Tensorboard

## 📊 Output & Analisi
- **Grafici**: `data/plots/` (progressione ante, reward, distribuzione mani)
- **Statistiche**: `data/results/training_statistics.txt`
- **Modelli**: `data/models/` (checkpoint e best model)
- **Log**: `data/logs/` (monitor.csv per ogni stage)

## 🧪 Testing
- Test automatici in `tests/` (Pytest)
- Esegui: `pytest tests/`

## 📝 Esempio di Training
```bash
python src/training.py --config configs/training_config.yaml
```

## 🔍 Troubleshooting
- **Problemi di dipendenze**: Usa `setup.py` o script installazione
- **Crash/instabilità**: Verifica versione Python, aggiorna pip, consulta `INSTALL.md`
- **Training lento**: Riduci timesteps/config
- **Reward/metriche anomale**: Controlla normalizzazione reward, curriculum, parametri PPO


## 👨‍💻 Credits
- Sviluppo: Francesco Falcon
- Basato su Stable-Baselines3, Gymnasium, PyTorch
- Ispirato a Balatro (gioco originale)

---
Per dettagli avanzati, troubleshooting approfondito e domande frequenti, consulta `INSTALL.md` e la documentazione nei file sorgente.
