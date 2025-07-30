#!/bin/bash
# =====================================
# Balatro RL Project - Installazione Automatica Unix/Linux/macOS
# =====================================
# 
# Questo script installa automaticamente tutte le dipendenze
# necessarie per il progetto Balatro RL su sistemi Unix.
#
# Uso: ./install.sh
#      oppure: bash install.sh
#

set -e  # Esci se c'è un errore

echo ""
echo "========================================"
echo " BALATRO RL PROJECT - INSTALLAZIONE"
echo "========================================"
echo ""

# Verifica che Python sia installato
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ ERRORE: Python non trovato!"
        echo ""
        echo "Installa Python 3.8+ dal tuo package manager:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
        echo "  macOS:         brew install python3"
        echo ""
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python trovato:"
$PYTHON_CMD --version
echo ""

# Verifica versione Python
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MIN_VERSION="3.8"

if [[ $(echo "$PYTHON_VERSION $MIN_VERSION" | awk '{print ($1 < $2)}') == 1 ]]; then
    echo "❌ ERRORE: Python $MIN_VERSION+ richiesto"
    echo "   Versione attuale: $PYTHON_VERSION"
    echo ""
    exit 1
fi

# Verifica che pip sia installato
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "❌ ERRORE: pip non trovato!"
    echo ""
    echo "Installa pip:"
    echo "  Ubuntu/Debian: sudo apt install python3-pip"
    echo "  CentOS/RHEL:   sudo yum install python3-pip"
    echo "  macOS:         brew install python3"
    echo ""
    exit 1
fi

# Menu di scelta
echo "Scegli il tipo di installazione:"
echo ""
echo "[1] Base - Solo dipendenze necessarie (Raccomandato)"
echo "[2] GPU - Include supporto GPU NVIDIA"
echo "[3] Sviluppo - Include strumenti per sviluppatori" 
echo "[4] Completa - Include tutto"
echo ""

read -p "Inserisci la tua scelta (1-4): " choice

# Esegui setup con opzioni appropriate
case $choice in
    1)
        echo ""
        echo "🚀 Installazione BASE in corso..."
        $PYTHON_CMD setup.py
        ;;
    2)
        echo ""
        echo "🚀 Installazione GPU in corso..."
        $PYTHON_CMD setup.py --gpu
        ;;
    3)
        echo ""
        echo "🚀 Installazione SVILUPPO in corso..."
        $PYTHON_CMD setup.py --dev
        ;;
    4)
        echo ""
        echo "🚀 Installazione COMPLETA in corso..."
        $PYTHON_CMD setup.py --all
        ;;
    *)
        echo ""
        echo "⚠️  Scelta non valida. Eseguo installazione base..."
        $PYTHON_CMD setup.py
        ;;
esac

# Controlla se l'installazione è riuscita
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo " ✅ INSTALLAZIONE COMPLETATA CON SUCCESSO!"
    echo "========================================"
    echo ""
    echo "Per iniziare:"
    echo "  $PYTHON_CMD src/training.py"
    echo ""
    echo "Per test:"
    echo "  $PYTHON_CMD tests/test_environment.py"
    echo ""
    echo "Per grafici:"
    echo "  $PYTHON_CMD generate_detailed_plots.py"
    echo ""
    echo "Per rendere eseguibili gli script:"
    echo "  chmod +x *.sh"
    echo ""
else
    echo ""
    echo "========================================"
    echo " ❌ INSTALLAZIONE FALLITA"
    echo "========================================"
    echo ""
    echo "Controlla i messaggi di errore sopra e riprova."
    echo ""
    echo "Soluzioni comuni:"
    echo "- Assicurati di avere Python 3.8+"
    echo "- Verifica connessione internet"
    echo "- Prova: pip install --upgrade pip"
    echo ""
    exit 1
fi
