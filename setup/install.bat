@echo off
REM =====================================
REM Balatro RL Project - Installazione Automatica Windows
REM =====================================
REM 
REM Questo script installa automaticamente tutte le dipendenze
REM necessarie per il progetto Balatro RL su Windows.
REM
REM Uso: Doppio click su install.bat oppure esegui da cmd
REM

echo.
echo ========================================
echo  BALATRO RL PROJECT - INSTALLAZIONE
echo ========================================
echo.

REM Verifica che Python sia installato
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato!
    echo.
    echo Installa Python da: https://www.python.org/downloads/
    echo Assicurati di aggiungere Python al PATH durante l'installazione
    echo.
    pause
    exit /b 1
)

echo Python trovato:
python --version
echo.

REM Chiedi all'utente che tipo di installazione vuole
echo Scegli il tipo di installazione:
echo.
echo [1] Base - Solo dipendenze necessarie (Raccomandato)
echo [2] GPU - Include supporto GPU NVIDIA
echo [3] Sviluppo - Include strumenti per sviluppatori
echo [4] Completa - Include tutto
echo.
set /p choice="Inserisci la tua scelta (1-4): "

REM Esegui setup con opzioni appropriate
if "%choice%"=="1" (
    echo.
    echo Installazione BASE in corso...
    python setup.py
) else if "%choice%"=="2" (
    echo.
    echo Installazione GPU in corso...
    python setup.py --gpu
) else if "%choice%"=="3" (
    echo.
    echo Installazione SVILUPPO in corso...
    python setup.py --dev
) else if "%choice%"=="4" (
    echo.
    echo Installazione COMPLETA in corso...
    python setup.py --all
) else (
    echo.
    echo Scelta non valida. Eseguo installazione base...
    python setup.py
)

echo.
if errorlevel 1 (
    echo.
    echo ========================================
    echo  INSTALLAZIONE FALLITA
    echo ========================================
    echo.
    echo Controlla i messaggi di errore sopra e riprova.
    echo.
    echo Soluzioni comuni:
    echo - Assicurati di avere Python 3.8+
    echo - Esegui come amministratore se necessario
    echo - Verifica connessione internet
    echo.
) else (
    echo.
    echo ========================================
    echo  INSTALLAZIONE COMPLETATA CON SUCCESSO!
    echo ========================================
    echo.
    echo Per iniziare:
    echo   python src/training.py
    echo.
    echo Per test:
    echo   python tests/test_environment.py
    echo.
    echo Per grafici:
    echo   python generate_detailed_plots.py
    echo.
)

echo Premi un tasto per chiudere...
pause >nul
