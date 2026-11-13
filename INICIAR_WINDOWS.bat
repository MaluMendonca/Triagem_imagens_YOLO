@echo off
chcp 65001 >NUL
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Preparando ambiente Python local...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Falha ao criar ambiente. Instale Python 3.11 ou 3.12 e marque Add Python to PATH.
        pause
        exit /b 1
    )
)
echo Verificando bibliotecas...
".venv\Scripts\python.exe" -m pip install -q -r requirements_windows.txt
if errorlevel 1 (
    echo Falha ao instalar dependencias. Verifique sua conexao.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" triagem_windows.py
if errorlevel 1 pause
