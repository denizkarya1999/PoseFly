@echo off
REM PoseFly launcher (no console)
REM Put this file next to main.py, or adjust the cd path below.

setlocal

REM --- go to the folder where this .bat lives ---
cd /d "%~dp0"

REM --- launch using pythonw (no terminal window) ---
REM If pythonw isn't found, use the full path to pythonw.exe (see note below).
start "" pythonw.exe main.py

endlocal