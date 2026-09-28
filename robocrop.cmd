@echo off
rem robocrop - Windows launcher; does what ./robocrop does on macOS and Linux.
rem
rem   .\robocrop.cmd --input ~/Pictures/portraits --output ./dataset
rem   .\robocrop.cmd --setup      install everything, check it loads, and exit
rem
rem Honours ROBOCROP_VENV, ROBOCROP_PYTHON (default: python, then py),
rem ROBOCROP_NO_VLM=1 and ROBOCROP_SKIP_SYNC=1, as ./robocrop does.

setlocal EnableExtensions

rem goto instead of ( ) blocks: a ")" in the folder path would end a block early.
set "here=%~dp0"
set "here=%here:~0,-1%"
set "venv=%here%\.venv"
if defined ROBOCROP_VENV set "venv=%ROBOCROP_VENV%"
set "py=%venv%\Scripts\python.exe"
set "stamp=%venv%\.robocrop-deps"
set "mode=full"
if "%ROBOCROP_NO_VLM%"=="1" set "mode=core"

if exist "%py%" goto sync

set "base="
if defined ROBOCROP_PYTHON call :try "%ROBOCROP_PYTHON%"
if not defined ROBOCROP_PYTHON call :try python
if not defined ROBOCROP_PYTHON if not defined base call :try py
if not defined base goto nopython

>&2 echo robocrop: creating virtualenv in %venv% (first run only)
"%base%" -m venv "%venv%" || goto novenv
"%py%" -m pip install --quiet --upgrade pip >nul 2>&1
del "%stamp%" >nul 2>&1

:sync
if "%ROBOCROP_SKIP_SYNC%"=="1" goto run
set "want=%stamp%.new"
>"%want%" echo v2:%mode%
type "%here%\requirements.txt" >>"%want%"
if "%mode%"=="full" type "%here%\requirements-caption.txt" >>"%want%"
fc /b "%want%" "%stamp%" >nul 2>&1 && del "%want%" && goto run

>&2 echo robocrop: installing dependencies (first run only, this takes a few minutes)
"%py%" -m pip install --quiet -r "%here%\requirements.txt" || goto depsfailed
if not "%mode%"=="full" goto synced
"%py%" -m pip install --quiet -r "%here%\requirements-caption.txt" || goto captionfailed
:synced
move /y "%want%" "%stamp%" >nul

:run
if "%~1"=="--setup" goto setup
set "srcpath=%here%\src"
if defined PYTHONPATH set "srcpath=%srcpath%;%PYTHONPATH%"
set "PYTHONPATH=%srcpath%"
"%py%" -m robocrop %*
exit /b %ERRORLEVEL%

:setup
"%py%" -c "import cv2, numpy, PIL" || goto noload
>&2 echo robocrop: setup complete
exit /b 0

:try
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1 && set "base=%~1"
exit /b 0

:nopython
>&2 echo robocrop: Python 3.11 or newer is required; install it from https://www.python.org/downloads/ or set ROBOCROP_PYTHON
exit /b 1

:novenv
>&2 echo robocrop: could not create a virtualenv at %venv%
exit /b 1

:depsfailed
>&2 echo robocrop: dependency install failed; see the output above
exit /b 1

:captionfailed
>&2 echo robocrop: caption dependency install failed; re-run with ROBOCROP_NO_VLM=1 to skip it
exit /b 1

:noload
>&2 echo robocrop: the libraries installed but don't load; see the error above
exit /b 1
