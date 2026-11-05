@echo off
rem ---------------------------------------------------------------
rem bmdrt one-click launcher (Windows).
rem Double-click  -> process everything in inbox\ then pause.
rem From a shell  -> bmdrt.bat auto data.csv --quick   (args passed through)
rem Change the interpreter below if your env moves.
rem ---------------------------------------------------------------
setlocal
set "PY=D:\autoeis_env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0bmdrt.py" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RC%
