@echo off
rem ---------------------------------------------------------------
rem nested_fit launcher: standalone EXPLORATORY nested-topology fit.
rem Does NOT touch bmdrt.py / bmdrt.bat behaviour.
rem Double-click -> fit everything in inbox\ then pause.
rem From a shell  -> nested_fit.bat data.csv --budget quick --with-voigt
rem Change the interpreter below if your env moves.
rem ---------------------------------------------------------------
setlocal
set "PY=D:\autoeis_env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0nested_fit.py" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RC%
