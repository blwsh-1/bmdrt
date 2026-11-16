@echo off
rem ---------------------------------------------------------------
rem bmdrt: the ONLY Windows launcher (2026-11-16).
rem
rem Double-click   -> production AUTO (voigt family) over inbox\.
rem From a shell   -> bmdrt.bat <subcommand> [args]
rem
rem   bmdrt.bat nested  data.csv --budget quick   (nested, PROVISIONAL)
rem   bmdrt.bat compare data.csv                  (voigt + nested side by side)
rem   bmdrt.bat auto    data.csv --quick          (production, voigt)
rem   bmdrt.bat list / status / check / config    (maintenance)
rem
rem Subcommands are auto-discovered from cmd_<name>.py; nothing is
rem registered here. `nested_fit.bat` was removed on 2026-11-16 --
rem use `bmdrt.bat nested` instead (same main(), same arguments).
rem Change the interpreter below if your env moves.
rem ---------------------------------------------------------------
setlocal
set "PY=D:\autoeis_env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if "%~1"=="" (
  echo [bmdrt] no subcommand: running AUTO ^(production, voigt family^) over inbox\
  echo [bmdrt] for the nested family use:  bmdrt.bat nested spectrum.csv
)
"%PY%" "%~dp0bmdrt.py" %*
set "RC=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %RC%
