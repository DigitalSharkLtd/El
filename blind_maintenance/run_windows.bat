@echo off
rem Blind Maintenance Benchmark - Windows launcher.
rem Needs Python 3.11+ (https://www.python.org/downloads/, tick "Add python to PATH").
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=python
where py >nul 2>nul && set PY=py -3

echo [1/4] Installing dependencies (numpy, pytest)...
%PY% -m pip install -r requirements.txt || goto :fail

echo.
echo [2/4] Correctness tests (section 8)...
%PY% -m pytest -q -p no:cacheprovider tests || goto :fail

echo.
echo [3/4] Smoke test (technical check, not an experiment)...
%PY% scripts\smoke_test.py || goto :fail

echo.
echo [4/4] Recomputing all 768 confirmatory worlds and comparing with the published results (1-5 min)...
%PY% scripts\reproduce_check.py || goto :fail

echo.
echo DONE. Report: REPORT_RU.md   Tables: results\confirmatory\summary_tables.md
pause
exit /b 0

:fail
echo.
echo ERROR: a step failed, see the messages above.
pause
exit /b 1
