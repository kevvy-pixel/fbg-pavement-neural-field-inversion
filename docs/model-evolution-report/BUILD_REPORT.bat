@echo off
setlocal
cd /d "%~dp0"

where xelatex >nul 2>nul
if errorlevel 1 (
  echo [ERROR] XeLaTeX was not found.
  echo Install TeX Live or MiKTeX with XeLaTeX support, then run this file again.
  pause
  exit /b 1
)

echo [1/3] Regenerating figures if Python is available...
where python >nul 2>nul
if not errorlevel 1 (
  python make_figures.py
) else (
  echo Python was not found; using the figures already included in this folder.
)

echo [2/3] Compiling the report...
xelatex -interaction=nonstopmode -halt-on-error FBG_FiLM_Model_Evolution_Report_CN.tex
if errorlevel 1 goto :failed
xelatex -interaction=nonstopmode -halt-on-error FBG_FiLM_Model_Evolution_Report_CN.tex
if errorlevel 1 goto :failed

echo [3/3] Done.
echo Output: FBG_FiLM_Model_Evolution_Report_CN.pdf
pause
exit /b 0

:failed
echo [ERROR] Compilation failed. Check FBG_FiLM_Model_Evolution_Report_CN.log.
pause
exit /b 1
