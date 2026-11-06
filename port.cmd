@echo off
setlocal
cd /d "%~dp0"
"D:\LLM_local\tools\exl3-gfx1201\python\python.exe" -I -X utf8 -u "%~dp0runtime.py" %*
exit /b %errorlevel%
