@echo off
setlocal
cd /d "%~dp0"
call "%~dp0tools\python.cmd" -I -X utf8 -u "%~dp0runtime.py" %*
exit /b %errorlevel%
