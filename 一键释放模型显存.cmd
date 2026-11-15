@echo off
chcp 65001 >nul
call "%~dp0tools\python.cmd" -X utf8 "%~dp0tools\cleanup_vram.py" --models %*
set "cleanup_exit=%errorlevel%"
echo.
pause
exit /b %cleanup_exit%
