@echo off
call "%~dp0port.cmd" script server.py %*
exit /b %errorlevel%
