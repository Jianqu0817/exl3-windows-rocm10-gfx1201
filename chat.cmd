@echo off
call "%~dp0port.cmd" chat %*
exit /b %errorlevel%
