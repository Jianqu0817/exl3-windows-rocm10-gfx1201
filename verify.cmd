@echo off
call "%~dp0port.cmd" verify %*
exit /b %errorlevel%
