@echo off
setlocal
if not defined EXL3_FAST_DECODE set "EXL3_FAST_DECODE=0"
set "EXL3_QC_STAGING_PAGES=64"
call "%~dp0start_tauritavern.cmd" -cs 122880 -cq 4 -chunk_size 512 %*
exit /b %errorlevel%
