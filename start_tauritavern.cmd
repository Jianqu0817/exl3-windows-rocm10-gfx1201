@echo off
setlocal
if not "%EXL3_FAST_DECODE%"=="0" (
  if not defined EXL3_DECODE_PLACEMENT set "EXL3_DECODE_PLACEMENT=probe"
  if not defined EXL3_GEMV_TILES_T set "EXL3_GEMV_TILES_T=2"
)
:serve
call "%~dp0start_huihui.cmd" --alias huihui-qwen3.8-27b --thinking off --complete-sentences %*
if not "%errorlevel%"=="75" exit /b %errorlevel%
if not "%EXL3_DECODE_PLACEMENT%"=="probe" exit /b 75
set "EXL3_DECODE_PLACEMENT=head-first"
goto serve
