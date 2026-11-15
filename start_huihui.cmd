@echo off
setlocal
if not defined EXL3_GEMV_M4_FDOT2 set "EXL3_GEMV_M4_FDOT2=1"
if not defined EXL3_MODEL_DIR set "EXL3_MODEL_DIR=%~dp0..\..\models\Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw"
if not defined EXL3_DRAFT_DIR set "EXL3_DRAFT_DIR=%~dp0..\..\Qwen3.8-27B-DFlash2-EXL3-3.0bpw-tuned"
call "%~dp0server.cmd" -m "%EXL3_MODEL_DIR%" ^
  -dm "%EXL3_DRAFT_DIR%" -ndt 3 ^
  -cs 32768 -cq 4 -rcs 0.5 -chunk_size 1024 --parallel 1 --vision off --port 18543 %*
exit /b %errorlevel%
