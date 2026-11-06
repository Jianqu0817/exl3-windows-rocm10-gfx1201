@echo off
setlocal
call "%~dp0server.cmd" -m "%~dp0..\..\models\Huihui-Qwen3.8-27B-abliterated-EXL3-3.0bpw" ^
  -dm "%~dp0..\..\Qwen3.8-27B-DFlash2-EXL3-3.0bpw-tuned" -ndt 3 ^
  -cs 32768 -cq 4 -rcs 0.5 -chunk_size 1024 --parallel 1 --vision off --port 18543 %*
exit /b %errorlevel%
