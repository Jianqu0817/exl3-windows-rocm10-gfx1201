@echo off
setlocal
if not defined EXL3_PYTHON if exist "%~dp0..\python\python.exe" set "EXL3_PYTHON=%~dp0..\python\python.exe"
if not defined EXL3_PYTHON set "EXL3_PYTHON=D:\LLM_local\tools\exl3-gfx1201\python\python.exe"
if not exist "%EXL3_PYTHON%" (
  echo Set EXL3_PYTHON to the full path of the compatible runtime python.exe. 1>&2
  exit /b 2
)
"%EXL3_PYTHON%" %*
exit /b %errorlevel%
