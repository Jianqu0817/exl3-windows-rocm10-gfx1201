@echo off
powershell -NoProfile -Command "$target='%~dp0runtime.py'; Get-NetTCPConnection -LocalPort 18543 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$_); if($p.Name -eq 'python.exe' -and $p.CommandLine.Contains($target) -and $p.CommandLine.Contains('server.py')) { Stop-Process -Id $p.ProcessId } }"
exit /b %errorlevel%
