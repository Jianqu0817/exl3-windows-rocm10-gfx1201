"""Run only this port's build at low priority; stop its process tree on RAM pressure."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time

class Memory(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]

def available_gib():
    state = Memory()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return state.available / 2**30

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    while available_gib() < 10:
        if "--wait" not in sys.argv:
            raise SystemExit("Build deferred: less than 10 GiB RAM available")
        print(f"Waiting for RAM: {available_gib():.2f} GiB available; need 10 GiB", flush=True)
        time.sleep(15)
    env = os.environ.copy()
    jobs = str(min(2, max(1, int(env.get("MAX_JOBS", "1")))))
    env.update(MAX_JOBS=jobs, PYTHONNOUSERSITE="1")
    env.setdefault("VSDIR", r"D:\LLM_local\tools\VS2022")
    (root / "logs").mkdir(exist_ok=True)
    with (root / "logs/build.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(["cmd.exe", "/d", "/c", str(root / "tools/build/build_ext.cmd")],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
        print(f"Build PID {child.pid}, MAX_JOBS={jobs}, log={log.name}", flush=True)
        while child.poll() is None:
            free = available_gib()
            if free < 5:
                print(f"Stopping own build: RAM available {free:.2f} GiB", flush=True)
                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                raise SystemExit(75)
            time.sleep(5)
        raise SystemExit(child.returncode)
