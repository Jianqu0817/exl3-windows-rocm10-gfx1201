"""Model selection and a real harmless parent/child process termination check."""
import ctypes as C
from ctypes import wintypes as W
import json
import subprocess
import sys
from cleanup_vram import ROOT, model_processes, stop_processes


def main():
    runtime = str(ROOT / 'tools/exl3-gfx1201/python/python.exe')
    def proc(pid, name, exe, command):
        return dict(ProcessId=pid, Name=name, ExecutablePath=exe, CommandLine=command)
    cases = [
        proc(1, 'python.exe', runtime, 'python runtime.py script server.py -m model'),
        proc(2, 'python.exe', runtime, 'python bench_30k.py -m model'),
        proc(3, 'python.exe', runtime, 'python ordinary_script.py'),
        proc(4, 'python.exe', r'C:\Other\python.exe', 'python runtime.py script server.py -m model'),
        proc(5, 'python.exe', runtime, 'python -c multiprocessing.spawn.spawn_main()'),
        proc(6, 'python.exe', runtime, 'python -c multiprocessing.spawn.spawn_main()'),
        proc(7, 'llama-server.exe', str(ROOT / 'build/llama-server.exe'), 'llama-server -m model'),
        proc(8, 'dwm.exe', r'C:\Windows\System32\dwm.exe', ''),
    ]
    raw = {'process_local': {'pid_5_gpu': 512 * 2**20, 'pid_6_gpu': 0}}
    assert [p['ProcessId'] for p in model_processes(cases, raw, 'gpu')] == [1, 2, 5, 7]

    external = r"C:\Runtime\python.exe"
    exact_cmd = f'"{external}" "{ROOT / "runtime.py"}" script server.py -m model'
    nearby = exact_cmd.replace(str(ROOT / "runtime.py"), str(ROOT.parent / "other" / "runtime.py"))
    assert [p["ProcessId"] for p in model_processes([
        proc(9,"python.exe",external,exact_cmd), proc(10,"python.exe",external,nearby)
    ], raw, "gpu")] == [9]

    code = ('import subprocess,sys,time; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"]); '
            'print(p.pid,flush=True); time.sleep(120)')
    parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    kernel = C.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    kernel.OpenProcess.restype = W.HANDLE
    kernel.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, child_pid)
    assert handle
    try:
        stopped = stop_processes([{'ProcessId': parent.pid, 'Name': 'test-python.exe'}], tree=True)
        assert len(stopped) == 1
        assert parent.wait(timeout=10) != 0
        assert kernel.WaitForSingleObject(handle, 5000) == 0
    finally:
        kernel.CloseHandle(handle)
        if parent.poll() is None:
            subprocess.run(['taskkill.exe', '/PID', str(parent.pid), '/T', '/F'], capture_output=True)
    print(json.dumps({'status': 'PASS', 'selection_cases': len(cases) + 2,
                      'real_process_tree_stopped': True, 'live_model_stopped': False}))


if __name__ == '__main__':
    main()
