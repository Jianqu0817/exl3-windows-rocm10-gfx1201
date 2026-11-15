"""Refresh desktop GPU helpers, or explicitly stop this workspace's model processes."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import winreg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_counters import GpuCounters

STATE = ROOT / "state/vram-cleanup"
PREFS = r"Software\Microsoft\DirectX\UserGpuPreferences"
APPS = {"WeChatAppEx.exe", "Weixin.exe", "QQ.exe", "ChatGPT.exe",
        "msedgewebview2.exe", "msedge.exe"}
MODEL_EXES = {"llama-server.exe", "llama-cli.exe", "llama-kvmem-cli.exe",
              "ninfer.exe", "mini_main.exe"}


def adapter_names():
    class Desc(C.Structure):
        _fields_ = [("description", W.WCHAR * 128), ("vendor", W.UINT),
                    ("device", W.UINT), ("subsystem", W.UINT), ("revision", W.UINT),
                    ("video", C.c_size_t), ("system", C.c_size_t), ("shared", C.c_size_t),
                    ("luid_low", W.DWORD), ("luid_high", W.LONG), ("flags", W.UINT)]

    def method(obj, index, result, *args):
        table = C.cast(obj, C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(result, C.c_void_p, *args)(table[index])

    factory = C.c_void_p()
    iid = C.create_string_buffer(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    hr = C.WinDLL("dxgi").CreateDXGIFactory1(iid, C.byref(factory))
    if hr:
        raise OSError(f"CreateDXGIFactory1: {hr:#x}")
    names, target, largest = {}, None, 0
    try:
        index = 0
        while True:
            adapter = C.c_void_p()
            if method(factory, 12, W.LONG, W.UINT, C.POINTER(C.c_void_p))(factory, index, C.byref(adapter)):
                break
            try:
                desc = Desc()
                hr = method(adapter, 10, W.LONG, C.POINTER(Desc))(adapter, C.byref(desc))
                if not hr:
                    key = f"luid_0x{desc.luid_high & 0xffffffff:08X}_0x{desc.luid_low:08X}_phys_0"
                    names[key] = desc.description
                    if desc.video > largest:
                        target, largest = key, desc.video
            finally:
                method(adapter, 2, W.ULONG)(adapter)
            index += 1
    finally:
        method(factory, 2, W.ULONG)(factory)
    return names, target


def sample():
    counters = GpuCounters()
    try:
        time.sleep(1)
        raw = counters.read()
        return raw
    finally:
        counters.close()


def processes(include_models=False):
    names = ",".join(f"'{name}'" for name in sorted(APPS))
    script = (
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
        "$session=(Get-Process -Id $PID).SessionId; "
        "Get-CimInstance Win32_Process | Where-Object { $_.SessionId -eq $session "
        + ("" if include_models else f"-and $_.Name -in @({names}) ") + "} | "
        "Select-Object ProcessId,Name,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script],
                            capture_output=True, encoding="utf-8", errors="replace", check=True)
    value = json.loads(result.stdout) if result.stdout.strip() else []
    return value if isinstance(value, list) else [value]


def model_processes(procs, raw, adapter):
    selected = []
    for p in procs:
        # External/shared Python is ours only when running this checkout's server.
        cmd = p.get('CommandLine') or ''
        server_here = bool(re.search(r'(?:^|\s)"?' + re.escape(str(ROOT / "runtime.py")) + r'"?(?=\s|$)', cmd, re.I)) and bool(
            re.search(r'\bscript\s+server\.py(?:\s|$)', cmd, re.I))
        if not p.get('ExecutablePath') or not (Path(p['ExecutablePath']).is_relative_to(ROOT) or server_here):
            continue
        name = p['Name'].lower()
        cmd = p.get('CommandLine') or ''
        python = name in ('python.exe', 'pythonw.exe')
        server = python and 'runtime.py' in cmd.lower() and bool(
            re.search(r'\bscript\s+server\.py(?:\s|$)', cmd, re.I))
        script = python and bool(re.search(
            r'(?:^|[\\/\s\"])(?:bench_30k|bench_draft|chat|completion)\.py(?:[\s\"]|$)', cmd, re.I)) and bool(
            re.search(r'(?:^|\s)(?:-m|--model_dir)(?:\s|=)', cmd))
        # A multiprocessing worker can outlive its parent; require this runtime and substantial GPU storage.
        gpu_bytes = sum(size for key in ('process_local', 'process_nonlocal')
                        for instance, size in raw.get(key, {}).items()
                        if instance.startswith(f"pid_{p['ProcessId']}_") and instance.endswith(adapter or 'NO_ADAPTER'))
        worker = python and 'multiprocessing.spawn' in cmd and gpu_bytes >= 256 * 2**20
        if server or script or worker or name in MODEL_EXES:
            selected.append(p)
    return selected


def stop_processes(candidates, tree=False):
    stopped = []
    for p in candidates:
        command = ['taskkill.exe', '/PID', str(p['ProcessId']), '/F']
        if tree:
            command.append('/T')
        result = subprocess.run(command, capture_output=True)
        if result.returncode == 0:
            stopped.append({'pid': p['ProcessId'], 'name': p['Name']})
        else:
            print(f"未结束 PID {p['ProcessId']}（可能已退出或无权限）")
    return stopped


def set_preferences(procs):
    STATE.mkdir(parents=True, exist_ok=True)
    backup_file = STATE / "gpu-preferences-before.json"
    backup = json.loads(backup_file.read_text(encoding="utf-8")) if backup_file.exists() else {}
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, PREFS, access=winreg.KEY_READ | winreg.KEY_WRITE) as key:
        for path in sorted({p["ExecutablePath"] for p in procs if p["ExecutablePath"]}):
            try:
                old = winreg.QueryValueEx(key, path)[0]
            except FileNotFoundError:
                old = None
            if path not in backup:
                backup[path] = old
                backup_file.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")
            value = re.sub(r"GpuPreference=\d+;", "", old or "") + "GpuPreference=1;"
            winreg.SetValueEx(key, path, 0, winreg.REG_SZ, value)


def restore_preferences():
    backup = json.loads((STATE / "gpu-preferences-before.json").read_text(encoding="utf-8"))
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, PREFS, access=winreg.KEY_READ | winreg.KEY_WRITE) as key:
        for path, value in backup.items():
            try: current = winreg.QueryValueEx(key, path)[0]
            except FileNotFoundError: continue
            if current != re.sub(r"GpuPreference=\d+;", "", value or "") + "GpuPreference=1;":
                continue
            if value is None:
                try:
                    winreg.DeleteValue(key, path)
                except FileNotFoundError:
                    pass
            else:
                winreg.SetValueEx(key, path, 0, winreg.REG_SZ, value)
    print("已还原本程序修改的 GPU 首选项；正常重启对应应用后生效。")


def main():
    parser = argparse.ArgumentParser(description="整理桌面显存；--models 停止本项目模型服务并释放其占用")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview", action="store_true", help="只查看，不修改或结束进程")
    mode.add_argument("--restore", action="store_true", help="还原本程序修改的 GPU 首选项")
    parser.add_argument('--models', action='store_true',
                        help='停止本项目模型推理进程及子进程，包含正在运行的服务；可配合 --preview')
    args = parser.parse_args()
    if args.models and args.restore:
        parser.error('--models 不能与 --restore 一起使用')
    if args.restore:
        restore_preferences()
        return

    names, target = adapter_names()
    raw, procs = sample(), processes(args.models)
    before = raw["adapter_dedicated"]
    if args.models:
        candidates = model_processes(procs, raw, target)
        print('释放模型显存：停止本项目推理进程及其子进程，包括正在使用的服务。')
        print('按进程识别，不依赖窗口、监听端口或父进程是否存活。\n')
    else:
        candidates = [p for p in procs if re.search(r"(?:^|\s)--type=gpu-process(?:\s|$)", p["CommandLine"] or "")
                  or (p["Name"] == "ChatGPT.exe" and
                      "--utility-sub-type=on_device_model.mojom.OnDeviceModelService" in (p["CommandLine"] or ""))]
        candidates = [p for p in candidates if sum(
            size for name, size in raw["process_local"].items()
            if name.startswith(f"pid_{p['ProcessId']}_") and name.endswith(target or "NO_ADAPTER")
        ) >= 16 * 2**20]
        print("显存整理：桌面应用优先核显，刷新图形辅助进程。窗口可能短暂闪烁。")
        print("保留应用主进程、推理服务、DWM、资源管理器和显卡驱动。\n")
    for p in candidates:
        print(f"{'预览' if args.preview else '结束' if args.models else '刷新'} {p['Name']}  PID {p['ProcessId']}")
    if not candidates:
        print('没有识别到本项目模型进程。' if args.models else "没有发现仍占用独显 16 MiB 以上的可刷新辅助进程。")
    refreshed = []
    if not args.preview:
        if not args.models:
            set_preferences(procs)
        refreshed = stop_processes(candidates, tree=args.models)
        time.sleep(2)
    after_raw = raw if args.preview else sample()
    after = after_raw['adapter_dedicated']
    print("\n适配器专用显存（GiB）：")
    for adapter, used in before.items():
        now = after.get(adapter, used)
        print(f"{names.get(adapter, adapter)}: {used / 2**30:.2f} → {now / 2**30:.2f}，减少 {(used-now) / 2**30:.2f}")
    if not args.preview:
        STATE.mkdir(parents=True, exist_ok=True)
        remaining = model_processes(processes(True), after_raw, target) if args.models else []
        (STATE / ('last-model-run.json' if args.models else 'last-run.json')).write_text(json.dumps(
            {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "names": names,
             "before": before, "after": after, "refreshed": refreshed,
             'remaining_models': [{'pid': p['ProcessId'], 'name': p['Name']} for p in remaining]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        if args.models:
            if remaining:
                for p in remaining:
                    print(f"仍有模型进程：{p['Name']} PID {p['ProcessId']}，可能未能结束或被自动重启。")
                raise SystemExit(1)
            print('\n本项目模型进程已结束。系统、驱动及其他应用使用的显存会保留。')
        else:
            print("\n完成。系统和正在使用的模型显存会保留；反复点击不会保证继续释放。")


if __name__ == "__main__":
    main()
