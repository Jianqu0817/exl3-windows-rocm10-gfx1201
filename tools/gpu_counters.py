"""Read native WDDM PDH counters; keep allocation and segment-residency fields separate."""
import ctypes as C
from ctypes import wintypes as W

class ValueUnion(C.Union):
    _fields_ = [("longValue", W.LONG), ("doubleValue", C.c_double), ("largeValue", C.c_longlong), ("stringValue", W.LPWSTR)]

class Value(C.Structure):
    _fields_ = [("status", W.DWORD), ("data", ValueUnion)]

class Item(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", Value)]

COUNTERS = {
    "process_dedicated": r"\GPU Process Memory(*)\Dedicated Usage",
    "process_shared": r"\GPU Process Memory(*)\Shared Usage",
    "process_local": r"\GPU Process Memory(*)\Local Usage",
    "process_nonlocal": r"\GPU Process Memory(*)\Non Local Usage",
    "process_committed": r"\GPU Process Memory(*)\Total Committed",
    "adapter_dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
    "adapter_shared": r"\GPU Adapter Memory(*)\Shared Usage",
    "adapter_committed": r"\GPU Adapter Memory(*)\Total Committed",
    "adapter_local": r"\GPU Local Adapter Memory(*)\Local Usage",
    "adapter_nonlocal": r"\GPU Non Local Adapter Memory(*)\Non Local Usage",
    "engine_utilization": r"\GPU Engine(*)\Utilization Percentage",
}

class GpuCounters:
    def __init__(self):
        self.api = C.WinDLL("pdh")
        self.api.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE)]
        self.api.PdhAddEnglishCounterW.argtypes = [W.HANDLE, W.LPCWSTR, C.c_size_t, C.POINTER(W.HANDLE)]
        self.api.PdhCollectQueryData.argtypes = [W.HANDLE]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.c_void_p]
        self.api.PdhCloseQuery.argtypes = [W.HANDLE]
        self.query = W.HANDLE()
        self.handles, self.errors = {}, {}
        status = self.api.PdhOpenQueryW(None, 0, C.byref(self.query))
        if status:
            raise OSError(f"PdhOpenQueryW: 0x{status & 0xffffffff:08x}")
        for name, path in COUNTERS.items():
            handle = W.HANDLE()
            status = self.api.PdhAddEnglishCounterW(self.query, path, 0, C.byref(handle))
            if status:
                self.errors[name] = f"0x{status & 0xffffffff:08x}"
            else:
                self.handles[name] = handle
        self.api.PdhCollectQueryData(self.query)

    def read(self):
        status = self.api.PdhCollectQueryData(self.query)
        if status:
            return {"collection_error": f"0x{status & 0xffffffff:08x}"}
        result = {}
        for name, handle in self.handles.items():
            size, count = W.DWORD(), W.DWORD()
            self.api.PdhGetFormattedCounterArrayW(handle, 0x200, C.byref(size), C.byref(count), None)
            if not size.value:
                result[name] = {}
                continue
            buffer = C.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(handle, 0x200, C.byref(size), C.byref(count), buffer)
            if status:
                result[name] = {"_error": f"0x{status & 0xffffffff:08x}"}
                continue
            items = C.cast(buffer, C.POINTER(Item))
            result[name] = {items[i].name: items[i].value.data.doubleValue
                            for i in range(count.value) if items[i].value.status in (0, 1)}
        return result

    def close(self):
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = None

    @staticmethod
    def for_pid(raw, pid):
        prefix = f"pid_{pid}_"
        processes = {k: {n: v for n, v in values.items() if n.startswith(prefix)}
                     for k, values in raw.items() if k.startswith("process_") or k == "engine_utilization"}
        adapters = set()
        for values in processes.values():
            for name in values:
                if "luid_" in name:
                    adapters.add("luid_" + name.split("luid_", 1)[1].split("_eng_", 1)[0])
        for k, values in raw.items():
            if k.startswith("adapter_"):
                processes[k] = {n: v for n, v in values.items()
                                if any(n.startswith(adapter) for adapter in adapters)}
        return {"adapters": sorted(adapters), "counters": processes}

if __name__ == "__main__":
    import json, time
    counters = GpuCounters()
    time.sleep(1)
    print(json.dumps({"errors": counters.errors, "sample": counters.read()}, indent=2))
    counters.close()
