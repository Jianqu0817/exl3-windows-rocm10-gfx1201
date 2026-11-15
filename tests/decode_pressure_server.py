"""Real server with a transient loader scratch, for the fresh-process restart test."""
from pathlib import Path
import os,runpy,torch
from exllamav3.modules.linear import Linear
old=Linear.load_exl3
def pressure(self,key):
    if self.caps.get("logits_output") and self.is_exl3_storage(key):
        tmp=torch.empty(512<<20,device=self.device,dtype=torch.uint8)
        tmp.fill_(1);torch.cuda.synchronize(self.device);del tmp
    return old(self,key)
Linear.load_exl3=pressure
print("PLACEMENT_TEST_PID",os.getpid(),os.environ.get("EXL3_DECODE_PLACEMENT"),flush=True)
runpy.run_path(str(Path(__file__).resolve().parents[1]/"server.py"),run_name="__main__")
