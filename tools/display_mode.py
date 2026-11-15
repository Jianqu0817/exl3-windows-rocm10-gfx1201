"""Use the existing active motherboard HDMI path; restore uses the original CCD topology on this PC."""
import base64, ctypes as C, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cleanup_vram import adapter_names
U=C.c_uint32
class Luid(C.Structure):
    _fields_=[("lo",U),("hi",U)]
class Source(C.Structure):
    _fields_=[("luid",Luid),("id",U),("idx",U),("status",U)]
class Target(C.Structure):
    _fields_=[("luid",Luid),("id",U),("idx",U),("tech",U),("rotation",U),("scaling",U),("hz_n",U),("hz_d",U),("scan",U),("available",U),("status",U)]
class PathInfo(C.Structure):
    _fields_=[("source",Source),("target",Target),("flags",U)]
class Mode(C.Structure):
    _fields_=[("kind",U),("id",U),("luid",Luid),("payload",U*12)]
assert C.sizeof(PathInfo)==72 and C.sizeof(Mode)==64
api=C.windll.user32
def check(n):
    if n: raise OSError(n,C.FormatError(n))
def snapshot():
    np,nm=U(),U()
    check(api.GetDisplayConfigBufferSizes(2,C.byref(np),C.byref(nm)))
    ps,ms=(PathInfo*np.value)(),(Mode*nm.value)()
    check(api.QueryDisplayConfig(2,C.byref(np),ps,C.byref(nm),ms,None))
    return (PathInfo*np.value)(*ps[:np.value]),(Mode*nm.value)(*ms[:nm.value])
names,_=adapter_names()
def identify(p):
    l=p.target.luid
    return names.get(f"luid_0x{l.hi:08X}_0x{l.lo:08X}_phys_0"),p.target.tech
def apply(ps,ms):
    for flags in (0x40,0x80|0x200):
        check(api.SetDisplayConfig(len(ps),ps,len(ms),ms,0x20|flags))
backup=Path(__file__).resolve().parents[1] / 'state/display-before.json'
def restore():
    data=json.loads(backup.read_text())
    ps,ms=[(t*data[k+"Count"]).from_buffer_copy(base64.b64decode(data[k])) for t,k in ((PathInfo,"paths"),(Mode,"modes"))]
    apply(ps,ms)
    def signature(paths, modes):
        return sorted((identify(p),p.target.id,tuple(modes[p.source.idx].payload[:2]),p.target.hz_n,p.target.hz_d) for p in paths)
    if signature(*snapshot()) != signature(ps,ms):
        raise RuntimeError("Restored modes differ from the saved configuration; backup kept")
    backup.unlink()

ps,ms=snapshot()
print("Active:",[(identify(p),list(ms[p.source.idx].payload[:2]),p.target.hz_n/max(1,p.target.hz_d)) for p in ps],flush=True)
if sys.argv[1:] == ["restore"]:
    restore()
elif sys.argv[1:] == ["hdmi"]:
    match=[p for p in ps if identify(p)==("AMD Radeon(TM) Graphics",5)]
    if len(ps)==1 and len(match)==1: sys.exit(0)
    if not (len(match)==1 and len(ps)==2 and any(identify(p)==("AMD Radeon RX 9070 XT",10) for p in ps)):
        sys.exit("Both AMD iGPU HDMI and RX 9070 XT DP must already be active. Use Windows Display Settings for other layouts.")
    p=match[0]
    data={k:base64.b64encode(bytes(a)).decode() for k,a in (("paths",ps),("modes",ms))}
    data.update(pathsCount=len(ps),modesCount=len(ms))
    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("x") as f: json.dump(data,f)
    # Keep the HDMI signal timing and resolution; a sole desktop source must be at origin.
    s,t=ms[p.source.idx],ms[p.target.idx]
    assert s.kind==1 and t.kind==2
    s.payload[3]=s.payload[4]=0
    p.source.idx,p.target.idx=0,1
    apply((PathInfo*1)(p),(Mode*2)(s,t))
    time.sleep(2)
    if [identify(p) for p in snapshot()[0]] != [("AMD Radeon(TM) Graphics",5)]:
        restore()
        raise RuntimeError("Unexpected active topology; original restored")
print("Now:",[identify(p) for p in snapshot()[0]])
