"""Download Qwen3-0.6B (base, safetensors) from ModelScope to D:\models."""
import urllib.request, json, os, time

OUT = r"D:\models\Qwen3-0.6B-Base"
os.makedirs(OUT, exist_ok=True)
API = "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-0.6B-Base/repo/files?Revision=master"
LOG = open(r"D:\user\flypoet\qwen06_dl.log", "a", encoding="utf-8")

def log(m):
    line = "[" + time.strftime("%H:%M:%S") + "] " + m
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
d = json.load(urllib.request.urlopen(req, timeout=20))
files = (d.get("Data") or {}).get("Files") or []
need = [f["Path"] for f in files if f["Path"].endswith((".safetensors", ".json", ".txt"))
        and "gguf" not in f["Path"].lower()]
log(f"files to fetch: {need}")
BASE = "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-Base/resolve/master/"
for p in need:
    dst = os.path.join(OUT, os.path.basename(p))
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        log(f"cached {p}"); continue
    t0 = time.time()
    urllib.request.urlretrieve(BASE + p, dst)
    log(f"done {p} {os.path.getsize(dst)/1e6:.1f}MB in {time.time()-t0:.0f}s")
log("QWEN06 DOWNLOAD DONE")
