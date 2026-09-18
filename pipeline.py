"""Overnight pipeline: download full corpus -> prepare -> train std -> train fly."""
import subprocess, sys, os, time, json, base64, urllib.request, urllib.parse

os.chdir(r"D:/user/flypoet/data")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
PY = r"C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe"
LOG = open(r"D:/user/flypoet/pipeline.log", "a", encoding="utf-8")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()

def api_file(repo_path, tries=4):
    url = ("https://api.github.com/repos/chinese-poetry/chinese-poetry/contents/"
           + urllib.parse.quote(repo_path))
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": "token " + TOKEN, "User-Agent": "req"})
            d = json.load(urllib.request.urlopen(req))
            data = base64.b64decode(d["content"])
            if data[:1] == b"[":          # valid JSON array
                return data
            raise ValueError("not JSON array")
        except Exception:
            time.sleep(2 * (i + 1))
    raise IOError(f"download failed: {repo_path}")

# ---- phase 1: corpus
files = [l.strip() for l in open("poet_files.txt", encoding="utf-8") if l.strip()]
todo = []
for name in files:
    dst = os.path.basename(name)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        try:
            if open(dst, "rb").read(1) != b"[":
                todo.append(("全唐诗/" + name, dst))
        except FileNotFoundError:
            todo.append(("全唐诗/" + name, dst))
log(f"corpus: {len(todo)} files to fetch (of {len(files)})")
t0 = time.time()
fail = []
for i, (src, dst) in enumerate(todo):
    try:
        open(dst, "wb").write(api_file(src))
    except Exception as e:
        fail.append(dst); log(f"FAIL {dst}: {e}")
    if (i + 1) % 30 == 0:
        log(f"  {i+1}/{len(todo)} {time.time()-t0:.0f}s")
if fail:
    log(f"retrying {len(fail)} failed files once more...")
    for src, dst in [( "全唐诗/" + f, f) for f in fail]:
        try: open(dst, "wb").write(api_file(src))
        except Exception as e: log(f"STILL FAIL {dst}: {e}")
log(f"corpus done in {time.time()-t0:.0f}s")

# ---- phase 2: prepare
log("prepare_data...")
r = subprocess.run([PY, "-X", "utf8", r"D:\user\flypoet\prepare_data.py"],
                   capture_output=True, text=True, encoding="utf-8")
log(r.stdout[-1500:] if r.stdout else "no stdout")
if r.returncode != 0:
    log("PREPARE FAILED: " + r.stderr[-1500:]); sys.exit(1)

# ---- phase 3+4: training arms
for arm in ("std", "fly"):
    log(f"train arm={arm} start")
    r = subprocess.run([PY, "-X", "utf8", r"D:\user\flypoet\train_flypoet.py",
                        "--arm", arm, "--steps", "4000"],
                       capture_output=True, text=True, encoding="utf-8")
    tail = (r.stdout or "")[-800:]
    log(f"train {arm} exit={r.returncode}\n{tail}")
    if r.returncode != 0:
        log(r.stderr[-1500:])
log("PIPELINE DONE")
