"""Probe PersistentAutotuner cache edge cases on a real filesystem (issue #536)."""
import logging, os, stat, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from rl_engine.platforms import autotuner as at

CANDS = (at.TuneConfig(), at.TuneConfig((("num_warps", 8),)))
timing = lambda c: 1.0 if c == CANDS[0] else 0.5
logging.basicConfig(level=logging.WARNING, format="    log: %(message)s")
results = {}

def run(name, tuner, expect_source="tuned"):
    sel = tuner.select({"op": name}, CANDS, timing)
    again = tuner.select({"op": name}, CANDS, timing)
    ok = sel.config == CANDS[1] and sel.source == expect_source and again.source == "memory"
    results[name] = ok
    print(f"{name}: first={sel.source} second={again.source} chose_fastest={sel.config == CANDS[1]} -> {'PASS' if ok else 'FAIL'}")

with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    # 1. read-only cache directory (0500, owned by this user)
    ro = root / "readonly"; ro.mkdir(mode=0o700); ro.chmod(0o500)
    run("readonly_dir", at.PersistentAutotuner(ro))
    print("    files left:", sorted(p.name for p in ro.iterdir()))
    ro.chmod(0o700)
    # 2. atomic replace fails (os.replace raises OSError)
    d2 = root / "replace_fail"; d2.mkdir(mode=0o700)
    real = os.replace
    def boom(*a, **k): raise OSError(28, "No space left on device")
    at.os.replace = boom
    try: run("replace_fails", at.PersistentAutotuner(d2))
    finally: at.os.replace = real
    left = sorted(p.name for p in d2.iterdir())
    print("    files left:", left)
    results["replace_fails_no_temp"] = not [n for n in left if not n.endswith(".lock")]
    # 3. directory fsync fails after a successful replace: record must persist
    d3 = root / "fsync_fail"; d3.mkdir(mode=0o700)
    real_fsync = os.fsync; real_open = os.open
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError(5, "I/O error")
        return real_fsync(fd)
    at.os.fsync = fsync
    try: run("dir_fsync_fails", at.PersistentAutotuner(d3))
    finally: at.os.fsync = real_fsync
    json_files = [p for p in d3.iterdir() if p.suffix == ".json"]
    results["dir_fsync_fails_persisted"] = len(json_files) == 1
    print("    persisted json:", len(json_files))
    # 4. restart reads the persisted record from disk
    restarted = at.PersistentAutotuner(d3).select({"op": "dir_fsync_fails"}, CANDS, lambda c: 1/0)
    results["restart_reads_disk"] = restarted.source == "disk" and restarted.config == CANDS[1]
    print("restart:", restarted.source, "->", "PASS" if results["restart_reads_disk"] else "FAIL")
print("ALL PASS" if all(results.values()) else f"FAILED: {[k for k,v in results.items() if not v]}")
sys.exit(0 if all(results.values()) else 1)
