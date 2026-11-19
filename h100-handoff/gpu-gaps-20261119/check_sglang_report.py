#!/usr/bin/env python
"""Check an SGLang rollout report (h3_sglang_rollout_report.py) and compare it with B200.

    python check_sglang_report.py <OUT>/report.json [reference/sglang-report-b200-c9826852.json]

Exit 0 when, at every TP size in the report:
- every bound output equals the training operator bitwise on every rank, also on a row
  prefix where one is measured (not for the AdaLN tables), and the module is restored to
  SGLang's own output after remove();
- every bound output equals TP1;
- in the whole DiT, every bound forward ran once per model forward (empty readback);
- the plugin hooked, installed, repeated, refused Spectrum and caught the bypass.
SGLang's native differences are printed next to B200's for information; they are not gated.
"""

import json
import sys


def main() -> int:
    rep = json.load(open(sys.argv[1]))
    ref = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    env = rep["environment"]
    print(
        "gpu:", env.get("gpu"), "| sglang:", env.get("sglang"), "| tp:", sorted(rep["tp"], key=int)
    )
    bad = []
    for tp, t in sorted(rep["tp"].items(), key=lambda x: int(x[0])):
        for name, ranks in t["rows"].items():
            for rank, v in ranks.items():
                for key in ("bound_equals_canonical", "restored_equals_native"):
                    if v.get(key) is not True:
                        bad.append(f"TP{tp} {name} {rank}: {key}={v.get(key)}")
                # The AdaLN tables have no prefix measurement (no rows); the key is absent there.
                if v.get("bound_prefix_equals_rows") is False:
                    bad.append(f"TP{tp} {name} {rank}: bound_prefix_equals_rows=False")
        if tp != "1":
            eq = t.get("bound_equals_tp1")
            if eq is None:
                bad.append(f"TP{tp}: bound_equals_tp1 missing")
            else:
                bad += [f"TP{tp} {k}: bound_equals_tp1 false" for k, ok in eq.items() if not ok]
        if "full_model" not in t or "plugin" not in t:
            bad.append(
                f"TP{tp}: full_model or plugin section missing (report from an older commit?)"
            )
            continue
        fm = t["full_model"]
        for rank, r in ((k, v) for k, v in fm.items() if k.startswith("rank")):
            if r["readback"]:
                bad.append(f"TP{tp} full_model {rank}: readback {r['readback']}")
            wrong = {m: c for m, c in r["calls"].items() if c != fm["forwards"]}
            if wrong:
                bad.append(f"TP{tp} full_model {rank}: calls != forwards: {wrong}")
        for rank, p in ((k, v) for k, v in t["plugin"].items() if k.startswith("rank")):
            for key in ("hooked", "installed", "repeatable"):
                if p.get(key) is not True:
                    bad.append(f"TP{tp} plugin {rank}: {key}={p.get(key)}")
            if p.get("readback"):
                bad.append(f"TP{tp} plugin {rank}: readback {p['readback']}")
            for key in ("spectrum_refused", "bypass_caught"):
                if not p.get(key):
                    bad.append(f"TP{tp} plugin {rank}: {key} missing")
    print("\nSGLang native max |diff| vs training, largest over ranks (informational):")
    print(f"{'output':20} " + " ".join(f"{'TP'+tp:>16}" for tp in sorted(rep["tp"], key=int)))
    for name in next(iter(rep["tp"].values()))["rows"]:
        cells = []
        for tp in sorted(rep["tp"], key=int):
            here = max(v["native_max_abs_diff"] for v in rep["tp"][tp]["rows"][name].values())
            there = None
            if ref and tp in ref["tp"]:
                there = max(v["native_max_abs_diff"] for v in ref["tp"][tp]["rows"][name].values())
            cells.append(f"{here:g}" + (f" (B200 {there:g})" if there is not None else ""))
        print(f"{name:20} " + " ".join(f"{c:>16}" for c in cells))
    print()
    for b in bad:
        print("FAIL", b)
    print("RESULT:", "PASS" if not bad else f"FAIL ({len(bad)})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
