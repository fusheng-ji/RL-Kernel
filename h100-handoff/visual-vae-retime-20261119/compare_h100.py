#!/usr/bin/env python
"""Compare the new H100 evidence with the H100 run at 1e1ff922.

    python compare_h100.py reference/evidence-h100-1e1ff922.json <OUT>/evidence-h100.json

Exit 0 when every required verdict is true and every sha256 under outputs / checks /
inputs / manifest is unchanged. Prints the timing medians of both runs.
"""

import json
import re
import sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SECTIONS = ("outputs", "checks", "inputs", "manifest")
REQUIRED = (
    "run_to_run",
    "layout",
    "diffusers_parity_pinned",
    "contract_batch_invariant",
    "tiling_deterministic",
    "lut_equals_cuda_release",
    "normalization_cpu_cuda_bitwise",
)


def digests(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from digests(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from digests(v, f"{path}[{i}]")
    elif isinstance(node, str) and HEX64.match(node):
        yield path, node


def main() -> int:
    old, new = (json.load(open(p)) for p in sys.argv[1:3])
    ok = True
    print("commit:", new["rl_kernel_commit"][:8], "dirty:", new["rl_kernel_dirty"])
    print("device:", new["environment"]["device_name"], new["environment"]["compute_capability"])
    failed = [k for k in REQUIRED if new["verdicts"].get(k) is not True]
    print("required verdicts:", "all true" if not failed else f"FALSE: {failed}")
    ok &= not failed
    for k in sorted(set(old["verdicts"]) - set(REQUIRED)):
        print(f"  {k}: {old['verdicts'][k]} -> {new['verdicts'].get(k)}")
    a = {p: h for s in SECTIONS for p, h in digests(old[s], s)}
    b = {p: h for s in SECTIONS for p, h in digests(new[s], s)}
    changed = sorted(p for p in a.keys() | b.keys() if a.get(p) != b.get(p))
    print(f"sha256 fields: {len(a)} old, {len(b)} new, {len(changed)} differ")
    for p in changed:
        print(f"  {p}: {a.get(p, '-')[:12]} -> {b.get(p, '-')[:12]}")
    ok &= not changed
    print(f"\n{'timing (median ms)':40} {'1e1ff922':>10} {'new':>10}")
    for impl, ops in new["timing"].items():
        for op, t in ops.items():
            before = old["timing"].get(impl, {}).get(op, {}).get("median_ms")
            print(f"{impl + ' / ' + op:40} {before or float('nan'):10.1f} {t['median_ms']:10.1f}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
