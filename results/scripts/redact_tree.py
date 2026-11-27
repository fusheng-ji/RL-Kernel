#!/usr/bin/env python3
"""Single redaction table for all handoff result trees. Reads paths from env (stage_push.local.sh)."""
import os
import re
import sys


def esc_path(key, repl):
    val = os.environ.get(key, "")
    if not val or "<" in val:
        return None
    return (re.escape(val), repl)


def build_rules():
    u = os.environ.get("LEAK_USER", "<USER>")
    j1 = os.environ.get("LEAK_JOB1", "<JOB_1>")
    j2 = os.environ.get("LEAK_JOB2", "<JOB_2>")
    g = os.environ.get("LEAK_GROUP", "<GROUP>")

    R = []
    for key, repl in (
        ("LEAK_PATH_WORK", "<WORK>"),
        ("LEAK_PATH_SCRATCH_USER", "<SCRATCH>"),
        ("LEAK_PATH_PROJECT_FS", "<PROJECT_FS>"),
        ("LEAK_PATH_HOME", "<HOME>"),
        ("LEAK_PATH_PROJECT_USER", "<PROJECT_FS>"),
    ):
        pair = esc_path(key, repl)
        if pair:
            R.append(pair)

    if j1 and "<" not in j1:
        R.append((rf"(?<!\d){re.escape(j1)}(?!\d)", "<JOB_1>"))
    if j2 and "<" not in j2:
        R.append((rf"(?<!\d){re.escape(j2)}(?!\d)", "<JOB_2>"))

    R += [
        (r"cpu_job_\d+\.log", "cpu_job.log"),
        (r"cpu-0\d\d", "node-c"),
        (r"--qos=\w+", "--qos=<QOS>"),
        (r"0x[0-9a-f]{16}", "<GUID>"),
        (r"(?i)[0-9A-F]{8}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]", "<PCI>"),
        (r"\b10(?:\.\d{1,3}){3}\b", "<IP>"),
        (r"\b192\.168(?:\.\d{1,3}){2}\b", "<IP>"),
        (r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b", "<IP>"),
        (r"GPU-[0-9a-f]{8}-[0-9a-f-]{27}", "<GPU-UUID>"),
    ]

    for var, repl in (
        ("LEAK_PAT_SCRATCH_A", "$SCRATCH_ROOT"),
        ("LEAK_PAT_SCRATCH_B", "SCRATCH_ROOT"),
        ("LEAK_REPL_E", "<E>"),
        ("LEAK_REPL_V", "<V>"),
        ("LEAK_REPL_E2", "<E2>"),
        ("LEAK_REPL_ENV", "<ENV>"),
        ("LEAK_PAT_USER_EXTRA", "<USER>"),
        ("LEAK_PAT_BUNDLE_A", "$BUNDLE/<BUNDLE_SCRIPTS_DIR>"),
        ("LEAK_PAT_BUNDLE_B", "$B/<BUNDLE_SCRIPTS_DIR>"),
        ("LEAK_PAT_BUNDLE_C", "/<BUNDLE_SCRIPTS_DIR>"),
        ("LEAK_PAT_BUNDLE_D", "/<BUNDLE_SCRIPTS_DIR>/"),
        ("LEAK_PAT_BUNDLE_E", "<BUNDLE_SCRIPTS_DIR>/"),
        ("LEAK_PAT_BUNDLE_UC", "BUNDLE_SCRIPTS"),
        ("LEAK_PAT_PARTITION", "<PARTITION>"),
        ("LEAK_PAT_NODE", "node-x"),
        ("LEAK_PAT_NODES", "<NODES>"),
        ("LEAK_PAT_CPU_PART", "<CPU_PARTITION>"),
        ("LEAK_PAT_CLUSTER_LOGIN", "<CLUSTER>"),
        ("LEAK_PAT_CLUSTER_HOST", "<CLUSTER>"),
        ("LEAK_PAT_FS", "<FS>"),
        ("LEAK_PAT_GROUP_FS", "<GROUP_FS>"),
    ):
        pat = os.environ.get(var, "")
        if pat and "<" not in pat:
            R.append((pat, repl))

    if u and "<" not in u:
        R.append((re.escape(u), "<USER>"))
    if g and "<" not in g:
        R.append((re.escape(g), "<GROUP>"))

    return R


def redact_text(s: str) -> str:
    for a, b in build_rules():
        s = re.sub(a, b, s)
    return s


def redact_file(path: str) -> None:
    s = open(path, errors="replace").read()
    open(path, "w").write(redact_text(s))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: redact_tree.py FILE...")
    for p in sys.argv[1:]:
        redact_file(p)


if __name__ == "__main__":
    main()