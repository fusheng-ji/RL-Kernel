# H100 re-run, 2026-11-13

Which open PR heads were re-run on H100 and why. A re-run happens when the code changed since the earlier Hopper measurement; otherwise it is stated why not.

| PR | head | action |
|---|---|---|
| #557 | `7261a37a` (local) | re-run with two fixes made in #543: unit tests, chain preflight, layers and **l2** pass on EP4 (see its README); the earlier `5294b766` run (check scripts, EP4 MoE gate) is in its README |
| #548 | `2202486d` | re-run: both framework rows now PASS |
| #536 | `e5373328` | re-run: correctness suite |
| #495 | `fe2768f2` | re-run: logp test suite |
| #491 | `8c03d134` | re-run: tests and 2/4-GPU collective benchmark (no 8-GPU) |
| #538 | `4b2d8323` | not re-run: the tree is identical to the measured `e23c4a97` |
| #547 | `e78e4ae9` | not re-run: since the measured `6b8b05e5` only `pyproject.toml` (version, description) and an Ascend `.asc` file changed |
| #503, #507, #543, #544, #549 | | not run one by one: they are layers of the #557 stack; the stack-top run above covers the shared code (the GDN reference change in #503/#507/#557). the operator-case id and `load_checkpoint` fixes are local commits in #543 merged up to #544, #549, #557, #551 (not pushed) |
| #551 | `e09fa18e` (local) | not re-run; it only carries the same two merged fixes |
