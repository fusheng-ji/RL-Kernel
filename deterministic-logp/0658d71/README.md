# Deterministic CUDA LogP validation figures

Source implementation: `0658d7140c274d8d0b9371c33b55e9caf0975e59`.
Hardware: one NVIDIA B200. These assets visualize recorded acceptance results; generating them does not rerun the CUDA tests.

## Acceptance results

![Acceptance results](validation-overview.png)

The main suite contains 701 passes, 0 failures and 30 skips. The skipped cases are 15 SM90/Hopper-only tests and 15 Ascend-only tests, neither applicable to B200. Eight additional FP64 order 9–10 cases passed and are displayed separately from the main suite. The report also records 21 bitwise forward comparisons, 48 backend profiles and three strict deterministic GRPO steps.

## Numerical results

![Derivative error by order](derivative-errors.png)

Each point is the maximum absolute error within a precision, derivative order and method, aggregated from the recorded recursive acceptance cases. Reverse/mixed derivatives and nested JVPs are separate lines. Comparisons use independent PyTorch references. All plotted cases passed their individual assertions. Dashed lines show the common absolute tolerances for orders 4 and above; order 1–3 case-specific tolerances are in [derivative-errors.csv](derivative-errors.csv). Relative tolerance is zero. Exact-zero measurements are noted and omitted from logarithmic axes, rather than assigned fabricated positive values. A line break caused by an exact zero is not an untested order.

Inputs and direction vectors are bounded test fixtures. Error trends in this figure do not establish trends for larger workloads or untested derivative orders. These figures contain no latency or memory benchmarks.

## Algorithm and tested range

![Finite validation range](tested-order-scope.png)

The algorithm has no hard-coded derivative-order limit: derivative rules recurse through the same differentiable primitives. Higher orders do not require new order-specific backward kernels. Compute/memory requirements and floating-point error depend on the workload, so this architectural property is distinct from validation at every possible order.

FP32, FP16, BF16 and explicit FP64 reverse-mode directional/mixed derivatives and nested JVPs were tested at orders 1–8. FP64 reverse-mode directional/mixed derivatives were additionally tested at orders 9–10. Forward-mode coverage stops at order 8. Gray cells mean **not tested**, not failed or unsupported. Orders beyond the matrix are also untested. The scope is single-GPU acceptance, including dense/indexed/out and transform combinations; it is not a multi-GPU or production-model training guarantee.

## Provenance and reproduction

- [Full recorded report](validation-report.json): individual results, skip reasons, extension/source provenance, profiler evidence and numerical errors.
- [Compact figure data](figure-data.json) and [per-case error CSV](derivative-errors.csv).
- [SHA-256 manifest](manifest.json): hashes of the files used for this visualization.
- PNG files support direct GitHub PR embedding; matching SVG files provide vector exports.

Regenerate the figures from the uploaded report:

```bash
python -m pip install matplotlib==3.11.2 numpy==2.4.6
python generate_figures.py
```

To rerun the underlying GPU acceptance, use `runs/b200-autograd/run-recursive-validation.sh` from the original acceptance workspace. The original report includes its absolute rerun commands and input artifact hashes; this asset branch alone does not contain the complete GPU test workspace.

PR-ready embeds:

```markdown
![Acceptance results](https://raw.githubusercontent.com/fusheng-ji/RL-Kernel/pr_assest/deterministic-logp/0658d71/validation-overview.png)
![Derivative error by order](https://raw.githubusercontent.com/fusheng-ji/RL-Kernel/pr_assest/deterministic-logp/0658d71/derivative-errors.png)
![Algorithm and tested range](https://raw.githubusercontent.com/fusheng-ji/RL-Kernel/pr_assest/deterministic-logp/0658d71/tested-order-scope.png)
```
