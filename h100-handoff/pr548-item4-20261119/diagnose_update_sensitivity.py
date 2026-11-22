# SPDX-License-Identifier: Apache-2.0
"""Is a step-0 update difference explained by the gradient difference through AdamW?

    PYTHONPATH=<src> <py> -I diagnose_update_sensitivity.py --reference <ref>/actor-fp32-0 \
        --other <ref>/actor-fp32-1 --other <ref>/actor-fp32-impl \
        --other <cand>/megatron-fp32-actor-tp2 --other <cand>/megatron-fp32-actor-tp4 \
        --output <out>/update-sensitivity.json [--copy-to <out>/tensors] [--scan]

CPU only; reads evidence tensors through ``fixed_contract.read_tensor`` (hash-checked), writes
nothing into the run directories. For each run and each listed step-0 update tensor it
recomputes the first AdamW step in FP64 from that run's own unclipped gradient and grad norm:

    g_c = g * min(1, max_grad_norm / (grad_norm + 1e-6))
    u   = -lr * wd * p0 - lr * g_c / (|g_c| + eps)        (step 1: m_hat = g_c, sqrt(v_hat) = |g_c|)

Both runs start from the same ``p0``, so the predicted update difference is
``-lr * (a_other - a_reference)`` with ``a = g_c / (|g_c| + eps)``; the weight-decay term cancels.
``residual`` is how far the observed difference is from that prediction (FP32 rounding of the
weights only, if the hypothesis holds). ``--scan`` repeats it for every step-0 update tensor.
``--copy-to`` copies the listed tensors' step-0 gradient and update files (a few KB each).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.validation.models.qwen3_ppo.fixed_contract import CONFIG, OPTIMIZER, read_tensor

DEFAULT = (
    "model.layers.17.post_attention_layernorm.weight",
    "model.layers.24.post_attention_layernorm.weight",
    "model.layers.8.input_layernorm.weight",
)


def scaled(run, report, name):
    """Return ``g / (|g| + eps)`` in FP64 for the run's clipped step-0 gradient of ``name``."""
    g = read_tensor(run, report["artifacts"][f"0/gradient/{name}"]).double()
    norm = float(read_tensor(run, report["artifacts"]["0/grad_norm"]))
    g_c = g * min(1.0, CONFIG.max_grad_norm / (norm + 1e-6))
    return g, g_c, g_c / (g_c.abs() + OPTIMIZER["eps"])


def compare(ref, ref_report, run, report, name, top):
    """Return the observed and predicted step-0 update difference of ``name`` for one run."""
    g_r, gc_r, a_r = scaled(ref, ref_report, name)
    g_o, _, a_o = scaled(run, report, name)
    u_r = read_tensor(ref, ref_report["artifacts"][f"0/update/{name}"]).double()
    u_o = read_tensor(run, report["artifacts"][f"0/update/{name}"]).double()
    observed = (u_o - u_r).flatten()
    predicted = (-OPTIMIZER["lr"] * (a_o - a_r)).flatten()
    order = observed.abs().topk(min(top, observed.numel())).indices
    eps_ratio = (gc_r.abs() / OPTIMIZER["eps"]).flatten()
    return dict(
        elements=observed.numel(),
        max_abs_observed=float(observed.abs().max()),
        max_abs_predicted=float(predicted.abs().max()),
        residual=float((observed - predicted).abs().max()),
        max_abs_gradient_difference=float((g_o - g_r).abs().max()),
        reference_update_max_abs=float(u_r.abs().max()),
        elements_below_eps=int((eps_ratio < 1).sum()),
        top=[
            dict(
                index=int(i),
                gradient_reference=float(g_r.flatten()[i]),
                gradient_other=float(g_o.flatten()[i]),
                clipped_gradient_over_eps=float(eps_ratio[i]),
                update_difference=float(observed[i]),
                predicted_update_difference=float(predicted[i]),
            )
            for i in order
        ],
    )


def main():
    """Write the per-run, per-tensor comparison JSON; exit 0 whatever it finds."""
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--other", type=Path, action="append", required=True)
    p.add_argument("--tensor", action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--copy-to", type=Path)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("refusing to overwrite --output")
    names = a.tensor or list(DEFAULT)
    ref_report = json.loads((a.reference / "report.json").read_text())
    result = dict(optimizer=OPTIMIZER, max_grad_norm=CONFIG.max_grad_norm, runs={})
    for run in [a.reference, *a.other]:
        report = json.loads((run / "report.json").read_text())
        if a.copy_to:
            for name in names:
                for key in (f"0/gradient/{name}", f"0/update/{name}", "0/grad_norm"):
                    row = report["artifacts"][key]
                    target = a.copy_to / run.name / key.replace("/", "__")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(run / row["path"], str(target) + ".pt")
        if run == a.reference:
            continue
        entry = dict(
            grad_norm=float(read_tensor(run, report["artifacts"]["0/grad_norm"])),
            tensors={n: compare(a.reference, ref_report, run, report, n, a.top) for n in names},
        )
        if a.scan:
            scan = {}
            for key in ref_report["artifacts"]:
                if key.startswith("0/update/"):
                    n = key[len("0/update/") :]
                    row = compare(a.reference, ref_report, run, report, n, 1)
                    scan[n] = {
                        k: row[k]
                        for k in (
                            "max_abs_observed",
                            "max_abs_predicted",
                            "residual",
                            "elements_below_eps",
                        )
                    }
            entry["scan"] = dict(
                tensors=len(scan),
                max_residual=max(v["residual"] for v in scan.values()),
                worst_residual=max(scan, key=lambda n: scan[n]["residual"]),
                rows=scan,
            )
        result["runs"][run.name] = entry
        print(
            run.name,
            json.dumps(
                {
                    n: {k: v[k] for k in ("max_abs_observed", "max_abs_predicted", "residual")}
                    for n, v in entry["tensors"].items()
                }
            ),
            flush=True,
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
