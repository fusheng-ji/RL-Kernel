#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step-3 counterfactual probe for ws1_dual_trajectory (handoff branch; not in #571 PR).

Tests whether B200 step-3 single-vs-batched spike (~0.62) requires divergent step-2 latents
(Euler accumulation) or persists with a shared step-2 latent (layout sensitivity at step 3).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

COMMIT = "eb7840bb6ad59c0025dc7c655a9dd9e22f950500"
SIGMAS = (1.0, 0.75, 0.5, 0.25, 0.0)
STEP3 = 3
REPO: Path


def _setup_path(repo: Path, h3_c_so_dir: Path | None) -> None:
    global REPO
    REPO = repo.resolve()
    sys.path.insert(0, str(REPO))
    if h3_c_so_dir is not None:
        so = list(h3_c_so_dir.glob("rl_engine/_C*.so"))
        if so:
            import shutil

            dst = REPO / "rl_engine"
            for s in so:
                shutil.copy2(s, dst / s.name)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def load_latent_pair(latent_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    single = load_file(str(latent_dir / "rlkernel_strict-single.safetensors"))
    batched = load_file(str(latent_dir / "rlkernel_strict-batched.safetensors"))
    key_v, key_a = f"{STEP3 - 1}.latent.video", f"{STEP3 - 1}.latent.audio"

    def pick(st):
        return {
            "video": st[key_v].cuda().float().clone(),
            "audio": st[key_a].cuda().float().clone(),
        }

    return pick(single), pick(batched)


def run_step3_velocity(
    model,
    base,
    noise_for_other,
    *,
    latent: dict[str, torch.Tensor],
    layout: str,
    candidate: bool,
) -> dict[str, torch.Tensor]:
    from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution

    video = latent["video"].clone()
    audio = latent["audio"].clone()
    sigma, sigma_next = SIGMAS[STEP3], SIGMAS[STEP3 + 1]
    enc = base["encoder_hidden_states"].detach()
    trace = FullModelTrace()
    with torch.no_grad(), full_model_execution(
        model, candidate=candidate, trace=trace, strict_boundaries=candidate
    ):
        if layout == "single":
            hs, ahs, es = video.bfloat16(), audio.bfloat16(), enc
        else:
            other_v = noise_for_other["video"].bfloat16()
            other_a = noise_for_other["audio"].bfloat16()
            hs = torch.cat([other_v, video.bfloat16()])
            ahs = torch.cat([other_a, audio.bfloat16()])
            es = torch.cat([enc * 0.5, enc])
        inputs = {
            **base,
            "hidden_states": hs,
            "audio_hidden_states": ahs,
            "encoder_hidden_states": es,
            "timestep": torch.tensor([0.0, sigma], device=hs.device),
        }
        v_video, v_audio = model(**inputs)
        row = 0 if layout == "single" else 1
        return {
            "video": v_video[row : row + 1].float().clone(),
            "audio": v_audio[row : row + 1].float().clone(),
            "sigma_delta": sigma_next - sigma,
        }


def run_full_trajectory(
    checkpoint: Path,
    *,
    seed: int,
    output_dir: Path,
) -> None:
    """Run upstream h3_dual_trajectory --strict-only to materialize safetensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "tools/validation/models/h3_dual_trajectory.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_dir),
        "--seed",
        str(seed),
        "--strict-only",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    subprocess.run(cmd, check=True, env=env)


def compare_velocities(v_s: dict, v_b: dict) -> dict:
    return {
        "max_abs_velocity_video": _max_abs(v_s["video"], v_b["video"]),
        "max_abs_velocity_audio": _max_abs(v_s["audio"], v_b["audio"]),
    }


def classify_b200(b_shared_single: float, a_replay: float) -> str:
    if b_shared_single < 1e-3 and a_replay > 0.1:
        return "supports_latent_accumulation"
    if b_shared_single > 0.3:
        return "supports_step3_layout_sensitivity"
    return "inconclusive"


args_checkpoint: Path


def main() -> None:
    global args_checkpoint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=None,
        help="dir with rlkernel_strict-{single,batched}.safetensors; if missing, run full trajectory first",
    )
    parser.add_argument(
        "--skip-full-run",
        action="store_true",
        help="require --latent-dir with existing safetensors",
    )
    parser.add_argument(
        "--delta-scale-sweep",
        type=str,
        default="",
        help="optional comma-separated K values: batched latent = single + K*(batched-single) for arm A",
    )
    parser.add_argument(
        "--rl-kernel-repo",
        type=Path,
        default=None,
        help="RL-Kernel checkout at eb7840bb (default: repo root when script lives under handoff/b200-euler-step3-probe/)",
    )
    parser.add_argument(
        "--h3-c-so-dir",
        type=Path,
        default=None,
        help="optional tree containing rl_engine/_C*.so (Hopper-only hack; do not use on B200)",
    )
    args = parser.parse_args()
    args_checkpoint = args.checkpoint
    default_repo = Path(__file__).resolve().parents[2]
    repo = args.rl_kernel_repo or default_repo
    _setup_path(repo, args.h3_c_so_dir)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from diffusers import MiniMaxH3Transformer3DModel
    from tools.validation.models.h3_full_model import packed_inputs

    latent_dir = args.latent_dir or (args.output / "full_run_tensors")
    if not args.skip_full_run:
        if not (latent_dir / "rlkernel_strict-single.safetensors").exists():
            print("Running full 4-step strict trajectory for safetensors...", flush=True)
            run_full_trajectory(args.checkpoint, seed=args.seed, output_dir=latent_dir)
        else:
            print("Reusing existing safetensors in", latent_dir, flush=True)

    t0 = time.perf_counter()
    model = (
        MiniMaxH3Transformer3DModel.from_pretrained(
            str(args.checkpoint / "transformer"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    load_s = time.perf_counter() - t0
    base = packed_inputs("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    noise = {
        "video": torch.randn(base["hidden_states"].shape, generator=gen).cuda(),
        "audio": torch.randn(base["audio_hidden_states"].shape, generator=gen).cuda(),
    }
    other = {
        k: torch.randn(v.shape, generator=torch.Generator().manual_seed(77 + i)).cuda()
        for i, (k, v) in enumerate(noise.items())
    }

    l_single, l_batched = load_latent_pair(latent_dir)
    d_delta = {
        "max_abs_latent_video_at_step2": _max_abs(l_single["video"], l_batched["video"]),
        "max_abs_latent_audio_at_step2": _max_abs(l_single["audio"], l_batched["audio"]),
    }

    # A: natural step-3 entry latents (each layout uses its own step-2 end state)
    v_a_s = run_step3_velocity(
        model, base, other, latent=l_single, layout="single", candidate=True
    )
    v_a_b = run_step3_velocity(
        model, base, other, latent=l_batched, layout="batched", candidate=True
    )
    arm_a = compare_velocities(v_a_s, v_a_b)
    arm_a["name"] = "A_replay_step3_entry"

    # B: shared single step-2 latent
    v_b_s = run_step3_velocity(
        model, base, other, latent=l_single, layout="single", candidate=True
    )
    v_b_b = run_step3_velocity(
        model, base, other, latent=l_single, layout="batched", candidate=True
    )
    arm_b = compare_velocities(v_b_s, v_b_b)
    arm_b["name"] = "B_shared_single_step2_latent"

    # C: shared batched step-2 latent
    v_c_s = run_step3_velocity(
        model, base, other, latent=l_batched, layout="single", candidate=True
    )
    v_c_b = run_step3_velocity(
        model, base, other, latent=l_batched, layout="batched", candidate=True
    )
    arm_c = compare_velocities(v_c_s, v_c_b)
    arm_c["name"] = "C_shared_batched_step2_latent"

    cc = torch.cuda.get_device_capability()
    delta_sweep = []
    if args.delta_scale_sweep.strip():
        base_delta_v = l_batched["video"] - l_single["video"]
        base_delta_a = l_batched["audio"] - l_single["audio"]
        for k_str in args.delta_scale_sweep.split(","):
            k = float(k_str.strip())
            l_fake = {
                "video": l_single["video"] + k * base_delta_v,
                "audio": l_single["audio"] + k * base_delta_a,
            }
            v_s = run_step3_velocity(
                model, base, other, latent=l_single, layout="single", candidate=True
            )
            v_b = run_step3_velocity(
                model, base, other, latent=l_fake, layout="batched", candidate=True
            )
            cmp = compare_velocities(v_s, v_b)
            cmp["delta_scale_k"] = k
            delta_sweep.append(cmp)

    report = {
        "experiment": "h3_dual_trajectory_step3_probe",
        "seed": args.seed,
        "commit": COMMIT,
        "rl_kernel_repo": str(REPO),
        "execution_status": "complete",
        "device": torch.cuda.get_device_name(),
        "compute_capability": f"sm{cc[0]}{cc[1]}",
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "torch": torch.__version__,
        "load_model_seconds": load_s,
        "D_delta_at_step2": d_delta,
        "arms": {"A": arm_a, "B": arm_b, "C": arm_c},
        "optional_delta_scale_sweep": delta_sweep,
        "historical_b200_A_replay_from_pr_assets": {
            "max_abs_velocity_video_step3": 0.6215000152587891,
            "source": "pending/h3-full-transformer/reports/experiments/h3-full-transformer-b200/dual_trajectory_seed33.json",
            "note": "Use sm100 GPU run of this script to reproduce A/B on device; not substitutable with H100 forward.",
        },
        "conclusion_b200_style": classify_b200(
            arm_b["max_abs_velocity_video"], arm_a["max_abs_velocity_video"]
        ),
        "interpretation": {
            "supports_latent_accumulation": "B_shared max_abs << A_replay max_abs on video velocity at step 3",
            "supports_step3_layout_sensitivity": "B_shared max_abs still large (~same as A) with identical step-2 latent",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "seed33_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
