# AReaL driver-node slowdown: notes for continuing on H100

Base: `perf/areal-training-overhead-round1` at e3b181ec. This branch adds only this
directory. **It contains no change to RL-Kernel code and no fix for the slowdown.**

The slowdown has not been reproduced or explained. No placement A/B was run. The
hypotheses in the original handoff (cpusets, threads, NUMA, shared filesystem, file
descriptors) are all still open, and the acceptance check (actor train within 5% between
the layouts) has to be measured on H100.

The notes below come from bringing up the native arm in another environment with the
versions listed in the report. None of them depends on the GPU model.

## Check the runs

1. **`verify run` exits 0 when a native-only run fails.** Without a strict arm, `analyze`
   returns 0 even if the trainer exited non-zero and no step ran. Seen once, and the code
   path in `verify.py` confirms it. Check
   `returncode` in `<out>/native/run.json` and the number of rows in
   `<out>/native/stats.jsonl`, not the exit code or the Slurm state.
2. **Do not start `verify` from the RL-Kernel source tree** when RL-Kernel is installed
   non-editable. The source tree shadows the installed package and `_C` is then missing.
   The log says `Core binary extension (_C) unavailable`.

## Check on the H100 environment

3. **Worker import time.** AReaL's local scheduler gives a forked worker 60 s to get ready
   (`areal/infra/scheduler/local.py:300` in v2.1.1). With the venv on a network
   filesystem, `import areal.infra.rpc.rpc_server` took 41-44 s (warm, on a workstation), and
   one run on a cluster node died on this
   limit. From local disk it took 6.3 s with the same CPU time. Staging the venv on
   node-local disk fixed it, with AReaL unmodified. On H100, time this import on the driver
   node and on another node:

   ```bash
   time python -c "import areal.infra.rpc.rpc_server"
   ```

   This concerns start-up only. It is not evidence about the per-step train slowdown.
4. **Gradient clipping fallback.** Megatron 0.19.0 warns `Transformer Engine and Apex are
   not installed` from `optimizer/clip_grads.py` even with TE present: TE 2.11 has no
   `multi_tensor_scale_tensor`, so clipping uses Megatron's local code. The layers do use
   TE. Look for the same warning in the H100 logs, so that both setups are known to match.

## Transformer Engine 2.11

5. **Two CUDA runtimes.** TE's cuDNN loader opens both `libcudart.so.12` and
   `libcudart.so.13` and raises `Multiple libcudart libraries found` if both load, for
   example with a torch cu12x wheel on a host with CUDA 13 in its linker cache. Run
   `te_fused_attention_check.py` on one GPU to check. If it fails, apply
   `te_cudnn_loader_cuda12_only.patch` in `3rdparty/cudnn-frontend` and rebuild. Do not
   set `NVTE_FUSED_ATTN=0` instead: that changes the attention path of the native arm. If
   the check passes, no patch is needed, and stock TE should be kept.
6. **Building from source.** `nccl.h`, `nvtx3` and `cudnn.h` from the venv's `nvidia/*`
   wheels have to be on `CPATH`, with unversioned `libnccl.so` and `libcudnn.so` links on
   `LIBRARY_PATH`. With `--no-deps`, also install `onnxscript`.

## Next steps on H100

1. Run the original handoff's sync native runs (3-5 steps) with the actor on the driver
   node and off it. Keep both logs.
2. During both runs, record for the actor and vLLM processes: `taskset -cp <pid>`,
   `Cpus_allowed_list` in `/proc/<pid>/status`, thread counts and NUMA placement.
3. Change one thing per run: `srun --cpu-bind=none`, pinning the driver to other cores,
   limiting threads, moving the driver and Ray head to a CPU node.
4. Redact cluster, partition, node and job names, hostnames, absolute paths, user names,
   IPs and GPU UUIDs before anything leaves the cluster.
