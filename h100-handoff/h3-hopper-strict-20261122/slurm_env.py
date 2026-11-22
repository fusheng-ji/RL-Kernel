# SPDX-License-Identifier: Apache-2.0
"""Run an RL-Kernel multi-rank validation tool under torchrun.

The tools read the SLURM rank variables (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID) and open their
own TCP rendezvous at MASTER_ADDR:MASTER_PORT. Under torchrun this maps RANK / WORLD_SIZE /
LOCAL_RANK onto the SLURM names and moves the tool's rendezvous one port up (torchrun's own store
holds MASTER_PORT). Under srun (SLURM_PROCID already set) nothing is changed.

    torchrun --nnodes 2 --node-rank <0|1> --nproc-per-node 4 --master-addr <node-a> --master-port 29700 \\
        slurm_env.py tools/validation/models/h3_parallel_full_model.py --tp 4 --cp 2 ...
"""

import os
import runpy
import sys

if "SLURM_PROCID" not in os.environ:
    os.environ["SLURM_PROCID"] = os.environ["RANK"]
    os.environ["SLURM_NTASKS"] = os.environ["WORLD_SIZE"]
    os.environ["SLURM_LOCALID"] = os.environ["LOCAL_RANK"]
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 1)
tool, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(tool, run_name="__main__")
