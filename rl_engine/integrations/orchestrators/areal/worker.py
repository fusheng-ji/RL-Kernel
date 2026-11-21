# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Drop-in replacement for AReaL's worker command.

Set ``actor.scheduling_spec[0].cmd`` to::

    python3 -m rl_engine.integrations.orchestrators.areal.worker

The scheduler appends its usual arguments. This module registers the RL-Kernel
patches selected by ``RL_KERNEL_AREAL_MODE`` and then runs
``areal.infra.rpc.rpc_server`` unchanged in the same process.
"""

from __future__ import annotations

import runpy
import sys

RPC_SERVER_MODULE = "areal.infra.rpc.rpc_server"


def main(argv: list[str] | None = None) -> None:
    from rl_engine.integrations.orchestrators.areal.patches import install

    mode = install()
    print(f"[RL-Kernel][areal] worker mode={mode or 'off'}", flush=True)
    args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [RPC_SERVER_MODULE, *args]
    runpy.run_module(RPC_SERVER_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
