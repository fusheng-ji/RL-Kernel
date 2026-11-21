# SPDX-License-Identifier: Apache-2.0
"""AReaL adapter: strict train/rollout logp alignment without editing AReaL.

Entry points (see ``docs/integrations/orchestrators/areal/README.md``):

* ``worker``      replacement actor worker command (installs the patches)
* ``run_trainer`` runs an AReaL training script and tees committed stats to JSONL
* ``verify``      inventory / preflight / run / analyze CLI

Submodules are imported explicitly; importing this package pulls in nothing.
"""
