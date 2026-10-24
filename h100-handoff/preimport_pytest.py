"""Run pytest with megatron.core imported first (when it is installed).

`tests/models/qwen3_next/test_qwen3_next_training_checkpoint.py` monkeypatches `torch.cuda.device_count`
to 0 in its fixture. If `megatron.core` is imported for the first time after that, flashinfer (pulled in
by megatron.core) reads the device count at import time and raises "Invalid device id", so 15 of the 25
tests fail instead of passing. Importing megatron.core before pytest starts avoids that.
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: F401,E402

try:
    import megatron.core.optimizer.cpu_offloading.hybrid_optimizer  # noqa: F401,E402
except ImportError:  # megatron-core not installed: the dependent tests skip
    pass

import pytest  # noqa: E402

if __name__ == "__main__":  # guard: torch.multiprocessing.spawn re-imports this module in children
    sys.exit(pytest.main(sys.argv[1:]))
