import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "source"
sys.path.insert(0, str(SOURCE))

import hashlib
import json
import subprocess
import warnings

import torch
from torch.profiler import profile, ProfilerActivity
from rl_engine import _C
from rl_engine.runtime.registry import kernel_registry
from rl_engine.backends.cuda.logprob.logp import FusedLogpGenericOp, DeterministicLogpCUDAOp
from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import TritonBatchInvariantLogpOp

assert Path(_C.__file__).resolve().parent == SOURCE / "rl_engine"

report = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
    'extension': _C.__file__,
    'extension_sha256': hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
    'registry': {}, 'backends': [],
}
for name in ['logp', 'logp_deterministic', 'batch_invariant_logp']:
    op = kernel_registry.get_op(name)
    report['registry'][name] = f'{type(op).__module__}.{type(op).__name__}'
# Prime CUPTI before collecting the first backend's trace.
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
    torch.ones(16, device='cuda').sum().item()
    torch.cuda.synchronize()
for cls in [FusedLogpGenericOp, DeterministicLogpCUDAOp, TritonBatchInvariantLogpOp]:
    op = cls()
    x = torch.randn(4, 257, device='cuda', requires_grad=True)
    ids = torch.tensor([0, 17, 129, 256], device='cuda')
    op(x, ids)
    torch.cuda.synchronize()
    row = {'backend': f'{cls.__module__}.{cls.__name__}'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            y = op(x, ids)
            row['output_requires_grad'] = y.requires_grad
            row['grad_fn'] = type(y.grad_fn).__name__ if y.grad_fn is not None else None
            if y.requires_grad:
                grad = torch.autograd.grad(y.sum(), x, allow_unused=True)[0]
                row['input_gradient_present'] = grad is not None
            else:
                row['input_gradient_present'] = False
            torch.cuda.synchronize()
        row['warnings'] = [str(item.message) for item in caught]
    row['cuda_kernels'] = sorted({event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA})
    report['backends'].append(row)
(Path(__file__).resolve().parent / "logp-backends.json").write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
