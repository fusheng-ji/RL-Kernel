import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "source"
sys.path.insert(0, str(SOURCE))

import json
import torch
from rl_engine.backends.cuda.logprob.logp import DeterministicLogpCUDAOp

op = DeterministicLogpCUDAOp()
records = []
for logits, ids, baseline in torch.load(Path(__file__).resolve().parent.parent / "forward-baseline.pt", weights_only=True):
    logits = logits.cuda().requires_grad_(True)
    ids = ids.cuda()
    actual = op(logits, ids).detach().cpu()
    identical = torch.equal(actual.view(torch.uint8), baseline.view(torch.uint8))
    records.append({'dtype': str(logits.dtype), 'shape': list(logits.shape), 'bitwise_equal': identical})
(Path(__file__).resolve().parent / "forward-compatibility.json").write_text(json.dumps(records, indent=2) + '\n')
assert all(row['bitwise_equal'] for row in records)
print(f'{len(records)} pre-change binary comparisons: bitwise identical')
