"""B200 acceptance matrix; run against the isolated deployed checkout."""
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "source"
sys.path.insert(0, str(SOURCE))

import pytest
import torch

from rl_engine.backends.cuda.logprob.logp import FusedLogpGenericOp, DeterministicLogpCUDAOp
from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import TritonBatchInvariantLogpOp

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SHAPES = [(1, 1), (5, 257), (8, 4099), (4, 128256)]


@pytest.fixture(params=[FusedLogpGenericOp, DeterministicLogpCUDAOp, TritonBatchInvariantLogpOp])
def backend(request, record_property):
    assert torch.cuda.get_device_capability() == (10, 0)
    op = request.param()
    record_property('backend', f'{type(op).__module__}.{type(op).__name__}')
    record_property('device', torch.cuda.get_device_name())
    if isinstance(op, DeterministicLogpCUDAOp):
        record_property('forward', '_C.deterministic_logp_forward_with_lse when gradients enabled')
        record_property('backward', '_C.deterministic_logp_backward')
    elif isinstance(op, FusedLogpGenericOp):
        record_property('forward', '_C.fused_logp')
        record_property('backward', '_FusedLogpAutograd: PyTorch FP32 softmax VJP')
    else:
        record_property('forward', '_batch_invariant_logp_kernel (Triton)')
        record_property('backward', '_batch_invariant_logp_bwd_kernel (Triton)')
    return op


def inputs(shape, dtype, seed=1234):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    logits = torch.randn(shape, device='cuda', dtype=dtype, generator=generator)
    labels = torch.randint(shape[-1], shape[:-1], device='cuda', generator=generator)
    return logits, labels


def reference(logits, labels):
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def bitwise(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('seed', [1234, 2026])
def test_forward_matrix(backend, dtype, shape, seed, record_property):
    logits, labels = inputs(shape, dtype, seed)
    actual = backend(logits, labels)
    expected = reference(logits.double(), labels)
    atol = {torch.float32: 2e-5, torch.float16: 8e-3, torch.bfloat16: 6.25e-2}[actual.dtype]
    record_property('max_abs_error_vs_fp64', (actual.double() - expected).abs().max().item())
    record_property('absolute_tolerance', atol)
    record_property('input_dtype', str(dtype))
    record_property('output_dtype', str(actual.dtype))
    assert actual.shape == labels.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=0)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('vocab', [257, 128256])
def test_backward_and_optimizer(backend, dtype, vocab, record_property):
    logits, labels = inputs((4, vocab), dtype)
    candidate = torch.nn.Parameter(logits.clone())
    expected = torch.nn.Parameter(logits.clone())
    weights = torch.tensor([0.25, -0.5, 0.75, -1.0], device='cuda')
    candidate_optimizer = torch.optim.SGD([candidate], lr=0.1)
    reference_optimizer = torch.optim.SGD([expected], lr=0.1)
    original = candidate.detach().clone()
    tolerance = {torch.float32: 3e-6, torch.float16: 5e-4, torch.bfloat16: 4e-3}[dtype]
    gradient_errors = []
    update_errors = []
    for _ in range(3):
        candidate_optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        (backend(candidate, labels).float() * weights).sum().backward()
        (reference(expected.float(), labels) * weights).sum().backward()
        assert candidate.grad is not None and torch.isfinite(candidate.grad).all()
        gradient_errors.append((candidate.grad.float() - expected.grad.float()).abs().max().item())
        torch.testing.assert_close(candidate.grad, expected.grad, atol=tolerance, rtol=0)
        candidate_optimizer.step()
        reference_optimizer.step()
        update_errors.append((candidate.float() - expected.float()).abs().max().item())
        torch.testing.assert_close(candidate, expected, atol=tolerance * 2, rtol=0)
    record_property('max_gradient_error', max(gradient_errors))
    record_property('max_parameter_error', max(update_errors))
    record_property('gradient_atol', tolerance)
    record_property('parameter_atol', tolerance * 2)
    record_property('optimizer_steps', 3)
    assert not torch.equal(candidate, original)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('vocab', [257, 128256])
def test_forward_layout_bitwise(backend, dtype, vocab):
    logits, labels = inputs((7, vocab), dtype)
    baseline = backend(logits, labels)
    for _ in range(3):
        bitwise(backend(logits, labels), baseline)
    for chunk in [1, 2, 4]:
        result = torch.cat([backend(logits[i:i+chunk], labels[i:i+chunk]) for i in range(0, 7, chunk)])
        bitwise(result, baseline)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device='cuda')
    bitwise(backend(logits[order], labels[order]), baseline[order])
    padded, padded_labels = inputs((16, vocab), dtype, 2026)
    padded[3:10] = logits
    padded_labels[3:10] = labels
    bitwise(backend(padded, padded_labels)[3:10], baseline)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('vocab', [257, 128256])
def test_gradient_layout_bitwise(backend, dtype, vocab):
    logits, labels = inputs((7, vocab), dtype)
    weights = torch.linspace(-1, 1, 7, device='cuda')

    def gradient(x, ids, upstream):
        x = x.detach().clone().requires_grad_(True)
        return torch.autograd.grad((backend(x, ids).float() * upstream).sum(), x)[0]

    baseline = gradient(logits, labels, weights)
    bitwise(gradient(logits, labels, weights), baseline)
    for chunk in [1, 2, 4]:
        result = torch.cat([gradient(logits[i:i+chunk], labels[i:i+chunk], weights[i:i+chunk]) for i in range(0, 7, chunk)])
        bitwise(result, baseline)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device='cuda')
    bitwise(gradient(logits[order], labels[order], weights[order]), baseline[order])
    padded, padded_labels = inputs((16, vocab), dtype, 2026)
    padded[3:10] = logits
    padded_labels[3:10] = labels
    padded_weights = torch.zeros(16, device='cuda')
    padded_weights[3:10] = weights
    padded_gradient = gradient(padded, padded_labels, padded_weights)
    bitwise(padded_gradient[3:10], baseline)
    assert torch.count_nonzero(padded_gradient[:3]) == 0
    assert torch.count_nonzero(padded_gradient[10:]) == 0


@pytest.mark.parametrize('dtype', DTYPES)
def test_policy_parameter_updates(backend, dtype, record_property):
    torch.manual_seed(1234)
    candidate = torch.nn.Linear(8, 257, device='cuda')
    expected = torch.nn.Linear(8, 257, device='cuda')
    expected.load_state_dict(candidate.state_dict())
    features, _ = inputs((4, 8), torch.float32)
    labels = torch.tensor([0, 17, 129, 256], device='cuda')
    weights = torch.tensor([0.25, -0.5, 0.75, -1.0], device='cuda')
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1) for model in [candidate, expected]]
    original = candidate.weight.detach().clone()
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    gradient_errors = []
    parameter_errors = []
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        (backend(candidate(features).to(dtype), labels).float() * weights).sum().backward()
        (reference(expected(features).to(dtype).float(), labels) * weights).sum().backward()
        for actual, ref in zip(candidate.parameters(), expected.parameters()):
            assert actual.grad is not None and torch.isfinite(actual.grad).all()
            gradient_errors.append((actual.grad - ref.grad).abs().max().item())
            torch.testing.assert_close(actual.grad, ref.grad, atol=tolerance, rtol=0)
        for optimizer in optimizers:
            optimizer.step()
        for actual, ref in zip(candidate.parameters(), expected.parameters()):
            parameter_errors.append((actual - ref).abs().max().item())
            torch.testing.assert_close(actual, ref, atol=tolerance, rtol=0)
    record_property('max_policy_gradient_error', max(gradient_errors))
    record_property('max_policy_parameter_error', max(parameter_errors))
    record_property('atol', tolerance)
    record_property('optimizer_steps', 3)
    record_property('policy', 'Linear(8, 257), FP32 parameters, selected execution dtype')
    assert not torch.equal(candidate.weight, original)
