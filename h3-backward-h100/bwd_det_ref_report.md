# Backward vs a deterministic diffusers reference (H100, sm90, torch 2.13.0+cu130)

Method (`det_diffusers_backward.py`): for #484-#489, gradients of three implementations on identical
inputs and upstream gradients: native diffusers replay, the same replay under
`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (the deterministic
reference), and the rl-kernel op. Each is run twice to test repeatability. Errors are max|err| / max|ref|
against an FP64 autograd of the reference formula (S <= 4097; larger S has no FP64 golden here).

## Findings

1. **A deterministic diffusers reference exists and is repeatable.** Every gradient that is not repeatable
   in native diffusers (`table`, `shift`, `scale`, `gate`, and the `temb`, `weight`, `bias` of #489; all go
   through `index_select`'s atomic backward) is repeatable under deterministic mode, at S = 257, 4097
   and 32768. Gradients that were already repeatable are unchanged by deterministic mode (det == native).
2. **The kernel does not match it bitwise, with a few exceptions.**
   - Bitwise equal to the deterministic reference: `d_residual` and `d_sublayer` of #488 (all S), `b2` of
     #484 at S = 4, `b` of #485 at S = 4.
   - Not bitwise equal: every other gradient. The fraction of differing elements is 26-43% for `dx`,
     `dweight` (#487, #489), 70-100% for the gradients that go through the gather (`table`, `shift`,
     `scale`, `gate`, `temb`, `weight`, `bias`), and about 100% for the FP32 MLP gradients.
3. **The difference is not noise around a correct answer; the reference is the less accurate one.**
   For the gather-path gradients the deterministic diffusers reference stays as inaccurate as the native
   one, because it still accumulates BF16 values in a fixed order:

   | Gradient (S = 4097) | kernel error | deterministic diffusers error | native diffusers error |
   |---|---|---|---|
   | #486 `table` | 2.4e-3 | 5.5e-2 | 5.9e-2 |
   | #487 `shift` | 2.9e-3 | 5.9e-2 | 6.4e-2 |
   | #488 `gate` | 2.7e-3 | 5.7e-2 | 5.0e-2 |
   | #489 `temb` | 1.5e-3 | 5.0e-2 | 4.8e-2 |
   | #489 `weight` | 4.3e-3 | 8.0e-2 | 8.7e-2 |

   For gradients without the gather (`dx`, `dweight`, the MLP and projection weights) all three are at the
   same BF16/FP32 rounding level (kernel 0.8-1.4x of diffusers).
4. **Where the reductions split.** Equal-shape element-wise gradients (`d_residual`, `d_sublayer`) agree
   bit for bit. Every gradient that sums over rows (weights, biases, table, shift/scale/gate, temb) is a
   reduction whose order differs: the kernel reduces in FP32 in a fixed tree, diffusers reduces in BF16
   through `index_add` or cuBLAS. The first elements to differ are in those reductions; the
   element-wise paths never differ.

## Consequence for the PR text

- "Backward is repeat-bitwise and within the contract against an FP64 golden" is the right contract; a
  bitwise-to-diffusers backward is not achievable and not desirable (matching it would mean reproducing
  a 10-30x larger error).
- The new evidence to add: a deterministic diffusers reference is repeat-bitwise but 10-30x less accurate on the
  gather-path gradients than the kernel.

Raw data: `bwd_<op>/<op>__summary.json` (per size and gradient), `<op>__nat.json` and `__det.json`.
