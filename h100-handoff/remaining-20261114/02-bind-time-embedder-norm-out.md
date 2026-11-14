# 02. Bind `time_embedder` and `norm_out` to their rows in strict mode

On top of item 01 (new branch from `fix/h3-merged-rows-review-v2`, e.g. `feat/h3-strict-time-embedder-norm-out`). It is a
separate PR, because it changes model-level numbers that item 01 measures.

## Why

`rl_engine/validation/models/h3_full_model.py` binds six boundary modules to rows in strict mode
(`STRICT_BOUNDARY_ROWS`: `proj_in`, `proj_out`, `audio_proj_in`, `audio_proj_out`, `context_embedder`, `token_refiner`).
Two are still on the provider (`PROVIDER_BOUNDARIES = ("time_embedder", "norm_out")`), and
`docs/operators/h3-full-transformer.md` "Limits" says so. Both have merged rows of their own:

| Module | Row | PR | Op (`KernelRegistry().get_op(row, device="cuda")`) |
|---|---|---|---|
| `time_embedder` | `timestep_mlp_fp32` | #484 | `op.forward(x, w1, b1, w2, b2)` (`rl_engine/backends/cuda/model_specific/minimax_h3/timestep_mlp.py`) |
| `norm_out` | `final_adaln_out` | #489 | `op.forward(x, norm_weight, temb, weight, bias, timestep_indices, eps)` (`.../final_adaln_out.py`) |

Once both are bound, the only provider arithmetic left in a strict full-model run is `time_proj` (its own row
`timestep_sinusoid_h3`, #483; bind it too if its signature fits, see below).

## Provider signatures (read them in `<PROVIDER>`, the pinned `f53d5520`)

File: `<PROVIDER>/src/diffusers/models/transformers/transformer_minimax_h3.py`. A diffusers 0.40.0 copy on another
machine had these. **Check that `f53d5520` matches before relying on them.**

```python
# MiniMaxH3Transformer3DModel.forward
temb = self.time_proj(timestep)
temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))      # diffusers TimestepEmbedding, FP32
...
hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))

# MiniMaxH3AdaLayerNormOut
def __init__(self, hidden_size, time_embed_dim, eps):
    self.norm = nn.RMSNorm(hidden_size, eps=eps)          # eps = config.final_norm_eps
    self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
def forward(self, hidden_states, temb, timestep_indices):
    shift, scale = self.linear(silu(temb).to(dtype(self.linear))).chunk(2, dim=-1)
    hidden_states = self.norm(hidden_states)
    return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)
```

`TimestepEmbedding.forward(sample, condition=None)` runs `linear_1`, `act` (SiLU), `linear_2`. A non-`None` `condition`
or a `cond_proj` / `post_act` in the module means the row does not apply: refuse it, don't ignore it.

## Change

1. `h3_full_model.py`:
   - move `time_embedder` and `norm_out` into `STRICT_BOUNDARY_ROWS` and leave `PROVIDER_BOUNDARIES` empty (or remove it,
     and also the test that unions the two);
   - in `_strict_boundaries`, add two bindings that follow the `project()` pattern: save `(module, module.forward)` in
     `originals`, set an instance `forward`, and restore with `module.__dict__.pop("forward", None)`;
   - `time_embedder`: `forward(sample, condition=None)` → refuse a `condition`, then
     `op.forward(sample, m.linear_1.weight, m.linear_1.bias, m.linear_2.weight, m.linear_2.bias)`;
   - `norm_out`: `forward(hidden_states, temb, timestep_indices)` →
     `op.forward(x, m.norm.weight, temb, m.linear.weight, m.linear.bias, timestep_indices, m.norm.eps)`. The provider
     passes `(B, S, 5376)`. `validate_h3_final_adaln_out` (`rl_engine/reference/minimax_h3/final_adaln_out.py`) accepts
     `x.dim() >= 2` with `x.shape[-2] == len(timestep_indices)`, int64 indices and a `(T, D)` `temb`, so no reshape
     should be needed. Do not change the op;
   - on `test-h3`, `_strict_boundaries(model, trace, ops, native=frozenset())` leaves the modules named in `native` on the
     provider (they are left out of `row_ops`). The two new bindings must honour `native` the same way. Also,
     `for name in row_ops: project(name)` treats every entry as a linear projection, so keep the two new modules out of
     that loop;
   - `trace.boundary_bindings` then reports `status="row"` for both.
2. `tests/models/minimax_h3/test_h3_full_model.py`: extend
   `test_strict_boundaries_route_every_projection_through_its_row` (CPU, fake ops via `monkeypatch` on
   `KernelRegistry.get_op`) to cover the two new bindings, the argument order, the eps passed, a refused `condition`, and
   the restore on exit.
3. `docs/operators/h3-full-transformer.md`: the boundary list and "Limits" (drop "`time_embedder` and `norm_out` use the
   provider in this harness").
4. Check that `tools/validation/models/h3_full_transformer.py --provider-boundaries` still leaves **all** boundary modules
   on the provider. That is the B200 path, and it must not reach the SM90-only rows.

## Run

```bash
git worktree add --detach <SRC>/bind02 feat/h3-strict-time-embedder-norm-out
scripts/run_h100.sh tests     <SRC>/bind02 <OUT>/02/tests
scripts/run_h100.sh mypy      <SRC>/bind02 <OUT>/02/mypy
scripts/run_h100.sh fulltrans <SRC>/bind02 <OUT>/02/fulltrans
scripts/run_h100.sh dual      <SRC>/bind02 <OUT>/02/dual
```

**Expected:**
- tests: no failure beyond item 01's `followup` tree;
- `fulltrans`: candidate still bitwise repeatable, and `boundary_bindings` shows every module as `row`;
- `dual`: strict stays at item 01's result (0/16 expected).

Item 01's strict numbers are the reference: the output and input-gradient columns move again, because `temb` and the
final norm change arithmetic.

## Send back

The four output folders, the diff stat, and `SUMMARY.md` comparing `fulltrans` / `dual` with item 01.
