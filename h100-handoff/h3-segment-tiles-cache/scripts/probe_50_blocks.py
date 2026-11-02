"""50 chained candidate blocks (real graph + real CUDA ops, random weights) sharing one index."""
import sys, time, collections
sys.path.insert(0, ".")
import torch
from rl_engine.backends.extension import _C
import rl_engine
from rl_engine.backends.cuda.model_specific.minimax_h3 import adaln_row_gather as arg
from rl_engine.validation.models import h3_block, h3_provider as prov
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout

print(rl_engine.__file__, _C.__file__)
dev, H, NB, DISTINCT = "cuda", 5376, 50, 3
LAYOUT = sys.argv[1] if len(sys.argv) > 1 else "tiny"
spec = H3_BLOCK_LAYOUTS[LAYOUT]
packed = h3_fl2va_layout(num_text_tokens=spec["num_text_tokens"], num_latent_frames=spec["num_latent_frames"],
                         latent_height=spec["latent_height"], latent_width=spec["latent_width"],
                         num_audio_latents=spec["num_audio_latents"], device=dev)
S = packed["position_ids"].shape[0]
index = packed["timestep_indices"] * 3 + packed["token_tags"]  # built once, like the model's adaln_indices
print("layout", LAYOUT, "S", S, "table rows", int(index.max()) + 1)

g = torch.Generator(device="cpu").manual_seed(0)
def rnd(*shape, scale=0.02, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g) * scale).to(dtype).to(dev)
shapes = {"adaln_proj.linear.weight": (18 * H, 2688), "norm1.weight": (H,), "attn.to_q.weight": (7168, H),
          "attn.to_k.weight": (7168, H), "attn.to_v.weight": (7168, H), "attn.norm_q.weight": (128,),
          "attn.norm_k.weight": (128,), "attn.to_out.0.weight": (H, 7168), "norm2.weight": (H,),
          "ff.net.0.proj.weight": (2 * 14336, H), "ff.net.2.weight": (H, 14336)}
weight_sets = []
for _ in range(DISTINCT):
    w = {f"{h3_block.BLOCK}.{n}": rnd(*s) for n, s in shapes.items()}
    w[f"{h3_block.BLOCK}.adaln_proj.linear.bias"] = rnd(18 * H)
    for n in list(w):
        if n.endswith(("norm1.weight", "norm2.weight", "norm_q.weight", "norm_k.weight")):
            w[n] = (1 + w[n].float()).to(torch.bfloat16)
    weight_sets.append(w)
temb = torch.randn(h3_block.NUM_TIMESTEPS, 2688, generator=g).to(dev)
hidden0 = rnd(1, S, H, scale=1.0)
rope = prov.provider_rope(packed["position_ids"])
ops = h3_block.CandidateOps()

builds = collections.Counter()
real = arg._segment_tiles
def counting(idx, rows):
    builds[(tuple(idx.shape), rows)] += 1
    return real(idx, rows)
arg._segment_tiles = counting

def run(clear_each_call):
    builds.clear(); arg.clear_segment_tiles_cache()
    if clear_each_call:  # emulate the pre-change behaviour: every consumer rebuilds
        orig = arg._cached_segment_tiles
        arg._cached_segment_tiles = lambda s, t, b: b()
        import rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm as m1, \
               rl_engine.backends.cuda.model_specific.minimax_h3.gate_residual as m2
        # _repeated_segment_tiles calls arg._cached_segment_tiles via module global -> patched above
    x = hidden0.clone().requires_grad_()
    params = []
    h = x
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(NB):
        w = {k: v.detach().clone().requires_grad_() if False else v for k, v in weight_sets[i % DISTINCT].items()}
        ctx = dict(weights=w, hidden=h, temb=temb, index=index, rope=rope)
        h = h3_block.run_graph("candidate", ctx, ops)[h3_block.OUTPUT]
    torch.cuda.synchronize(); t1 = time.perf_counter()
    h.float().sum().backward()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    if clear_each_call:
        arg._cached_segment_tiles = orig
    return x.grad.clone(), sum(builds.values()), dict(builds), (t1 - t0) * 1e3, (t2 - t1) * 1e3

for w in weight_sets:   # leaves need no grad here: only the activation path matters for tile builds
    pass
grad_c, n_c, b_c, f_c, bw_c = run(False)
grad_u, n_u, b_u, f_u, bw_u = run(True)
grad_c2, n_c2, _, f_c2, bw_c2 = run(False)
print(f"cached   : {n_c} tile builds {b_c}  fwd {f_c:.0f} ms  bwd {bw_c:.0f} ms (2nd run: {n_c2} builds, bwd {bw_c2:.0f} ms)")
print(f"uncached : {n_u} tile builds {b_u}  fwd {f_u:.0f} ms  bwd {bw_u:.0f} ms")
print("input grad bitwise equal cached vs uncached:", torch.equal(grad_c, grad_u), "| cached repeat:", torch.equal(grad_c, grad_c2))
print("peak mem GiB", torch.cuda.max_memory_allocated() / 2**30)
