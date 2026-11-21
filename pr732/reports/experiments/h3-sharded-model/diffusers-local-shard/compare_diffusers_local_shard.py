"""Bytes of h3_sharded_load.cut_shard against Diffusers 0.43's TP loader slicing (_local_shard).

For every transformer block of the released checkpoint, every TP size and every rank, the seven
stored shards of RL-Kernel's sharded model are compared with what Diffusers 0.43 reads for the same
rank under MiniMaxH3Transformer3DModel._tp_plan: ``colwise`` (to_q/k/v), ``rowwise`` (to_out.0,
ff.net.2) and ``PackedColwiseParallel([1, 1])`` (ff.net.0.proj, hidden then gate). Both read the
same safetensors slices; nothing is loaded in full. CPU only.

    python compare_diffusers_local_shard.py --rlk <RL-Kernel checkout> --checkpoint <transformer dir> --out <json>
"""

import argparse
import json
import sys
from pathlib import Path

import torch


class Mesh:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlk", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", default="2,4,8")
    args = ap.parse_args()
    sys.path.insert(0, args.rlk)

    import diffusers
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.hooks.tensor_parallel import (
        PackedColwiseParallel,
        _blocks_to_block_sizes,
        _local_shard,
    )
    from safetensors import safe_open

    from rl_engine.validation.models import h3_sharded_load as sl

    plan = MiniMaxH3Transformer3DModel._tp_plan
    # Diffusers style -> sharded dim of the weight; packed styles carry their proportional blocks
    style_dim = {"colwise": 0, "rowwise": 1}
    root = Path(args.checkpoint)
    weight_map = json.loads((root / sl.INDEX).read_text())["weight_map"]
    blocks = sorted({int(k.split(".")[1]) for k in weight_map if k.startswith(sl.BLOCKS + ".")})
    sources = sorted({src for src, _ in sl.SHARD_SOURCES.values()})
    compared, mismatched = 0, []
    for tp in (int(t) for t in args.tp.split(",")):
        for b in blocks:
            for src in sources:
                name = f"{sl.BLOCKS}.{b}.{src}"
                style = plan[f"{sl.BLOCKS}.*.{src.rsplit('.', 1)[0]}"]
                with safe_open(str(root / weight_map[name]), framework="pt") as f:
                    view = f.get_slice(name)
                    shape = tuple(view.get_shape())
                    for r in range(tp):
                        if isinstance(style, PackedColwiseParallel):
                            dim, sizes = 0, _blocks_to_block_sizes(shape[0], style.blocks)
                        else:
                            dim, sizes = style_dim[style], [shape[style_dim[style]]]
                        theirs = _local_shard(view, dim, sizes, Mesh(tp, r))
                        ours = [
                            sl.cut_shard(view, shape, how, tp, r).contiguous()
                            for shard_name, (s, how) in sl.SHARD_SOURCES.items()
                            if s == src
                        ]
                        mine = ours[0] if len(ours) == 1 else torch.cat(ours, dim=0)
                        compared += 1
                        if mine.dtype != theirs.dtype or not torch.equal(mine, theirs):
                            mismatched.append([tp, r, name])
        print(f"tp={tp}: {compared} compared, {len(mismatched)} mismatched so far", flush=True)
    report = {
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "blocks": len(blocks),
        "sources_per_block": sources,
        "tp": args.tp,
        "compared": compared,
        "mismatched": mismatched,
        "all_equal": not mismatched,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("diffusers", "blocks", "compared", "all_equal")}))


if __name__ == "__main__":
    main()
