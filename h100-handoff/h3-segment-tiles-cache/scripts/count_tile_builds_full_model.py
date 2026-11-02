"""Run the real full-model candidate pass and count how often the backward segment tiles are built.

Usage (from the repo root, same arguments as the ``h3_full_model`` candidate step of
``examples/models/minimax_h3/candidate.sbatch``):

    [H3_TILES_NO_CACHE=1] python count_tile_builds_full_model.py --checkpoint <ckpt> \
        --manifest rl_engine/validation/models/h3_full_manifest.json --mode candidate \
        --thresholds examples/models/minimax_h3/full-model-thresholds.json --output <out>

Writes ``<out>/tile_builds.json``. With the cache, a model that hands every block the same
``adaln_indices`` tensor builds once per distinct (index, table rows); ``H3_TILES_NO_CACHE=1``
forces every consumer to rebuild, which is the behaviour before the change. Nothing in the
repository is modified: the counters are installed on the module in this process only.
"""

import atexit
import collections
import json
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from rl_engine.backends.cuda.model_specific.minimax_h3 import adaln_row_gather as arg  # noqa: E402

builds, lookups = collections.Counter(), collections.Counter()
real_tiles, real_cached = arg._segment_tiles, arg._cached_segment_tiles


def counting_tiles(index, num_rows):
    builds[f"index_len={index.numel()},table_rows={num_rows}"] += 1
    return real_tiles(index, num_rows)


def counting_cached(sources, tag, build):
    lookups[str(tag)] += 1
    if os.environ.get("H3_TILES_NO_CACHE") == "1":
        return build()
    return real_cached(sources, tag, build)


arg._segment_tiles, arg._cached_segment_tiles = counting_tiles, counting_cached
out = Path(sys.argv[sys.argv.index("--output") + 1])


@atexit.register
def _write():
    out.mkdir(parents=True, exist_ok=True)
    (out / "tile_builds.json").write_text(
        json.dumps(
            dict(
                cache_disabled=os.environ.get("H3_TILES_NO_CACHE") == "1",
                tile_builds=sum(builds.values()),
                by_shape=dict(builds),
                consumer_lookups=sum(lookups.values()),
                by_tag=dict(lookups),
            ),
            indent=2,
        )
    )


sys.argv = ["h3_full_model", *sys.argv[1:]]
runpy.run_module("tools.validation.models.h3_full_model", run_name="__main__")
