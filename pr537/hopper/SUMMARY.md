# PR #537 H100 summary

head: `a73c66c5e2db7ec844e30ac763ce85d410179259`

| suite | passed | failed | skipped |
| --- | --- | --- | --- |
| pytest (2 GPUs) | 101 | 0 | 0 |
| acceptance `validate_adaln_repair` | qualified | — | — |

Within-H100 repeat/batch invariance only (BF16 `mma.sync`; bytes may differ from B200).

Model: Qwen/Qwen-Image @ `75e0b4be04f60ec59a75f475837eced720f823b6`.
