# PR #538 H100 summary

head: `2d9680d7b8b9c0f7d150bd7cb296cf496652963a` (stacked on #537)

pytest and acceptance chain same as #537; `validate_adaln_repair` **qualified** on H100-94GB.

Long-sequence cases S=4096 / 6889 / 6032 covered in fixed-mode audit JSON under `acceptance/fixed/`.

Within-H100 invariance only; no cross-architecture byte equality vs B200.
