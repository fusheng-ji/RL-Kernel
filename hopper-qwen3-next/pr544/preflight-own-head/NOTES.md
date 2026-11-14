`preflight` at #544 head `81be0811` plus `hopper_preflight_2a8f18e4.patch` (applied as a separate step and committed locally on a throwaway commit so the launcher accepts the tree; not pushed), 4 x H100.
Device guards passed (6 tests); `rollout-smoke` FAILED with the same FlashInfer `tvm::ffi::Error ... __ffi_repr__ is already registered` abort as at the stack top. Unedited.
