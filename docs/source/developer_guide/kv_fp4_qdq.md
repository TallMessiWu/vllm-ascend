# Experimental model KV QDQ (BF16 cache)

This opt-in experiment uses the [FP4 reference recipes](kv_fp4_reference.md) to
change newly produced K/V numerical values. Persistent cache storage, native
attention arithmetic and allocation remain BF16. It is **not** packed-cache
support, a performance feature or proof of native FP4 equivalence.

## Selection and controls

Pass an explicit configuration, for example:

```text
--additional-config '{"kv_fp4_qdq":{"format":"mx4","target":"kv"}}'
```

Formats: `g16-b`, `g32-b`, `g16-e`, `g32-e`, `mx4`. Targets: `off`, `k`, `v`, `kv`.
There is no implicit experimental format/target. `kv_fp4_qdq=null` (the default)
leaves ordinary backend selection unchanged. Unknown settings are rejected.

Use `target=off` as the **same-backend control**, not a score from another
framework version or attention backend. Selecting the experiment explicitly
selects the ordinary Ascend GQA backend family and the QDQ subclass, not an
alternate FA3/draft backend. Off mode returns the original K/V objects unchanged.
Existing fusion guards requiring the exact base implementation may opt out for
the subclass; therefore use the same subclass for control and interventions.

E4M3 formats additionally require explicit positive `k_outer_scale` and
`v_outer_scale`. These are fixed global K/V scalars for this first model path,
not automatically calibrated per-layer values. Do not copy activation or weight
scales, fit them on the evaluation set, or change them during an engine's
lifetime. The replay recipe and model recipe must match. No new default
calibration corpus or arbitrary production scale is supplied.

## Supported experimental scope

- Plain Qwen3 MoE, BF16 model and cache, single TP/DP/PP/PCP/DCP rank.
- No W/A quantization config (use a BF16 numerical carrier), LoRA, speculation,
  KV transfer, prefix-cache sharing, forced/dynamic expert load balancing.
- Causal decoder GQA without sliding windows, ALiBi, attention sinks, logit
  soft-capping or shared KV layers; not MLA, sparse attention or M3/Pro.
- Existing service lengths, concurrency and graph configuration are not changed.
  Hardware/graph correctness still needs real validation.

This initial branch does not combine the separate activation-QDQ implementation
with KV QDQ. It isolates KV error with BF16 activations. Final candidates must
be evaluated again under the intended W/A configuration.

## Integration and graph lifetime

The additional-config field is owned by `AscendConfig`; the existing platform
selector chooses `AscendKVFP4QDQBackend`. Its implementation inherits the ordinary
GQA implementation. No vLLM source, model runner, global monkey patch, native
code, dependencies or build settings are modified.

`reshape_and_cache` transforms new K/V after model-side normalization/RoPE and
passes **the same transformed tensors** to the parent writer and its return path.
Thus both direct prefill and later cached attention consume the QDQ values. Only
changing the write call while returning original K/V would miss direct prefill.
The split cache-update method applies the same transform; the backend explicitly
keeps forward-owned cache updates so ordinary forward does not double-quantize.
Past cache entries are not requantized. The implementation does not directly
quantize queries or change router/model weights; downstream hidden states,
queries and routing may still change because of KV error. Unsupported sharing
modes are rejected instead of overwriting slots.

Recipes are validated and immutable before warmup/capture. Forward uses only
shape checks and device arithmetic; no device `.item()`, `.cpu()`, value-derived
Python state, or activation caching is introduced. Temporary FP32 work is tiled
at 1024 token/head rows without changing scheduler budgets; this is not an HBM
peak guarantee. Replay must still be tested
with new input values, positions, page reuse and request/batch changes. CPU
Dynamo or mock writer results do not establish that NPU replay works.

## Validation status

CPU tests cover the numerical codec, configuration/rejection rules, the actual
selector method in isolation, off-mode identity, identical transformed values
at cache/direct-prefill seams, independent K/V targeting, changing inputs and
layout rejection before writes. The parent writer is a mock; complete framework
initialization/loading and native kernels are **not** tested by these cases.

Before interpreting a quality score: verify the small-tensor codec on NPU,
verify actual backend selection and same-backend off-mode output, then compare
one intervention at a time under the same runtime and protocol. Eager/ACL,
complete loader/model, long-context correctness, quality and performance remain
unverified until real-device results are supplied. This experimental fork is
not an upstream release; a user-facing release fragment is required if submitted
as an upstream feature PR.

### Compiled numerical component checks

Direct ACL capture of eager operators does not exercise compiler graph rewrites.
Two additional small-tensor cases isolate the MX4 bit-view/exponent/scale path and
the actual tiled QDQ helper, with G32-B as a control:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/e2e/pull_request/one_card/test_kv_fp4_codec.py \
  KVFP4DeviceTests.test_mx4_compiled_scale_trace \
  KVFP4DeviceTests.test_qdq_npugraph_ex_matches_reference
```

Select an assigned NPU and normally stop your previous model service/worker.
No model or weights are loaded. The test uses `npugraph_ex.get_npu_backend` with the service options
`force_eager=True`, `inplace_pass=False`, `clone_input=False`, `clone_output=False`.
Here `force_eager` executes the optimized FX graph before external capture; it
does not bypass `torch.compile`. Backend invocations are counted explicitly.
No static/super kernel option is enabled and no compiler or model-runner patch is
installed by the test.

It compares CPU, NPU eager, compiled execution and compiled-plus-ACL replay at
zero tolerance. Each graph keeps a stable input buffer, warmed on the capture
stream before capture; replay changes input values and is checked before another
compiled call could reuse its output buffers. Shapes include decode, prefill and
the 1024-row tile boundary. Failure stops the current case, so later input cases
may remain unverified. Missing compiler/device is an error, not a skip.

A passing component result does not establish correctness of the complete model,
its fusion/partition context, cache lifecycle, or all stream/concurrency patterns.
Keep full-model graph/eager validation and quality evaluation separate.
