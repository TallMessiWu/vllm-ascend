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

### MX4 group-error scale selection

The default MX4 scale selection remains `ceil`. Opt in to the alternative with:

```text
--additional-config '{"kv_fp4_qdq":{"format":"mx4","target":"kv","mx4_scale_selection":"mse"}}'
```

This applies to **both K and V**, independently for each new token/head/group32.
It retains E2M1 RNE, E8M0 scale bytes, the post-normalization/RoPE writer boundary,
and BF16 cache/attention. It changes only how the shared scale is selected:

1. Construct the existing ceil exponent and its immediately smaller valid
   exponent (both clamped to the reference's E8M0 range).
2. Quantize/dequantize each candidate, including the final BF16 carrier rounding.
3. Compare per-group squared errors in FP32, normalized by the same clamped amax
   before squaring to limit overflow/underflow. Choose the smaller scale only
   for strictly smaller error; ties retain ceil. Zero/invalid-group rules remain.
4. Pass the selected QDQ values to both cache writing and direct prefill.

Selection is device arithmetic on each invocation/replay, not a Python decision
based on tensor data. It reuses the existing persistent tensor exponent shift,
adds no host reads/synchronization, and does not revisit historical cache.
There is no calibration corpus, fitted threshold, outer scale, additional cache
metadata, precision fallback, new environment variable or native build change.

The model option requires `target=kv`, without copy-only or the old trace
diagnostic. Its startup marker includes `scale_selection=mse`. The local error
criterion is not an attention-aware objective and does **not** guarantee task
quality. Candidate evaluation adds temporary work; no performance benefit or
native-kernel equivalence is claimed.

The focused device regression is
`KVFP4DeviceTests.test_mx4_mse_opaque_compile_replay` in
`tests/e2e/pull_request/one_card/test_kv_fp4_codec.py`. It covers the literal
scale-jump counterexample, changed inputs, non-contiguous views and the
1024-row tile boundary under eager/opaque/ACL execution. It is a synthetic
component check, not full model graph or long-context acceptance. CPU/mock
tests alone do not establish NPU correctness.

### Value-preserving copy diagnostic

`diagnostic_copy` defaults to `false` and must be a JSON boolean. Opt in with:

```text
--additional-config '{"kv_fp4_qdq":{"format":"mx4","target":"kv","diagnostic_copy":true}}'
```

For each selected side, this calls `clone(memory_format=torch.contiguous_format)`
instead of QDQ. Values and BF16 dtype are retained, but the writer and returned
prefill path receive a new tensor with PyTorch-contiguous layout (not a guarantee
about NPU internal format). Unselected sides are returned
unchanged; `target=off` still returns both original objects. The format/recipe is
validated but not used for numerical conversion in copy-only mode. Startup logs
explicitly label `operation=copy-only (no quantization)`.

This is a diagnostic, not a repair, FP4 quality result or performance feature.
It compares identity/off, value-preserving materialization and lossy QDQ at the
same existing backend seam. A copy-only failure does not require FP4 numerical
error; a success does not exclude QDQ-specific allocation/lifetime issues,
because cloning does not reproduce all FP32 workspaces, tiling or timing.
No input mutation, cached tensors, host reads, graph-mode change, new native op
or model-runner patch is introduced. Copies are device operations recorded for
replay; real model graph validation remains required.

### Bounded replay seam diagnostic

`diagnostic_trace` defaults to `false`. It accepts a JSON boolean and requires
ordinary `mx4`, `target=kv`, with copy-only disabled. This temporary diagnostic
uses the existing worker extension `vllm_ascend.attention.kv_fp4_trace.KVTraceWorkerExtension`.
The local-loopback diagnostic server can use vLLM's existing development RPC
to call `kv_trace_control` with `arm` before requests and `report` after requests.
It needs no model-runner patch, native build or new dependency.

Each layer allocates one fixed snapshot in one-token warmup, retaining the
buffers for capture/replay. Captured device operations copy raw K/V, QDQ output,
the writer's effective slot and attention output; no host tensor read or
synchronization is added there. Only one-token graphs are sampled. Arm clears
device counters after startup; an idle-worker report synchronizes and compares
the last replay's same-input QDQ with NPU eager and CPU executions of the tested
codec, and compares the writer slot with the QDQ snapshot, using exact BF16 bits.
The eager computation uses the same flattened row shape on a contiguous snapshot;
it does not reproduce the original input's internal NPU format. Invalid slots,
missing layers, missing arm or zero replays cannot pass. Attention is checked
for finiteness only, not against an eager attention oracle.

This can distinguish a sampled QDQ discrepancy from a sampled writer discrepancy.
It does not cover prefill, earlier steps, other batches, complete attention,
model quality or performance. Copies and gathers perturb allocation/timing;
a normal answer under instrumentation is not a repair. Remove the diagnostic
when the investigation closes. The default experiment does not enable it or
the development RPC. Test-only CPU mocks do not validate NPU graph capture.

The detailed diagnostic (`trace_version=2`) additionally observes ten codec
stages in the first decoder layer only: FP32 blocks, amax, raw scale, scale bits,
exponent, E8M0 scale, final scale, normalized values, FP4 codes and FP32 output.
An optional device-only codec observer copies values to warmup-allocated buffers;
it does not change the reference formula. All other layers retain the seam
snapshots. The first layer also checks raw input before/after QDQ, and the idle
RPC directly compares CPU and NPU eager results rather than equating mismatch
counts. Missing stages are unverified. A changed input invalidates a simple
same-input interpretation; otherwise the earliest divergent stage narrows the
investigation. Extra copies still perturb timing and allocation, so disappearance
of a discrepancy is not a repair. This mode uses the same opt-in trace setting.

The next diagnostic (`trace_version=3`) tests an equivalent materialization:
the FP32 scale's int32 interpretation is cloned to owning device storage before
the unchanged exponent formula consumes it. This isolates reliance on shared
float/integer storage; it is not proof of a particular native-kernel defect or
a validated NPU repair. The later E8M0 scale construction is unchanged.
Materialization also applies to ordinary MX4 execution; only snapshots remain opt-in.
Reports now separate `first_replay_divergence` (replay versus NPU eager) from
`first_oracle_divergence` (NPU eager versus CPU). The old `first_divergence` name
aliases the former. Cross-device FP32 intermediate differences alone do not
mark a graph failure when final quantized values agree. An exponent-difference
histogram is read on CPU only at the idle reporting boundary.

`trace_version=4` leaves that active expression unchanged and adds shadow-only
integer sub-operation checks in the detailed layer. The same live bits feed
scalar and persistent-device-Tensor operand paths: right shift, exponent mask,
mantissa mask, nonzero comparison, bool-to-int32 conversion and addition.
Constants are allocated during warmup and refreshed in place by idle arm RPC;
their values are checked before interpreting the Tensor control. Snapshots and
constants retain their addresses. Each shadow path is compared with NPU eager
and CPU, as well as the active expression's exponent. Missing steps or corrupt
constants are unverified; no shadow value feeds the active codec. These extra
operations perturb timing and cannot establish a model repair. The client keeps
global layer/seam coverage while expanding only the detailed layer.

### Active MX4 tensor shift

The model backend now owns one immutable device int32 `[23]` per layer. Its first
eager warmup initializes the constant with a blocking CPU-to-device copy before
marking it ready; initialization during capture is rejected. Every subsequent
tile/request reuses the same Tensor address as the exponent right-shift operand.
No per-step host read, transfer or synchronization is added. Masks, rounding and
E8M0 reconstruction are unchanged. Off/copy-only/other formats do not allocate it.
This path also applies with diagnostic tracing disabled and needs no model-runner
patch or native rebuild. Direct MX4 graph users of `qdq_fp4` must similarly pass
a prepared `exponent_shift`; omission retains the offline/eager scalar reference.

The repair-validation report (`trace_version=5`) reads and validates every
serving shift buffer without refreshing it, then compares active QDQ/cache values.
Known-failing scalar shadows are disabled by default; the legacy shadow helper
remains an explicit internal test option. Passing first-layer shadow tests alone
does not validate all layers, uninstrumented execution or model quality.

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

Recipes are validated and immutable before warmup/capture. After the one-time
eager shift initialization described above, forward uses only metadata checks
and device arithmetic; no device `.item()`, `.cpu()`, value-derived
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

The compilation boundary matters: `AscendPlatform.opaque_attention_op()` returns
true; vLLM calls `unified_attention_with_output`, whose implementation calls the
backend's `forward` and then `reshape_and_cache`. This experiment's QDQ is inside
that opaque op. Directly compiling `_qdq_tiled` does **not** reproduce this boundary.

The revised checks use a test-local opaque custom op with an explicit mutable
output buffer, keeping the real tiled QDQ helper inside it. They assert that
Dynamo hands one opaque call to the compiler, not inlined codec arithmetic.
No production op, backend, compiler or model-runner patch is installed.

Select an assigned NPU, normally stop your previous model service/worker, and run
each case in a fresh process so a capture failure cannot contaminate the next:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/e2e/pull_request/one_card/test_kv_fp4_codec.py \
  KVFP4DeviceTests.test_mx4_opaque_compile_replay
ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/e2e/pull_request/one_card/test_kv_fp4_codec.py \
  KVFP4DeviceTests.test_g32_b_opaque_compile_replay
ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/e2e/pull_request/one_card/test_kv_fp4_codec.py \
  KVFP4DeviceTests.test_mx4_scale_floor_eager
```

The first two use `npugraph_ex.get_npu_backend` with the service options
`force_eager=True`, `inplace_pass=False`, `clone_input=False`, `clone_output=False`.
Here `force_eager` does not bypass `torch.compile`. Cold warmup starts with zeros;
capture uses both zero and nonzero inputs, followed by changed-value replay.
Input/output buffers stay alive at stable addresses, warmup uses the capture
stream, and comparisons run outside capture before any subsequent output reuse.
Shapes include 1/33/257 tokens, offset/non-contiguous synthetic split-QKV views,
and the 1024-row tile boundary. MX4 additionally replays nonzero floor inputs to
regress the scale-value correction. GLOBAL capture mode and zero tolerance remain.

The scale check retains its historical `test_mx4_scale_floor_eager` entry name.
It checks the active helper, effective zero/nonzero groups, this reference's
scale bytes 1..254 against literal FP32 bits, byte round-trip, FP4 midpoint
payloads and independent readers. It also directly captures the int32 producer
and uint8 reader paths, replaying reversed/minimum/maximum/ascending exponents
to check the lifetime and freshness of the new int32-to-FP32 view.
Legacy exp2/clamp mismatches are printed as observations, not substituted into
the corrected oracle. Any active mismatch remains RED; CPU fault injection
retains the zero and previous-neighbor results as negative controls. Do not treat a scale fix as proof of
a model-level root cause. Failure within a case leaves later inputs unverified;
missing runtime is an error, not a skip.

These tests load no model or weights and do not execute native cache writes or
attention. Passing them does not validate the full worker, fusion/partition,
cache or stream/concurrency context. Keep model correctness, quality and
performance separate.
