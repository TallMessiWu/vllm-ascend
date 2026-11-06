# Experimental FP4 KV reference tools

This is a numerical and byte-layout reference for experiments, **not a registered
attention backend, native FP4 kernel or deployment-performance implementation**.
Importing it does not change an existing model or service. It adds no native code,
build configuration or dependencies. Hardware tests remain required.

## Exact recipes

`vllm_ascend.quantization.kv_fp4` provides `FP4Recipe`, `qdq_fp4`, `pack_fp4` and
`unpack_fp4`. Inputs are BF16/FP32 with groups along the final dimension. Each
row/head is grouped independently. Tail groups are zero padded, never joined to
the next row. K and V must be supplied separately.

All candidates use E2M1 magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, saturation and
round-to-nearest, ties-to-even. Signed zero is retained (including code 8).
The even element occupies the low nibble; the odd element occupies the high
nibble. The output of numerical QDQ is rounded back to the input dtype.

| Group | Stored scale | Scale selection |
| --- | --- | --- |
| 16 / 32 | BF16 | `RNE_BF16(max(amax / 6, 2^-126))` |
| 16 / 32 | positive E4M3FN byte | `RNE_E4M3(clamp(amax / (6 * outer_scale), 2^-9, 448))` |
| 32 | E8M0 byte | upward power-of-two of `max(amax / 6, 2^-126)` |

E4M3 requires an **explicit** positive FP32 outer scale, fixed for the stored
payload. It is not borrowed from activation/weight calibration and is never
updated based on future tokens. This reference does not select a calibration
corpus or claim that an arbitrary outer scale is suitable for real KV tensors.
Zero groups use raw scale 1 (before the E4M3 outer scale). These conventions are
part of the experiment and must be reproduced by a native reader/writer.

The E4M3 scale input uses FP32 tensor/tensor division with a same-shape divisor
containing `float32(6 * outer_scale)`. Do not replace it with multiplication by a
rounded reciprocal: at `outer_scale=float32(0.02)` and `amax=6`, the quotient
rounds to 50, then E4M3 RNE selects 48. A rounded FP32 reciprocal can instead
produce 50.000003814697266 and select 52. The device-side divisor fill is replayed
with the graph; it reads no device values on the host and caches no input values.
No epsilon, tie snapping or test tolerance is used. This changes the arithmetic
path, not the scale format, calibration or RNE contract. Hardware verification
is still required for the tensor-division path.

The upward E8M0 recipe is encoded from FP32 exponent/mantissa bits to avoid
`ceil(log2(x))` losing the distinction immediately above a power of two. The
finite exponent range is [-126, 127]. Group16 with E8M0 is not called MXFP4.

**Do not infer native equivalence from dtype names.** For example, at base commit
`11c7d88b10db11a7d3f2f747256af2537c1f73cf`, the existing
`ops/triton/quantize_mxfp4_indexer.py` uses a lower-code midpoint rule, unlike this
reference's RNE. The `kv_compress_epilog_v2` writer has its own BF16-scale recipe
and layout. Any reuse requires an explicit contract comparison, including ties,
scale selection, signed zero, clipping and row/page layout.

## Storage and execution boundaries

`PackedFP4` stores uint8 packed data, actual BF16 or uint8 scale tensors, logical
shape and a recipe. `payload_bytes` counts padded data and scales, plus four bytes
for the E4M3 outer scale. It does **not** count allocator alignment, cache pages,
replication, temporary workspace or metadata containers; it is not an HBM claim.
The outer scale may be shared by many rows in one payload. Merging payloads with
different outer scales requires preserving their separate metadata.

Packing and unpacking validate data and descriptors, and can synchronize the
host with the device. Use them only in offline/component experiments. Unpacking
rejects non-finite scale metadata and overflow in the requested output dtype.
`qdq_fp4` has no tensor-to-host reads, only shape-dependent Python decisions and
device operations. Non-finite inputs poison their group; it does not hide them
by producing finite packed values. QDQ still uses high-precision memory and
arithmetic, and has no native-cache savings or speed claim.

## Tests

CPU unit tests compare against literal bytes, independent scalar table search,
struct-based BF16 rounding, native CPU E4M3 casts and CPU Dynamo fullgraph with
changing values/shapes. Regression tests inject rounded-reciprocal scalar division
and check exact E4M3 scale bytes at midpoints and their neighboring FP32 inputs.
They do not import real torch_npu or prove NPU behavior.

In an already installed NPU environment, from the source worktree:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/e2e/pull_request/one_card/test_kv_fp4_codec.py
```

Select a device assigned to you. This uses small synthetic tensors, no weights,
model or requests. It checks packed bytes, tail groups, invalid values and QDQ
ACL capture/replay with changing inputs, separately against NPU eager and CPU.
An additional scale regression checks tensor division and nearby midpoints across
four synthetic outer scales. `DIAG legacy_scalar_raw` may still show differences
on the unused old path; only the active path must pass. Missing runtime/device is an error,
not a skip. The imported implementation must come from the same worktree as the
test. Passing this test is only component evidence; real KV distributions,
attention outputs, full-model quality and native packed-cache performance remain
separate experiments.
