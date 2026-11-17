# Fixed-token MXFP4 weight routing comparison

This diagnostic compares a source-preserving BF16-DQ reference with an independently
copied MXFP4-W QDQ checkpoint stored as BF16. Both use the same registered routing
probe, native V2 fixed continuation, prefix reset and named worker RPCs as
[the reference diagnostic](moe_routing_check.md).
No runtime quantization, model-runner patch, activation/KV change or native low-bit
kernel is introduced. No build/dependency changes or rebuild are required.

## Candidate contract

The candidate has the same model config, index, tokenizer, generation settings and
BF16 storage layout as its reference. Recode only originally quantized attention
and expert linear weights; preserve router, embedding, normalization and output
head parameters. An external offline converter must first make a complete
independent copy; never alter or hard-link the reference.

The first recipe is group32 along logical K, E2M1, E8M0, RNE, exponent
`max(floor(log2(amax)) - 2, -127)`, no outer scale, followed by BF16 RNE storage.
It follows [Microsoft microxcaling at a fixed revision](https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py)
with `round='even'`, `flush_fp32_subnorms=False`, and `custom_cuda=False`.
Values outside the element range saturate. This is a defined numerical candidate,
not a claim of equivalence to an unspecified production conversion or native kernel.

The driver requires `.mxfp4_w_metadata/manifest.json` with source/output identity,
recipe, config/index SHA256, source/output shard size/mtime signatures and a
positive changed-value count. Reject incomplete conversions. Launch checks are
identity checks, not a new full payload hash pass; offline conversion remains a
separate responsibility.

## Run

Use a workspace launcher to set explicit device visibility, relative output/cache
paths and process isolation. The launcher invokes these phases in order:

```text
prepare -> reference -> candidate -> candidate-repeat -> analyze
```

Every invocation supplies `--model`, `--candidate`, `--source`, `--inputs`, and
`--output`. Outputs must remain relative to the execution directory. Collection
starts and shuts down only its own engine, one process at a time. Each side
records its own input hash; analysis rejects a different token/model identity.

The workload is four synthetic repeated-text samples, 512 prefill rows and 32
actual decode rows, single rank, FULL_DECODE_ONLY, original resource budgets.
Both sides enable native routed-expert output and use the same probe architecture.
This is a smoke fixture, not a representative calibration set.

## Results and boundaries

- Per-layer prefill/decode: expert-set overlap/equality, merged-weight difference,
  retained reference weight, load change and up to eight changed absolute positions.
- Candidate repeat: identical selected sets and merged weights are required. Its
  nondeterminism is an execution review, not attributed to quantization.
- Layers 0/23/47: hidden/logits/total MoE output differences, top-k margin, captured
  vs same-shape NPU eager arithmetic and independent CPU FP64 mathematical reference.
- Device/host counts, full coverage and native IDs establish bounded replay evidence;
  only a real NPU run can establish that evidence. CPU tests simulate the boundary.
- `DRIFT_MEASURED` means measurement completed, even when expert choices changed.
  `REVIEW_EXECUTION` is nonzero for unstable repeats, arithmetic or replay issues.
  No selected-set change in this fixture does not imply unchanged quality.

Local output differences do not isolate routing damage from changed expert
computation. Fixed-candidate expert/route oracles remain separate future work.
Model quality, recovery, long contexts, multi-rank execution, capacity and
performance remain unverified. Raw token/tensor snapshots stay local; share only
bounded aggregate diagnostics, never original prompts, responses or parameters.
