# Reference routing diagnostics

`moe_routing_check.py` is an opt-in, single-rank Qwen3 MoE diagnostic, not an
accuracy benchmark, router calibration, or quantized-model comparison.
It targets the vLLM 0.32.0 interfaces paired with the Ascend baseline
`11c7d88b10db11a7d3f2f747256af2537c1f73cf`.

## Workload and phases

The local tokenizer prepares four explicitly synthetic, repeated text fixtures.
Each has 512 prompt tokens and 33 forced sampled tokens. Only the first 32
sampled tokens are subsequently forwarded, giving exactly 512 prefill and 32
actual decode routing positions. The final sampled token has no routing data.
There is no chat-template, reasoning-quality, long-context, or performance claim.

Run these phases in separate processes in the order below, using the same local
BF16 reference checkpoint, editable Ascend source, input manifest, and output
directory. The workspace launcher supplies these arguments and explicit device
selection; the example does not choose or clean a device.

```bash
python examples/moe_routing_check.py --help
```

1. `prepare`: CPU tokenization, writing the fixed-token manifest locally.
2. `reference`: the unchanged Qwen architecture, native routed-expert IDs,
   two sequential repeats of all four requests. This is the baseline for
   repeatability and the additional probe's effect. Native ID capture remains
   enabled in both reference and probe processes.
3. `probe`: a separate model process, with only the architecture registration
   overridden to `Qwen3MoeRoutingProbeForCausalLM`. Delegates the original gate,
   router, expert computation and weight loader; records bounded per-layer
   device buffers. IDs and actual kernel merge weights cover all 48 layers;
   hidden states, logits, and MoE outputs cover layers 0, 23, and 47.
   Idle RPCs reset counters after warmup and read them after each request.
   Matching-shape NPU eager gate/top-k replay runs at that idle boundary.
4. `analyze`: CPU-only analysis of local snapshots. Reads the source router
   parameters directly from the existing sharded checkpoint; never exports,
   converts, or writes model parameters. Compares reference repeats, native IDs
   with internal snapshots, probe effects, captured versus NPU eager arithmetic,
   and a separate CPU FP64 mathematical gate/softmax reference.

The two model processes use BF16 model/KV, FULL_DECODE_ONLY, TP/DP/PP/PCP/DCP=1,
EP enabled, no EPLB/SP/LoRA/speculation, prefix caching disabled, and chunked
prefill enabled. Model limits remain 40960 context, 100 sequences, 16384 batched
tokens, and 0.95 memory utilization. Requests are intentionally sequential;
these are not changes to a running service's concurrency configuration.

## Lifecycle and interpretation

The diagnostic uses a unique model registry entry and router composition; it
does not patch the normal architecture, model runner, gate arithmetic, source
checkpoint, build files, dependencies, or process-global routing methods.
Buffers are allocated before warmup/capture and live for the model lifetime.
Device counts and per-slot visits reject zero coverage, extra/missing rows,
padding, overflow, and stale snapshots. CPU reads and saving happen only after
an idle synchronization. A graph replay is not inferred from a Python hook
count: the report requires 32 device executions beyond host router callbacks
per measured request/layer, as well as complete aligned native IDs.

- Sets are compared without treating a change of order as an expert change.
  Merge weights are aligned by logical expert ID. Boundary ties are reported.
- The FP64 oracle preserves the BF16 parameter/input values. It measures
  mathematical error and is not a bitwise-equivalent NPU kernel oracle.
- Nonidentical reference/probe routing, captured/eager logits or merge weights
  require review, not a relaxed accuracy threshold. Missing graph replay
  evidence also returns nonzero. This conservative gate does not itself prove
  a framework bug or harmful routing drift.
- `READY_FOR_CANDIDATE` means this reference observation workload is ready for
  an explicitly specified candidate. Model quality, quantization drift,
  routing-recovery benefit, and performance remain untested.

Raw token manifests and tensors stay local. Standard output contains aggregate
records prefixed `MOE1`; it does not print input text or complete answers.
The diagnostic artifacts contain activations and routing coefficients, not a
copy of model parameters. Never submit raw artifacts to a public issue.

## Tests and remaining hardware evidence

CPU unit tests are in `tests/ut/observability/test_moe_routing*.py`. They include
literal metric/softmax examples, fault rejection, inference-tensor reset,
changing-input CPU fullgraph buffers, and mocked upstream interface contracts.
The NPU component test is
`tests/e2e/pull_request/one_card/test_moe_routing_trace.py`; it checks changing
inputs in raw ACL replay, not a compiled full model. Full model loading, actual
NPU collection, graph integration and self-comparison require the example to
run on the target system. CPU/mock tests do not establish those results.
