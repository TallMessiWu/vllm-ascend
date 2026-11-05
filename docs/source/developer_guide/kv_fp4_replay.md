# Offline FP4 KV sensitivity replay

This tool implements local K-only, V-only and K+V interventions for five FP4
recipes. It is **not** a model capture hook, native attention implementation,
GPQA runner, packed cache or performance benchmark. Real capture integration
and full-model evaluation are separate work. It does not modify a service.

The numerical module is `vllm_ascend.quantization.kv_replay`. It inherits the
[FP4 codec contract](kv_fp4_reference.md), including explicit fixed E4M3 outer
scales. The standalone CPU tool is `examples/offline_kv_fp4_replay.py`; it loads
only the numerical modules from its own checkout, without importing framework
initialization or torch_npu. Requires existing PyTorch and safetensors.

## Snapshot contract

One safetensors file contains exactly:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `q` | `[S, Hq, D]` | Selected query rows, after the transforms used by attention |
| `k`, `v` | `[T, Hkv, D]` | Complete single-sequence KV prefix, absolute positions `[0,T)` |
| `query_positions` | `[S]`, int64 | Absolute positions of sampled queries, not sample row numbers |

Q/K/V share BF16 or FP32 dtype. `Hq` must be divisible by `Hkv`. K is post-RoPE
and post-normalization, matching the cache writer. This is the ordinary causal
Qwen3 GQA contract with scale `1/sqrt(D)`, no sliding window, sparse/indexer
selection, ALiBi, score soft-cap, cross attention or packed multiple requests.
Do not relabel another model's snapshot to bypass these restrictions.

Required string metadata:

- `contract=qwen3-causal-gqa-post-rope-v1`
- `source_kind=captured` or `synthetic`
- `layer`: decimal layer index in `[0,128)`

Provenance is **caller-declared**, not independently authenticated. The tool does
not print prompts, answers, token IDs or input tensor values. It rejects missing
or unsupported contracts, non-finite values, invalid positions/heads, empty
samples, checkpoints with extra tensor keys and files over 512 MiB. It supports
up to 128 selected queries, 40960 KV positions, 128 heads and head dimension 512
per snapshot. These are diagnostic input bounds, not server scheduling changes.

## Execution and output

From the checkout, with a real snapshot and independently fixed outer scales:

```bash
python3 examples/offline_kv_fp4_replay.py \
  --input path/to/snapshot.safetensors \
  --k-outer-scale K_SCALE --v-outer-scale V_SCALE
```

Replace the symbolic scale arguments with validated positive numbers; the tool
never estimates scales from the input or silently uses activation calibration.
There is no built-in real dataset or automatic snapshot collection. In
particular, a synthetic fixture is not promoted to real-model evidence.

The FP32 mathematical attention reference materializes one sampled query's
scores at a time. CPU FP64 reductions report max absolute error, RMSE, relative
L2, SQNR and zero fraction for K, V and the attention output, with per-head
statistics available through the Python API. Zero denominators have explicit
string values, not arbitrary epsilons or non-standard JSON Infinity/NaN.

The compact text report includes all 15 interventions and the three worst
attention heads per intervention. Default output is execution-relative
`outputs/kv-replay.txt`; `--output` must stay inside the execution directory and
cannot overlap the input. Publication is atomic and does not follow existing
output hard links into input/weight files. Failures publish no new report and
return nonzero; an existing report then belongs to an earlier run.

`COMPLETED` means numerical replay completed. It does not mean acceptable quality,
validated capture semantics, native-device equivalence, long-context task
success or memory/performance improvement. Real tensor capture and a fixed W/A
full-model control remain necessary before making those claims.
