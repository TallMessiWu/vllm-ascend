# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Offline single-sequence causal GQA reference. Never used by a live backend.

Q is sampled at explicit absolute token positions. K/V contain the complete
prefix [0,T), after the transforms used by the cache writer (including K RoPE).
There is no sliding window, cross attention, sparse/indexer masking or multi-
request packing. Unsupported contracts must be rejected, not silently replayed.
"""

import math

import torch

from .kv_fp4 import FP4Recipe, qdq_fp4


def _validate(q, k, v, query_positions):
    if any(t.ndim != 3 for t in (q, k, v)) or v.shape != k.shape:
        raise ValueError("Expected Q[S,Hq,D], K/V[T,Hkv,D] for a single sequence")
    if any(d == 0 for t in (q, k, v) for d in t.shape):
        raise ValueError("Replay requires nonempty samples")
    if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1]:
        raise ValueError("Incompatible GQA head counts/dimensions")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("Q/K/V must share device and dtype")
    if q.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Replay supports BF16/FP32 only")
    if query_positions.shape != (q.shape[0],) or query_positions.dtype != torch.int64:
        raise ValueError("Expected int64 absolute positions for sampled queries")
    if query_positions.device != q.device:
        raise ValueError("Positions must be on the tensor device")
    if not bool(((query_positions >= 0) & (query_positions < k.shape[0])).all()):
        raise ValueError("Query positions are outside the captured KV prefix")
    if any(not bool(torch.isfinite(t).all()) for t in (q, k, v)):
        raise ValueError("Cannot replay non-finite input")


def attention_reference(q, k, v, query_positions):
    """FP32 attention output, with one sampled query's scores materialized at once.

    A synchronous diagnostic reference, not native attention or a throughput
    benchmark. Validates tensor values; never call from capture/forward hooks.
    """
    _validate(q, k, v, query_positions)
    heads, dimension = q.shape[1:]
    kv_heads = k.shape[1]
    keys, values = k.float(), v.float()
    key_positions = torch.arange(k.shape[0], device=q.device)
    outputs = []
    for row in range(q.shape[0]):
        query = q[row].float().reshape(kv_heads, heads // kv_heads, dimension)
        scores = torch.einsum("hgd,khd->hgk", query, keys) * dimension**-0.5
        scores = scores.masked_fill(key_positions > query_positions[row], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hgk,khd->hgd", probabilities, values).reshape(heads, dimension))
    result = torch.stack(outputs)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite attention output")
    return result


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Synchronous, FP64 CPU statistics; no raw tensor values are returned."""
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("Metrics require aligned nonempty tensors")
    before = reference.detach().to(device="cpu", dtype=torch.float64)
    after = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool((torch.isfinite(before) & torch.isfinite(after)).all()):
        raise ValueError("Cannot summarize non-finite tensors")
    error = after - before
    signal = before.square().sum().item()
    noise = error.square().sum().item()
    # Explicit strings avoid non-standard JSON Infinity/NaN and arbitrary epsilons.
    relative = math.sqrt(noise / signal) if signal else (0.0 if noise == 0 else "undefined_zero_reference")
    if signal and noise:
        sqnr = 10 * math.log10(signal / noise)
    else:
        sqnr = "+inf" if signal else ("-inf" if noise else "undefined_both_zero")
    return {
        "max_abs": error.abs().max().item(),
        "rmse": math.sqrt(noise / before.numel()),
        "relative_l2": relative,
        "sqnr_db": sqnr,
        "zero_fraction": (after == 0).double().mean().item(),
    }


def compare_kv(q, k, v, query_positions, k_recipe: FP4Recipe, v_recipe: FP4Recipe) -> list[dict]:
    """Fixed Q/positions with K-only, V-only and K+V interventions.

    Output is local numerical sensitivity, not accumulated full-model quality.
    Per-head summaries are retained; physical bytes and performance are not
    inferred from a high-precision QDQ tensor.
    """
    if (k_recipe.group_size, k_recipe.scale_type) != (v_recipe.group_size, v_recipe.scale_type):
        raise ValueError("K/V must use the same candidate format; outer scales may differ")
    baseline = attention_reference(q, k, v, query_positions)
    k_low, v_low = qdq_fp4(k, k_recipe), qdq_fp4(v, v_recipe)
    results = []
    for mode, keys, values in (("k", k_low, v), ("v", k, v_low), ("kv", k_low, v_low)):
        output = attention_reference(q, keys, values, query_positions)
        results.append(
            {
                "mode": mode,
                "k": error_metrics(k, keys),
                "v": error_metrics(v, values),
                "attention": error_metrics(baseline, output),
                "k_heads": [error_metrics(k[:, h], keys[:, h]) for h in range(k.shape[1])],
                "v_heads": [error_metrics(v[:, h], values[:, h]) for h in range(v.shape[1])],
                "attention_heads": [error_metrics(baseline[:, h], output[:, h]) for h in range(q.shape[1])],
            }
        )
    return results
