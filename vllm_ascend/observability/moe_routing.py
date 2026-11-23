# SPDX-License-Identifier: Apache-2.0
"""CPU routing analysis and device-owned, bounded diagnostic snapshots.

No vLLM or NPU import. Each diagnostic model owns its buffers; no mutable
process-global state, host reads, file I/O or synchronization in record().
"""

import torch

PREFILL_ROWS = 512
DECODE_ROWS = 32
TRACE_ROWS = PREFILL_ROWS + DECODE_ROWS
ORACLE_LAYER = 23
ATTENTION_BANDS = {"layers_0_7": range(0, 8), "layers_8_15": range(8, 16), "layers_16_22": range(16, 23)}


def attention_projection_names(layers):
    return frozenset(f"model.layers.{layer}.self_attn.{proj}_proj.weight" for layer in layers for proj in "qkvo")


class RoutingTrace:
    """Single-request, single-rank trace. Buffers live for the model lifetime.

    Warmup is deliberately allowed to wrap; idle reset starts a measured
    request. Counts and visit multiplicities then make overflow/padding fatal.
    record() is device-only; Python calls are evidence, not a replay counter.
    """

    def __init__(self, capacity, hidden_size, num_experts, top_k, device, detailed=False):
        if min(capacity, hidden_size, top_k) < 1 or num_experts <= top_k:
            raise ValueError("Invalid routing trace dimensions")
        self.capacity, self.num_experts = capacity, num_experts
        self.detailed = detailed
        self.counts = torch.zeros(4, dtype=torch.int32, device=device)
        self.output_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.visits = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.output_visits = torch.zeros_like(self.visits)
        self.ids = torch.full((capacity, top_k), -1, dtype=torch.int32, device=device)
        self.weights = torch.zeros(capacity, top_k, dtype=torch.float32, device=device)
        self.hidden = torch.zeros(capacity, hidden_size, dtype=torch.bfloat16, device=device) if detailed else None
        self.logits = torch.zeros(capacity, num_experts, dtype=torch.float32, device=device) if detailed else None
        self.output = torch.zeros_like(self.hidden) if detailed else None
        self.host_calls = 0

    def record(self, hidden, logits, weights, ids):
        if hidden.dtype != torch.bfloat16 or logits.dtype != torch.float32 or weights.dtype != torch.float32:
            raise ValueError("Expected BF16 hidden and FP32 logits/routing weights")
        if (
            ids.ndim != 2
            or weights.shape != ids.shape
            or hidden.shape[0] != ids.shape[0]
            or logits.shape[0] != ids.shape[0]
        ):
            raise ValueError("Misaligned routing tensors")
        n = ids.shape[0]
        indices = (self.counts[0].to(torch.int64) + torch.arange(n, device=ids.device)) % self.capacity
        self.ids.index_copy_(0, indices, ids.to(torch.int32))
        self.weights.index_copy_(0, indices, weights.to(torch.float32))
        self.visits.index_add_(0, indices, torch.ones(n, dtype=torch.int32, device=ids.device))
        if self.detailed:
            self.hidden.index_copy_(0, indices, hidden)
            self.logits.index_copy_(0, indices, logits.to(torch.float32))
        self.counts[0].add_(n)
        self.counts[1].add_(1)
        self.counts[2 if n > 1 else 3].add_(n)
        self.host_calls += 1

    def record_output(self, output):
        if output.dtype != torch.bfloat16 or output.ndim != 2:
            raise ValueError("Expected BF16 two-dimensional MoE output")
        n = output.shape[0]
        indices = (self.output_rows.to(torch.int64) + torch.arange(n, device=output.device)) % self.capacity
        self.output.index_copy_(0, indices, output)
        self.output_visits.index_add_(0, indices, torch.ones(n, dtype=torch.int32, device=output.device))
        self.output_rows.add_(n)

    @torch.inference_mode()
    def reset(self):
        self.counts.zero_()
        self.output_rows.zero_()
        self.visits.zero_()
        self.output_visits.zero_()
        self.ids.fill_(-1)
        self.weights.zero_()
        self.host_calls = 0

    @torch.inference_mode()
    def snapshot(self, prefill_rows, decode_rows):
        expected = prefill_rows + decode_rows
        counts = self.counts.cpu().tolist()
        if expected != self.capacity or counts != [expected, 1 + decode_rows, prefill_rows, decode_rows]:
            raise ValueError(f"Missing/extra routing rows or calls: {counts}, expected={expected}")
        if not torch.all(self.visits.cpu() == 1):
            raise ValueError("Routing buffer contains missing/overwritten rows")
        ids, weights = self.ids.cpu().clone(), self.weights.cpu().clone()
        validate_routes(ids, weights, self.num_experts)
        result = {
            "ids": ids,
            "weights": weights,
            "device_calls": counts[1],
            "host_calls": self.host_calls,
            "prefill_rows": counts[2],
            "decode_rows": counts[3],
        }
        if self.detailed:
            if int(self.output_rows.cpu()[0]) != expected or not torch.all(self.output_visits.cpu() == 1):
                raise ValueError("Missing/extra MoE output rows")
            for name in ("hidden", "logits", "output"):
                value = getattr(self, name).cpu().clone()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite {name}")
                result[name] = value
        return result


def validate_routes(ids, weights, num_experts):
    if ids.device.type != "cpu" or weights.device.type != "cpu":
        raise ValueError("Analysis requires explicit CPU snapshots")
    if ids.ndim != 2 or not ids.numel() or ids.shape != weights.shape:
        raise ValueError("Nonempty [tokens, top_k] ids/weights required")
    if ids.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Expert ids must be integers")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonfinite or negative routing weights")
    if (ids < 0).any() or (ids >= num_experts).any():
        raise ValueError("Invalid expert id")
    ordered = ids.sort(dim=-1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("Repeated expert id in one token")


def tensor_error(actual, reference):
    if actual.device.type != "cpu" or reference.device.type != "cpu" or actual.shape != reference.shape:
        raise ValueError("Aligned CPU tensors required")
    if not actual.numel() or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Empty/nonfinite arithmetic comparison")
    delta = actual.double() - reference.double()
    norm = reference.double().norm().item()
    return {
        "exact": torch.equal(actual, reference),
        "abs_max": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": delta.norm().item() / norm if norm else None,
    }


def router_reference(hidden, weight, top_k, renormalize):
    if hidden.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("Router mathematical reference is CPU-only")
    logits = hidden.double() @ weight.double().t()
    if not torch.isfinite(logits).all() or not logits.numel():
        raise ValueError("Empty/nonfinite router reference")
    scores = logits.softmax(dim=-1)
    weights, ids = scores.topk(top_k, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    ordered = logits.sort(dim=-1, descending=True).values
    margin = ordered[:, top_k - 1] - ordered[:, top_k]
    return (
        logits,
        ids,
        weights,
        {
            "margin_min": margin.min().item(),
            "margin_median": margin.median().item(),
            "exact_boundary_ties": int((margin == 0).sum()),
        },
    )


def selection_metrics(ref_ids, ids, num_experts):
    for selected in (ref_ids, ids):
        validate_routes(selected, torch.ones_like(selected, dtype=torch.float32), num_experts)
    if ref_ids.shape != ids.shape:
        raise ValueError("Routing samples must have identical aligned shapes")
    matches = (ref_ids[:, :, None] == ids[:, None, :]).any(dim=-1)
    ref_load = torch.bincount(ref_ids.long().flatten(), minlength=num_experts).double()
    load = torch.bincount(ids.long().flatten(), minlength=num_experts).double()
    return {
        "tokens": ids.shape[0],
        "overlap_mean": matches.double().mean().item(),
        "set_equal_rate": matches.all(dim=-1).double().mean().item(),
        "load_total_variation": (ref_load - load).abs().sum().item() / (2 * ids.numel()),
        "max_expert_load": int(load.max()),
    }


def routing_metrics(ref_ids, ids, ref_weights, weights, num_experts):
    validate_routes(ref_ids, ref_weights, num_experts)
    validate_routes(ids, weights, num_experts)
    if ref_ids.shape != ids.shape:
        raise ValueError("Routing samples must have identical aligned shapes")
    matches = (ref_ids[:, :, None] == ids[:, None, :]).any(dim=-1)
    ref_dense = torch.zeros(ref_ids.shape[0], num_experts, dtype=torch.float64)
    dense = torch.zeros_like(ref_dense)
    ref_dense.scatter_(1, ref_ids.long(), ref_weights.double())
    dense.scatter_(1, ids.long(), weights.double())
    ref_load = torch.bincount(ref_ids.long().flatten(), minlength=num_experts).double()
    load = torch.bincount(ids.long().flatten(), minlength=num_experts).double()
    return {
        "tokens": ids.shape[0],
        "overlap_mean": matches.double().mean().item(),
        "set_equal_rate": matches.all(dim=-1).double().mean().item(),
        "retained_reference_weight_mean": (ref_weights.double() * matches).sum(dim=-1).mean().item(),
        "weight_l1_mean": (ref_dense - dense).abs().sum(dim=-1).mean().item(),
        "load_total_variation": (ref_load - load).abs().sum().item() / (2 * ids.numel()),
        "max_expert_load": int(load.max()),
    }
