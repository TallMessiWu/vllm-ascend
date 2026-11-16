# SPDX-License-Identifier: Apache-2.0
"""Idle-only RPC helpers, safe to import without importing a model class."""

from pathlib import Path

import torch

from vllm_ascend.observability.moe_routing import DECODE_ROWS, PREFILL_ROWS, TRACE_ROWS


@torch.inference_mode()
def reset_probe(model):
    torch.npu.synchronize()
    if len(model.routing_traces) != 48:
        raise ValueError("Missing routing trace layers")
    for trace in model.routing_traces:
        trace.reset()
    torch.npu.synchronize()
    return {"armed_layers": len(model.routing_traces)}


@torch.inference_mode()
def save_probe(model, path):
    """Idle-only RPC: synchronize, replay matching gate shapes, and save locally."""
    if not isinstance(path, str):
        raise ValueError("Snapshot destination must be a relative string path")
    dest = Path(path)
    if (
        dest.is_absolute()
        or ".." in dest.parts
        or dest.exists()
        or not dest.resolve().is_relative_to(Path.cwd().resolve())
    ):
        raise ValueError("Trace destination must be a new execution-relative file")
    torch.npu.synchronize()
    result = {"config": model.routing_config, "layers": []}
    for layer_id, trace in enumerate(model.routing_traces):
        row = trace.snapshot(PREFILL_ROWS, DECODE_ROWS)
        runner = model.model.layers[layer_id].mlp.experts
        row["gate_type"] = type(runner.gate).__name__
        row["parameter_dtype"] = str(runner.gate.weight.dtype)
        if trace.detailed:
            logits, ids, weights = [], [], []
            # Preserve the measured prefill/decode GEMM shapes; a single 544-row
            # replay would change the gate's batching and is not this oracle.
            chunks = [trace.hidden[:PREFILL_ROWS], *trace.hidden[PREFILL_ROWS:].split(1)]
            for hidden in chunks:
                gate_out = runner.gate(hidden)
                logit = gate_out[0] if isinstance(gate_out, tuple) else gate_out
                weight, selected = runner.router.base._compute_routing(hidden, logit, None)
                logits.append(logit)
                ids.append(selected)
                weights.append(weight)
            row["eager_logits"] = torch.cat(logits).cpu()
            row["eager_ids"] = torch.cat(ids).cpu()
            row["eager_weights"] = torch.cat(weights).cpu()
        result["layers"].append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, dest)
    return {"saved": str(dest), "layers": len(result["layers"]), "rows_per_layer": TRACE_ROWS}


class MoeRoutingWorkerExtension:
    """Named worker RPCs; only primitive arguments/metadata cross the wire.

    The framework imports this extension by qualified name before constructing
    the worker. Functions stay local to that worker; no pickle fallback or
    model-runner/global patch is needed.
    """

    def moe_routing_arm(self):
        self._moe_routing_armed = False
        result = reset_probe(self.get_model())
        self._moe_routing_armed = True
        return result

    def moe_routing_save(self, path):
        if not getattr(self, "_moe_routing_armed", False):
            raise ValueError("Snapshot requires a fresh arm before each measured request")
        self._moe_routing_armed = False
        return save_probe(self.get_model(), path)
