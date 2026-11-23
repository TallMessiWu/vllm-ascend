# SPDX-License-Identifier: Apache-2.0
"""Layer 23 reference routing for the single-request Qwen3 MoE diagnostic."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.models.qwen3_moe_routing_probe import ObservedRouter, Qwen3MoeRoutingProbeForCausalLM
from vllm_ascend.observability.moe_routing import (
    ATTENTION_BANDS,
    DECODE_ROWS,
    ORACLE_LAYER,
    attention_projection_names,
    validate_routes,
)

ATTENTION_WEIGHTS = attention_projection_names((ORACLE_LAYER,))


def load_reference_attention(path, names):
    root = Path(path).resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in sorted(names):
        shard = (root / index[name]).resolve()
        if not shard.is_relative_to(root):
            raise ValueError("Reference attention shard escapes the checkpoint")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weights[name] = handle.get_tensor(name)
    return weights


class Layer23OracleRouter(ObservedRouter):
    def __init__(self, base, trace):
        super().__init__(base, trace)
        self.reference_ids = torch.zeros_like(trace.ids)
        self.reference_weights = torch.zeros_like(trace.weights)
        self.enabled = torch.zeros((), dtype=torch.bool, device=trace.ids.device)

    def arm(self, reference):
        self.enabled.zero_()
        if reference is None:
            return
        ids, weights = reference["ids"], reference["weights"]
        validate_routes(ids, weights, self.trace.num_experts)
        if ids.shape != self.reference_ids.shape or weights.dtype != self.reference_weights.dtype:
            raise ValueError("Reference routes do not match the layer 23 buffers")
        self.reference_ids.copy_(ids)
        self.reference_weights.copy_(weights)
        self.enabled.fill_(True)

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        weights, ids = self.base._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)
        # The same device cursor drives the trace write below. Keep the enabled
        # tensor and lookup storage stable across capture and request resets.
        positions = self.trace.counts[0].to(torch.int64) + torch.arange(ids.shape[0], device=ids.device)
        positions = positions % self.trace.capacity
        ids = torch.where(self.enabled, self.reference_ids.index_select(0, positions).to(ids.dtype), ids)
        weights = torch.where(self.enabled, self.reference_weights.index_select(0, positions), weights)
        self.trace.record(hidden_states, router_logits, weights, ids)
        return weights, ids


class Qwen3MoeL23OracleForCausalLM(Qwen3MoeRoutingProbeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.attention_reference = getattr(config, "routing_probe_l23_attention_source", None)
        self.attention_band = getattr(config, "routing_probe_attention_band", None)
        runner = self.model.layers[ORACLE_LAYER].mlp.experts
        router = Layer23OracleRouter(runner.router.base, self.routing_traces[ORACLE_LAYER])
        runner.router = router
        runner.routed_experts.router = router
        self.oracle_router = router
        self.continuation_logits = None

    def arm_l23_trial(self, reference):
        if self.attention_reference is not None and reference is not None:
            raise ValueError("Attention fallback requires native expert selection")
        self.oracle_router.arm(reference)
        self.continuation_logits = []

    def load_weights(self, weights):
        if self.attention_reference is None:
            return super().load_weights(weights)
        names = (
            ATTENTION_WEIGHTS
            if self.attention_band is None
            else attention_projection_names(ATTENTION_BANDS[self.attention_band])
        )
        replacements = load_reference_attention(self.attention_reference, names)
        seen, changed = set(), set()

        def overlay():
            for name, original in weights:
                if name not in replacements:
                    yield name, original
                    continue
                replacement = replacements[name]
                if (
                    name in seen
                    or original.shape != replacement.shape
                    or original.dtype != torch.bfloat16
                    or replacement.dtype != original.dtype
                ):
                    raise ValueError(f"Invalid attention replacement: {name}")
                seen.add(name)
                if not torch.equal(original.cpu(), replacement):
                    changed.add(name)
                yield name, replacement

        loaded = super().load_weights(overlay())
        if seen != names or not changed:
            raise ValueError("Attention replacement was incomplete or had no numerical effect")
        self.routing_config["attention_fallback_loaded"] = sorted(seen)
        self.routing_config["attention_fallback_changed"] = sorted(changed)
        return loaded

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        # Runner V2 applies the LM head in sample_tokens, outside forward graph replay.
        if self.continuation_logits is not None:
            if logits is None or logits.ndim != 2 or logits.shape[0] != 1:
                raise ValueError("Expected one raw continuation logit row per step")
            self.continuation_logits.append(logits.detach().clone())
        return logits

    def take_continuation_logits(self):
        chunks, self.continuation_logits = self.continuation_logits, None
        self.oracle_router.enabled.zero_()
        if chunks is None or len(chunks) != DECODE_ROWS + 1:
            raise ValueError("Missing or extra continuation logit rows")
        return torch.cat(chunks).float().cpu()
